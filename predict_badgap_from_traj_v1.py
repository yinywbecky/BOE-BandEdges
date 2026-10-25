from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from boe_cashe import BPProcessor, get_centers
from model_badgap_v1 import DualBadGapModelContainer, SubNet


def parse_xyz_frames(xyz_path: Path) -> tuple[list[str], list[np.ndarray]]:
    """Read standard multi-frame XYZ with a fixed species/atom order."""
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    species_ref = None
    frames = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        n_atoms = int(lines[index])
        if n_atoms <= 0 or index + n_atoms + 2 > len(lines):
            raise ValueError(f"{xyz_path}: invalid or incomplete frame at line {index + 1}.")
        species = []
        coordinates = []
        for line in lines[index + 2:index + n_atoms + 2]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"{xyz_path}: atom row needs species and XYZ coordinates.")
            species.append(fields[0])
            coordinates.append([float(value) for value in fields[1:4]])
        frame = np.asarray(coordinates, dtype=float)
        if not np.isfinite(frame).all():
            raise ValueError(f"{xyz_path}: non-finite coordinates.")
        if species_ref is None:
            species_ref = species
        elif species != species_ref:
            raise ValueError(f"{xyz_path}: species order changes between frames.")
        frames.append(frame)
        index += n_atoms + 2
    if species_ref is None or not frames:
        raise ValueError(f"{xyz_path}: no XYZ frames.")
    return species_ref, frames


def build_features(boe_atoms: np.ndarray, n_col: int) -> np.ndarray:
    atoms = boe_atoms.reshape(len(boe_atoms), -1)
    if n_col == atoms.shape[1]:
        return atoms.astype(np.float32)
    centers = get_centers(boe_atoms).reshape(len(boe_atoms), -1)
    combined = np.hstack((atoms, centers))
    if n_col != combined.shape[1]:
        raise ValueError(f"Cannot build {n_col} features from this BOE geometry.")
    return combined.astype(np.float32)


def select_step_indices(total: int, start: int, stop: int | None, stride: int, max_steps: int | None) -> list[int]:
    if start < 0 or stride <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("Start must be nonnegative; stride and max-steps must be positive.")
    end = total if stop is None else min(stop, total)
    if end <= start:
        raise ValueError("Requested step range contains no frames.")
    indices = list(range(start, end, stride))
    return indices if max_steps is None else indices[:max_steps]


def load_normalization(path: Path, input_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = tuple(np.asarray(archive[key], dtype=np.float32) for key in ("mean1", "std1", "mean2", "std2"))
    for index, values in enumerate(arrays):
        if values.shape != (input_dim,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid normalization array shape or values.")
        if index in (1, 3) and (values <= 0).any():
            raise ValueError(f"{path}: standard deviations must be positive.")
    return arrays[0], arrays[1], arrays[2], arrays[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict VBM, CBM, band gap, and total energy from a fixed-cell XYZ trajectory.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--ref-poscar", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def predict_trajectory(args: argparse.Namespace) -> pd.DataFrame:
    summary = json.loads((args.model_dir / "summary_badgap_v1.json").read_text(encoding="utf-8"))
    if summary["model_module"] != "model_badgap_v1.py" or summary["model_output_dim"] != 3:
        raise ValueError("Expected a three-output model_badgap_v1 model bundle.")
    n_col = int(summary["input_dim"])
    means1, std1, means2, std2 = load_normalization(
        args.model_dir / summary["feature_normalization"], n_col
    )
    ref_path = args.ref_poscar if args.ref_poscar is not None else args.model_dir / summary["reference_structure"]
    if hashlib.sha256(ref_path.read_bytes()).hexdigest() != summary["reference_sha256"]:
        raise ValueError("Reference structure differs from the model bundle.")
    if (
        summary["boe_ordering_policy"] != BPProcessor.ORDERING_POLICY
        or float(summary["boe_neighbor_cutoff"]) != float(BPProcessor.BOE_NEIGHBOR_CUTOFF)
    ):
        raise ValueError("BOE ordering policy or neighbor cutoff differs from training.")
    reference = Structure.from_file(ref_path)
    species, frames = parse_xyz_frames(args.trajectory)
    if species != [site.specie.symbol for site in reference]:
        raise ValueError("Trajectory species/atom count does not match the reference.")
    indices = select_step_indices(len(frames), args.start, args.stop, args.stride, args.max_steps)
    for mode in ("1", "2"):
        BPProcessor(reference.copy(), mode=mode, ref_struct=reference, use_cache=True)
    model = DualBadGapModelContainer(
        SubNet(n_col, 3, int(summary["hidden_dim"])),
        SubNet(n_col, 3, int(summary["hidden_dim"])), device=args.device,
    )
    model.load(str(args.model_dir / summary["model1"]), str(args.model_dir / summary["model2"]))
    y_norm = summary["y_normalization"]
    if y_norm["mode"] != "offset_by_train_mean":
        raise ValueError("Unsupported target normalization.")
    target_means = np.array([y_norm[key] for key in ("vbm_mean", "cbm_mean", "energy_mean")], dtype=float)
    if not np.isfinite(target_means).all():
        raise ValueError("Non-finite target normalization.")
    records = []
    with torch.inference_mode():
        for step in indices:
            fractional = reference.lattice.get_fractional_coords(frames[step])
            features = []
            for mode, mean, std in (("1", means1, std1), ("2", means2, std2)):
                boes = BPProcessor.fast_BOE(fractional, reference.lattice.matrix, mode)
                if len(boes) != int(summary["boes_per_mode"]):
                    raise ValueError("BOE count differs from training.")
                normalized = (build_features(boes, n_col) - mean) / std
                features.append(torch.from_numpy(normalized).to(model.device))
            scalar = 0.5 * (model.model1(features[0]).sum(dim=0) + model.model2(features[1]).sum(dim=0))
            vbm, cbm, energy = scalar.cpu().numpy().astype(float) + target_means
            if not np.isfinite([vbm, cbm, energy]).all():
                raise ValueError(f"Non-finite prediction at frame {step}.")
            records.append({
                "step": step, "vbm_eV": float(vbm), "cbm_eV": float(cbm),
                "bandgap_eV": float(cbm - vbm), "total_energy_eV": float(energy),
            })
    result = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "traj_bandgap_predictions.csv", index=False)
    output_summary = {
        "model_dir": str(args.model_dir), "trajectory": str(args.trajectory),
        "selected_step_count": len(result), "start": args.start, "stop": args.stop,
        "stride": args.stride, "bandgap_mean_eV": float(result["bandgap_eV"].mean()),
        "cell_assumption": "fixed lattice from the bundled reference structure",
    }
    (args.output / "traj_bandgap_summary.json").write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
    print(f"Saved {len(result)} trajectory predictions to {args.output}")
    return result


def main() -> None:
    predict_trajectory(parse_args())


if __name__ == "__main__":
    main()
