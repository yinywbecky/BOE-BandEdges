from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from boe_cashe import BPProcessor
from model_badgap_v1 import DualBadGapModelContainer, SubNet
from run_badgap import (
    ATOM_NUM, LABEL_COLUMNS, VEFeaturesBadgapDataset, convert_metrics_units, make_split_loaders,
    plot_test_prediction_grid, set_seed,
)


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual BOE models for VBM, CBM, and total energy.")
    parser.add_argument("--dataset-path1", type=Path, default=BASE_DIR / "low_temp_v3_1005" / "dataset1" / "dataset_normal")
    parser.add_argument("--dataset-path2", type=Path, default=BASE_DIR / "low_temp_v3_1005" / "dataset2" / "dataset_normal")
    parser.add_argument("--result-root", type=Path, default=BASE_DIR / "result_bandgap_v1")
    parser.add_argument("--ref-poscar", type=Path, default=BASE_DIR / "POSCAR_REV.vasp")
    parser.add_argument("--input-dim", type=int, choices=(30, 42), default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vbm-weight", type=float, default=1.0)
    parser.add_argument("--cbm-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, width, and learning rate must be positive.")
    set_seed(args.seed)
    reference_bytes = args.ref_poscar.read_bytes()
    labels = {}
    for split in ("train", "val", "test"):
        first = pd.read_csv(args.dataset_path1 / f"{split}_label.csv")[LABEL_COLUMNS]
        second = pd.read_csv(args.dataset_path2 / f"{split}_label.csv")[LABEL_COLUMNS]
        if not first.equals(second):
            raise ValueError(f"Mode 1 and mode 2 {split} labels differ.")
        labels[split] = first
    offsets = -labels["train"].mean().to_numpy(dtype=np.float32)
    loaders = [
        make_split_loaders(path, args.input_dim, args.batch_size, args.seed, args.num_workers, offsets)
        for path in (args.dataset_path1, args.dataset_path2)
    ]
    train1, val1, test1 = loaders[0]
    train2, val2, test2 = loaders[1]
    dataset1, dataset2 = train1.dataset, train2.dataset
    if not isinstance(dataset1, VEFeaturesBadgapDataset) or not isinstance(dataset2, VEFeaturesBadgapDataset):
        raise TypeError("Expected BOE training datasets.")
    model = DualBadGapModelContainer(
        SubNet(args.input_dim, 3, args.hidden_dim),
        SubNet(args.input_dim, 3, args.hidden_dim), device=args.device,
    )
    args.result_root.mkdir(parents=True, exist_ok=True)
    model1_path = args.result_root / "model1_badgap_v1.pth"
    model2_path = args.result_root / "model2_badgap_v1.pth"
    history, best_states = model.fit(
        train1, train2, epochs=args.epochs, lr=args.lr,
        vbm_weight=args.vbm_weight, cbm_weight=args.cbm_weight, energy_weight=args.energy_weight,
        val_loader1=val1, val_loader2=val2,
        checkpoint_paths=(str(model1_path), str(model2_path)),
    )
    composite1 = model1_path.with_stem(f"{model1_path.stem}_composite_best")
    composite2 = model2_path.with_stem(f"{model2_path.stem}_composite_best")
    model.load(str(composite1), str(composite2))
    model.save(str(model1_path), str(model2_path))
    pd.DataFrame(history).to_csv(args.result_root / "history_badgap_v1.csv", index=False)

    metrics_raw = {}
    for split, first, second in (("train", train1, train2), ("val", val1, val2), ("test", test1, test2)):
        metrics, frames = model.evaluate(first, second)
        metrics_raw[split] = metrics
        pd.DataFrame([convert_metrics_units(metrics)]).to_csv(args.result_root / f"{split}_metrics.csv", index=False)
        for name, frame in frames.items():
            frame.to_csv(args.result_root / f"{split}_{name}_pred.csv", index=False)
    plot_test_prediction_grid(model.predict(test1, test2), args.result_root / "test_pred_vs_true_3x1.png")

    np.savez(
        args.result_root / "feature_normalization.npz",
        mean1=dataset1.mean, std1=dataset1.std,
        mean2=dataset2.mean, std2=dataset2.std,
    )
    reference_path = args.result_root / "reference.vasp"
    if args.ref_poscar.resolve() != reference_path.resolve():
        shutil.copyfile(args.ref_poscar, reference_path)
    summary = {
        "model_module": "model_badgap_v1.py", "model_output_dim": 3,
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "seed": args.seed, "atom_num": ATOM_NUM, "boes_per_mode": 20,
        "dataset_path1": str(args.dataset_path1), "dataset_path2": str(args.dataset_path2),
        "model1": model1_path.name, "model2": model2_path.name,
        "feature_normalization": "feature_normalization.npz", "reference_structure": reference_path.name,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "boe_ordering_policy": BPProcessor.ORDERING_POLICY,
        "boe_neighbor_cutoff": BPProcessor.BOE_NEIGHBOR_CUTOFF,
        "weights": {"vbm": args.vbm_weight, "cbm": args.cbm_weight, "energy": args.energy_weight},
        "y_normalization": {
            "mode": "offset_by_train_mean", "vbm_mean": float(-offsets[0]),
            "cbm_mean": float(-offsets[1]), "energy_mean": float(-offsets[2]),
        },
        "metric_units": {
            "vbm_rmse": "meV", "cbm_rmse": "meV", "energy_rmse": "meV/atom",
            "vbm_bias": "meV", "cbm_bias": "meV", "energy_bias": "meV/atom",
        },
        "train_metrics_raw": metrics_raw["train"], "val_metrics_raw": metrics_raw["val"],
        "test_metrics_raw": metrics_raw["test"],
        "checkpoint_selection": {
            "split": "val", "criterion": "sum of VBM, CBM, and total-energy RMSE in eV",
            "best_epoch": best_states["composite"]["epoch"],
        },
    }
    (args.result_root / "summary_badgap_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model bundle and evaluation to {args.result_root}")
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
