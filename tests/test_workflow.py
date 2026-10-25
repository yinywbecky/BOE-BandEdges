import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

import gen_center_all_data as centers
import gen_force_all_data as generation
import plot_hist_v2 as sampling
from model_badgap_v1 import DualBadGapModelContainer
from predict_badgap_from_traj_v1 import (
    load_normalization, parse_xyz_frames, predict_trajectory, select_step_indices,
)
from run_badgap import FORCE_COLUMNS, convert_metrics_units, make_split_loaders
from run_bandgap_v1 import train


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "generated"
        self.enterContext(patch.object(sampling, "BASE_DIR", str(self.root)))
        self.enterContext(patch.object(sampling, "OUTPUT_DIR", str(self.output)))
        self.enterContext(patch.object(centers, "BASE_DIR", str(self.root)))
        self.enterContext(patch.object(generation, "BASE_DIR", str(self.root)))
        for temp in sampling.TEMPERATURES:
            directory = self.root / temp
            directory.mkdir()
            raw = np.arange(10000, dtype=float)
            np.savetxt(directory / "all_energy.txt", -430 + raw / 10000 + int(temp) / 100)
            np.savetxt(directory / "vbm.txt", raw / 10000)
            np.savetxt(directory / "cbm.txt", 1 + raw / 10000)

    def sample(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return sampling.build_sampling_table()[0]

    def test_sampling_counts_and_absolute_raw_indices(self):
        selected = self.sample()
        self.assertEqual(len(selected), 5300)
        for temp, count in sampling.SAMPLE_COUNTS.items():
            rows = selected.loc[selected.temperature_K == int(temp)]
            self.assertEqual(len(rows), count)
            indices = rows.frame_index.to_numpy()
            self.assertTrue(((indices >= 6000) & (indices < 10000)).all())
            self.assertTrue(((indices - 6000) % 2 == 0).all())
            energy = np.loadtxt(self.root / temp / "all_energy.txt")
            np.testing.assert_allclose(rows.energy_eV, energy[indices], rtol=0, atol=1e-8)
        table = centers.load_frame_table(str(self.output / sampling.OUTPUT_INDEX_CSV))
        self.assertEqual(sum(len(rows) for rows in table.values()), 5300)

    def test_legacy_indices_are_rejected(self):
        selected = self.sample()
        selected["frame_index"] -= 6000
        path = self.output / sampling.OUTPUT_INDEX_CSV
        selected.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "energy/frame mismatch"):
            centers.load_frame_table(str(path))

    def test_band_labels_follow_the_same_raw_frame(self):
        selected = self.sample().groupby("temperature_K").head(2)
        labels = generation.build_label_dataframe(selected)
        expected_vbm = selected.frame_index.to_numpy() / 10000
        np.testing.assert_allclose(labels.vbm, expected_vbm)
        np.testing.assert_allclose(labels.cbm, expected_vbm + 1)
        np.testing.assert_allclose(labels.gap, 1)
        np.testing.assert_allclose(labels.energy, selected.energy_eV)

    def test_band_label_length_mismatch_is_rejected(self):
        selected = self.sample()
        np.savetxt(self.root / "100" / "vbm.txt", [0.0])
        with self.assertRaisesRegex(ValueError, "one finite VBM and CBM"):
            generation.build_label_dataframe(selected)

    def test_structure_level_splits_are_disjoint_and_reproducible(self):
        boes = pd.DataFrame(np.zeros((2000, 42)))
        forces = pd.DataFrame(np.zeros((2000, 6)))
        labels = pd.DataFrame({"energy": np.arange(100)})
        with contextlib.redirect_stdout(io.StringIO()):
            first = generation.split_dataset_with_length(boes, forces, labels, str(self.root / "mode1"), str(self.root))
            second = generation.split_dataset_with_length(boes, forces, labels, str(self.root / "mode2"), str(self.root))
        self.assertEqual({split: len(values) for split, values in first.items()}, {"train": 70, "val": 15, "test": 15})
        joined = np.concatenate(list(first.values()))
        self.assertEqual(len(np.unique(joined)), 100)
        for split in first:
            np.testing.assert_array_equal(first[split], second[split])
            features = pd.read_csv(self.root / "mode1" / "dataset_normal" / f"{split}_ve.csv")
            self.assertEqual(len(features), 20 * len(first[split]))


def write_training_fixture(directory: Path) -> None:
    directory.mkdir()
    rng = np.random.default_rng(12)
    for split, count, offset in (("train", 2, 0), ("val", 1, 10), ("test", 1, 20)):
        features = rng.normal(size=(20 * count, 42)) + offset
        features[:, 0] = 0.25
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(42)])
        for name in FORCE_COLUMNS:
            frame[name] = 0.0
        frame.to_csv(directory / f"{split}_ve.csv", index=False)
        labels = pd.DataFrame({
            "vbm": np.arange(count) * 0.1 + offset + 1,
            "cbm": np.arange(count) * 0.1 + offset + 3,
            "energy": np.arange(count) * 0.2 + offset - 430,
        })
        labels.to_csv(directory / f"{split}_label.csv", index=False)


class ModelBundleTests(unittest.TestCase):
    def test_validation_statistics_come_only_from_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset"
            write_training_fixture(path)
            first, validation, test = make_split_loaders(path, 42, 2, 42, 0)
            np.testing.assert_array_equal(first.dataset.mean, validation.dataset.mean)
            np.testing.assert_array_equal(first.dataset.std, test.dataset.std)
            self.assertEqual(first.dataset.std[0], 1.0)
            self.assertGreater(float(validation.dataset.labels[0, 0]), 9)
            self.assertGreater(float(test.dataset.labels[0, 0]), 19)

    def test_one_epoch_synthetic_bundle_and_portable_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("dataset1", "dataset2"):
                write_training_fixture(root / mode)
            args = argparse.Namespace(
                dataset_path1=root / "dataset1", dataset_path2=root / "dataset2",
                result_root=root / "bundle", ref_poscar=ROOT / "POSCAR_REV.vasp",
                input_dim=42, hidden_dim=4, epochs=1, batch_size=2, lr=1e-3,
                seed=42, num_workers=0, vbm_weight=1.0, cbm_weight=1.0,
                energy_weight=0.1, device="cpu",
            )
            original_fit = DualBadGapModelContainer.fit

            def check_validation(container, first, second, **kwargs):
                self.assertGreater(float(kwargs["val_loader1"].dataset.labels[0, 0]), 9)
                self.assertLess(float(kwargs["val_loader1"].dataset.labels[0, 0]), 11)
                return original_fit(container, first, second, **kwargs)

            with patch.object(DualBadGapModelContainer, "fit", new=check_validation):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = train(args)
            self.assertEqual(summary["checkpoint_selection"]["split"], "val")
            self.assertEqual(summary["checkpoint_selection"]["best_epoch"], 1)
            self.assertEqual(summary["model_output_dim"], 3)
            root.joinpath("dataset1").rename(root / "unavailable_dataset1")
            root.joinpath("dataset2").rename(root / "unavailable_dataset2")
            reference = Structure.from_file(ROOT / "POSCAR_REV.vasp")
            frame = f"{len(reference)}\nfixed reference cell\n" + "".join(
                f"{site.specie.symbol} {site.x:.12f} {site.y:.12f} {site.z:.12f}\n"
                for site in reference
            )
            trajectory = root / "traj.xyz"
            trajectory.write_text(frame + frame, encoding="ascii")
            prediction_args = argparse.Namespace(
                model_dir=args.result_root, trajectory=trajectory, output=root / "prediction",
                ref_poscar=None, start=0, stop=None, stride=1, max_steps=None, device="cpu",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = predict_trajectory(prediction_args)
            self.assertEqual(list(result.columns), ["step", "vbm_eV", "cbm_eV", "bandgap_eV", "total_energy_eV"])
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result.bandgap_eV, result.cbm_eV - result.vbm_eV)
            self.assertTrue(np.isfinite(result.to_numpy()).all())
            for key in ("model1", "model2"):
                file = args.result_root / summary[key]
                state = torch.load(file, map_location="cpu", weights_only=True)
                torch.save({name: torch.zeros_like(value) for name, value in state.items()}, file)
            with contextlib.redirect_stdout(io.StringIO()):
                known = predict_trajectory(prediction_args)
            offsets = summary["y_normalization"]
            np.testing.assert_allclose(known.vbm_eV, offsets["vbm_mean"])
            np.testing.assert_allclose(known.cbm_eV, offsets["cbm_mean"])
            np.testing.assert_allclose(known.total_energy_eV, offsets["energy_mean"])

    def test_intensive_band_edge_units(self):
        converted = convert_metrics_units({"vbm_rmse": 1.0, "cbm_bias": 1.0, "energy_rmse": 1.0})
        self.assertEqual(converted, {"vbm_rmse": 1000.0, "cbm_bias": 1000.0, "energy_rmse": 12.5})

    def test_incomplete_xyz_and_invalid_ranges_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "truncated.xyz"
            file.write_text("2\ncomment\nP 0 0 0\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                parse_xyz_frames(file)
        with self.assertRaises(ValueError):
            select_step_indices(10, 0, None, 0, None)
        self.assertEqual(select_step_indices(10, 2, 9, 2, 2), [2, 4])

    def test_missing_normalization_is_not_reconstructed_from_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                load_normalization(Path(temporary) / "missing.npz", 42)


if __name__ == "__main__":
    unittest.main()
