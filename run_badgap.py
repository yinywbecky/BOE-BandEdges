"""Shared dataset, normalization, and plotting helpers for run_bandgap_v1.py."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ATOM_NUM = 80
ENERGY_EV_TO_MEV_PER_ATOM = 1000.0 / ATOM_NUM
LABEL_COLUMNS = ["vbm", "cbm", "energy"]
FORCE_COLUMNS = ["fx0", "fy0", "fz0", "fx1", "fy1", "fz1"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VEFeaturesBadgapDataset(Dataset):
    """Keep each structure's BOEs together; forces are carried but not trained."""

    def __init__(
        self,
        ve_df: pd.DataFrame,
        label_df: pd.DataFrame,
        n_col: int,
        n_ve: int = 20,
        pre_mean: np.ndarray | None = None,
        pre_std: np.ndarray | None = None,
    ):
        if n_ve <= 0 or len(ve_df) % n_ve or len(label_df) != len(ve_df) // n_ve:
            raise ValueError("BOE and label rows must describe complete structures.")
        if label_df.empty or n_col not in (30, 42) or ve_df.shape[1] < n_col:
            raise ValueError("Expected a nonempty dataset with 30 or 42 input features.")
        missing = set(FORCE_COLUMNS).difference(ve_df.columns)
        missing_labels = set(LABEL_COLUMNS).difference(label_df.columns)
        if missing or missing_labels:
            raise ValueError(f"Missing force columns {sorted(missing)} or labels {sorted(missing_labels)}.")
        self.n_ve = n_ve
        self.length = len(label_df)
        self.features = np.ascontiguousarray(ve_df.iloc[:, :n_col].to_numpy(dtype=np.float32))
        self.forces = np.ascontiguousarray(ve_df[FORCE_COLUMNS].to_numpy(dtype=np.float32))
        self.labels = np.ascontiguousarray(label_df[LABEL_COLUMNS].to_numpy(dtype=np.float32))
        if not all(np.isfinite(values).all() for values in (self.features, self.forces, self.labels)):
            raise ValueError("Dataset contains non-finite values.")
        self.mean = np.zeros(n_col, dtype=np.float32)
        self.std = np.ones(n_col, dtype=np.float32)
        self.y_offsets = np.zeros(3, dtype=np.float32)
        if (pre_mean is None) != (pre_std is None):
            raise ValueError("Provide both feature mean and standard deviation.")
        if pre_mean is not None:
            self.normalize_X(pre_mean, pre_std)

    def normalize_X(self, pre_mean=None, pre_std=None) -> None:
        self.mean = (
            self.features.mean(axis=0, dtype=np.float64).astype(np.float32)
            if pre_mean is None else np.asarray(pre_mean, dtype=np.float32)
        )
        std = (
            self.features.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
            if pre_std is None else np.asarray(pre_std, dtype=np.float32)
        )
        self.std = np.where(std == 0, 1.0, std).astype(np.float32)
        if (
            self.mean.shape != (self.features.shape[1],)
            or self.std.shape != self.mean.shape
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid feature normalization statistics.")
        self.features = ((self.features - self.mean) / self.std).astype(np.float32)

    def normalize_y(self, offsets) -> None:
        offsets = np.asarray(offsets, dtype=np.float32)
        if offsets.shape != (3,) or not np.isfinite(offsets).all():
            raise ValueError("Expected three finite target offsets.")
        self.labels = (self.labels + offsets).astype(np.float32)
        self.y_offsets = offsets

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        start = index * self.n_ve
        stop = start + self.n_ve
        return self.features[start:stop], self.forces[start:stop], self.labels[index]


def read_split(dataset_path: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(dataset_path / f"{split}_ve.csv"),
        pd.read_csv(dataset_path / f"{split}_label.csv"),
    )


def make_split_loaders(
    dataset_path: Path,
    n_col: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    y_offsets: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ve, train_label = read_split(dataset_path, "train")
    train = VEFeaturesBadgapDataset(train_ve, train_label, n_col)
    train.normalize_X()
    if y_offsets is None:
        y_offsets = -train_label[LABEL_COLUMNS].mean().to_numpy(dtype=np.float32)
    train.normalize_y(y_offsets)
    datasets = [train]
    for split in ("val", "test"):
        ve, labels = read_split(dataset_path, split)
        dataset = VEFeaturesBadgapDataset(ve, labels, n_col, pre_mean=train.mean, pre_std=train.std)
        dataset.normalize_y(y_offsets)
        datasets.append(dataset)
    generator = torch.Generator().manual_seed(seed)
    loaders = [
        DataLoader(
            dataset, batch_size=batch_size, shuffle=index == 0,
            generator=generator if index == 0 else None,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        for index, dataset in enumerate(datasets)
    ]
    return loaders[0], loaders[1], loaders[2]


def convert_metrics_units(metrics: dict[str, float]) -> dict[str, float]:
    """Band edges are intensive (meV); total energy is reported in meV/atom."""
    return {
        key: float(value) * (ENERGY_EV_TO_MEV_PER_ATOM if key.startswith("energy_") else 1000.0)
        for key, value in metrics.items()
    }


def plot_test_prediction_grid(prediction: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, LABEL_COLUMNS):
        true = prediction[f"{name}_true"]
        predicted = prediction[f"{name}_pred"]
        lower, upper = float(min(true.min(), predicted.min())), float(max(true.max(), predicted.max()))
        padding = max((upper - lower) * 0.05, 0.01)
        limits = (lower - padding, upper + padding)
        ax.scatter(true, predicted, alpha=0.6, s=14)
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel="Reference (eV)", ylabel="Prediction (eV)", title=name.upper())
        scale = ENERGY_EV_TO_MEV_PER_ATOM if name == "energy" else 1000.0
        unit = "meV/atom" if name == "energy" else "meV"
        rmse = float(np.sqrt(np.mean((predicted - true) ** 2))) * scale
        ax.text(0.05, 0.95, f"RMSE = {rmse:.3f} {unit}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit("This module contains shared helpers. Train with run_bandgap_v1.py.")
