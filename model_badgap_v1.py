from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubNet(nn.Module):
    """Same hidden-size design as model_two_force/model_two SubNet, with 3 outputs."""

    def __init__(self, input_size: int, output_size: int = 3, c: int = 96, extra_hidden: bool = True):
        super().__init__()
        self.output_size = output_size
        self.extra_hidden = extra_hidden

        self.FC1 = nn.Linear(input_size, c)
        self.FC2 = nn.Linear(c, 4 * c)
        self.FC3 = nn.Linear(4 * c, 2 * c)
        self.FC4 = nn.Linear(2 * c, c)
        if self.extra_hidden:
            self.FC5 = nn.Linear(c, c)
            self.FC6 = nn.Linear(c, self.output_size)
        else:
            self.FC5 = nn.Linear(c, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.FC1(x))
        x = F.leaky_relu(self.FC2(x))
        x = F.leaky_relu(self.FC3(x))
        x = F.leaky_relu(self.FC4(x))
        x = self.FC5(x)
        if self.extra_hidden:
            x = F.leaky_relu(x)
            x = self.FC6(x)
        return x
    # def __init__(self, input_size: int, output_size: int = 3, c: int = 96, extra_hidden: bool = False):
    #     super().__init__()
    #     self.output_size = output_size
    #     self.extra_hidden = extra_hidden

    #     self.FC1 = nn.Linear(input_size, c)
    #     self.FC2 = nn.Linear(c, 2 * c)
    #     self.FC3 = nn.Linear(2 * c,  c)
    #     self.FC4 = nn.Linear(c, self.output_size)
    #     # if self.extra_hidden:
    #     #     self.FC5 = nn.Linear(c, c)
    #     #     self.FC6 = nn.Linear(c, self.output_size)
    #     # else:
    #     #     self.FC5 = nn.Linear(c, self.output_size)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = F.leaky_relu(self.FC1(x))
    #     x = F.leaky_relu(self.FC2(x))
    #     x = F.leaky_relu(self.FC3(x))
    #     x = F.leaky_relu(self.FC4(x))
    #     return x

@dataclass
class EvalSummary:
    vbm_rmse: float
    cbm_rmse: float
    energy_rmse: float


class DualBadGapModelContainer:
    """
    Joint optimization for two BOE modes.

    Each subnet outputs 3 values per VE:
    [vbm_part, cbm_part, energy_part]

    Structure-level scalar predictions are sums over VEs. The final scalar prediction
    used for loss is the average of the two subnets.
    """

    def __init__(self, model1: nn.Module, model2: nn.Module, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = model1.to(self.device)
        self.model2 = model2.to(self.device)
        self.mse = nn.MSELoss()

    def save(self, path1: str, path2: str) -> None:
        torch.save(self.model1.state_dict(), path1)
        torch.save(self.model2.state_dict(), path2)

    def load(self, path1: str, path2: str) -> None:
        self.model1.load_state_dict(torch.load(path1, map_location=self.device, weights_only=True))
        self.model2.load_state_dict(torch.load(path2, map_location=self.device, weights_only=True))
        self.model1.eval()
        self.model2.eval()

    def fit(
        self,
        loader1,
        loader2,
        epochs: int,
        lr: float = 1e-3,
        vbm_weight: float = 1.0,
        cbm_weight: float = 1.0,
        energy_weight: float = 1.0,
        val_loader1=None,
        val_loader2=None,
        checkpoint_paths: tuple[str | Path, str | Path] | None = None,
    ) -> tuple[list[dict[str, float]], dict[str, Any]]:
        optimizer = torch.optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()),
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

        history: list[dict[str, float]] = []
        self.model1.train()
        self.model2.train()

        best_states = {
            "vbm": {"rmse": float("inf"), "epoch": None, "state": None},
            "cbm": {"rmse": float("inf"), "epoch": None, "state": None},
            "energy": {"rmse": float("inf"), "epoch": None, "state": None},
            "composite": {"rmse": float("inf"), "epoch": None, "state": None},
        }

        for epoch in range(1, epochs + 1):
            loss_vbm_sum = 0.0
            loss_cbm_sum = 0.0
            loss_energy_sum = 0.0
            loss_total_sum = 0.0
            n_batches = 0

            for (x1, _f1, y1), (x2, _f2, y2) in zip(loader1, loader2):
                x1 = x1.to(self.device, dtype=torch.float32, non_blocking=True)
                y1 = y1.to(self.device, dtype=torch.float32, non_blocking=True)

                x2 = x2.to(self.device, dtype=torch.float32, non_blocking=True)
                y2 = y2.to(self.device, dtype=torch.float32, non_blocking=True)

                if y1.shape != y2.shape:
                    raise ValueError("Label shape mismatch between mode1 and mode2 loaders.")

                optimizer.zero_grad(set_to_none=True)

                b1, nve1, d1 = x1.shape
                b2, nve2, d2 = x2.shape
                if b1 != b2 or nve1 != nve2:
                    raise ValueError("Batch shape mismatch between mode1 and mode2 loaders.")

                out1 = self.model1(x1.reshape(b1 * nve1, d1)).reshape(b1, nve1, -1)
                out2 = self.model2(x2.reshape(b2 * nve2, d2)).reshape(b2, nve2, -1)

                pred_scalar1 = out1[:, :, 0:3].sum(dim=1)
                pred_scalar2 = out2[:, :, 0:3].sum(dim=1)
                pred_scalar = 0.5 * (pred_scalar1 + pred_scalar2)

                pred_vbm = pred_scalar[:, 0]
                pred_cbm = pred_scalar[:, 1]
                pred_energy = pred_scalar[:, 2]

                true_vbm = y1[:, 0]
                true_cbm = y1[:, 1]
                true_energy = y1[:, 3] if y1.shape[1] >= 4 else y1[:, 2]

                loss_vbm = self.mse(pred_vbm, true_vbm)
                loss_cbm = self.mse(pred_cbm, true_cbm)
                loss_energy = self.mse(pred_energy, true_energy)

                loss = (
                    vbm_weight * loss_vbm
                    + cbm_weight * loss_cbm
                    + energy_weight * loss_energy
                )

                loss.backward()
                optimizer.step()

                loss_vbm_sum += loss_vbm.item()
                loss_cbm_sum += loss_cbm.item()
                loss_energy_sum += loss_energy.item()
                loss_total_sum += loss.item()
                n_batches += 1

            scheduler.step()

            epoch_record = {
                "epoch": float(epoch),
                "vbm_mse": loss_vbm_sum / max(n_batches, 1),
                "cbm_mse": loss_cbm_sum / max(n_batches, 1),
                "energy_mse": loss_energy_sum / max(n_batches, 1),
                "total_loss": loss_total_sum / max(n_batches, 1),
                "lr": optimizer.param_groups[0]["lr"],
            }

            if val_loader1 is not None and val_loader2 is not None:
                self.model1.eval()
                self.model2.eval()
                val_metrics, _ = self.evaluate(val_loader1, val_loader2)
                self.model1.train()
                self.model2.train()
                epoch_record.update({
                    "val_vbm_rmse": val_metrics["vbm_rmse"],
                    "val_cbm_rmse": val_metrics["cbm_rmse"],
                    "val_energy_rmse": val_metrics["energy_rmse"],
                })

                for task in ("vbm", "cbm", "energy"):
                    key = f"{task}_rmse"
                    if val_metrics[key] < best_states[task]["rmse"]:
                        best_states[task]["rmse"] = val_metrics[key]
                        best_states[task]["epoch"] = epoch
                        best_states[task]["state"] = {
                            "model1": {k: v.cpu().clone() for k, v in self.model1.state_dict().items()},
                            "model2": {k: v.cpu().clone() for k, v in self.model2.state_dict().items()},
                        }
                composite = sum(val_metrics[f"{t}_rmse"] for t in ("vbm", "cbm", "energy"))
                if composite < best_states["composite"]["rmse"]:
                    best_states["composite"]["rmse"] = composite
                    best_states["composite"]["epoch"] = epoch
                    best_states["composite"]["state"] = {
                        "model1": {k: v.cpu().clone() for k, v in self.model1.state_dict().items()},
                        "model2": {k: v.cpu().clone() for k, v in self.model2.state_dict().items()},
                    }

            history.append(epoch_record)

            print(
                f"Epoch {epoch:03d} "
                f"vbm={epoch_record['vbm_mse']:.6f} "
                f"cbm={epoch_record['cbm_mse']:.6f} "
                f"energy={epoch_record['energy_mse']:.6f} "
                f"total={epoch_record['total_loss']:.6f}"
                + (
                    f" val_vbm={epoch_record['val_vbm_rmse']:.6f}"
                    f" val_cbm={epoch_record['val_cbm_rmse']:.6f}"
                    f" val_energy={epoch_record['val_energy_rmse']:.6f}"
                    if "val_vbm_rmse" in epoch_record
                    else ""
                )
            )

        if checkpoint_paths is not None:
            base1, base2 = Path(checkpoint_paths[0]), Path(checkpoint_paths[1])
            for task in ("vbm", "cbm", "energy", "composite"):
                info = best_states[task]
                if info["state"] is not None:
                    path1 = base1.with_stem(f"{base1.stem}_{task}_best")
                    path2 = base2.with_stem(f"{base2.stem}_{task}_best")
                    torch.save(info["state"]["model1"], path1)
                    torch.save(info["state"]["model2"], path2)
                    print(
                        f"Saved {task} best checkpoint (epoch={info['epoch']}, "
                        f"val_rmse={info['rmse']:.6f}) to {path1}, {path2}"
                    )

        return history, best_states

    def predict(self, loader1, loader2) -> dict[str, Any]:
        self.model1.eval()
        self.model2.eval()

        true_vbm = []
        true_cbm = []
        true_energy = []

        pred_vbm = []
        pred_cbm = []
        pred_energy = []

        with torch.inference_mode():
            for (x1, _f1, y1), (x2, _f2, y2) in zip(loader1, loader2):
                x1 = x1.to(self.device, dtype=torch.float32, non_blocking=True)
                x2 = x2.to(self.device, dtype=torch.float32, non_blocking=True)

                b1, nve1, d1 = x1.shape
                b2, nve2, d2 = x2.shape
                if b1 != b2 or nve1 != nve2:
                    raise ValueError("Batch shape mismatch between mode1 and mode2 loaders.")

                out1 = self.model1(x1.reshape(b1 * nve1, d1)).reshape(b1, nve1, -1)
                out2 = self.model2(x2.reshape(b2 * nve2, d2)).reshape(b2, nve2, -1)

                scalar1 = out1[:, :, 0:3].sum(dim=1)
                scalar2 = out2[:, :, 0:3].sum(dim=1)
                scalar = 0.5 * (scalar1 + scalar2)

                batch_pred_vbm = scalar[:, 0]
                batch_pred_cbm = scalar[:, 1]
                batch_pred_energy = scalar[:, 2]

                pred_vbm.append(batch_pred_vbm.cpu().numpy())
                pred_cbm.append(batch_pred_cbm.cpu().numpy())
                pred_energy.append(batch_pred_energy.cpu().numpy())

                true_vbm.append(y1[:, 0].cpu().numpy())
                true_cbm.append(y1[:, 1].cpu().numpy())
                true_energy.append((y1[:, 3] if y1.shape[1] >= 4 else y1[:, 2]).cpu().numpy())

        return {
            "vbm_true": np.concatenate(true_vbm, axis=0),
            "vbm_pred": np.concatenate(pred_vbm, axis=0),
            "cbm_true": np.concatenate(true_cbm, axis=0),
            "cbm_pred": np.concatenate(pred_cbm, axis=0),
            "energy_true": np.concatenate(true_energy, axis=0),
            "energy_pred": np.concatenate(pred_energy, axis=0),
        }

    @staticmethod
    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def evaluate(self, loader1, loader2) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
        pred = self.predict(loader1, loader2)

        metrics = {
            "vbm_rmse": self._rmse(pred["vbm_true"], pred["vbm_pred"]),
            "cbm_rmse": self._rmse(pred["cbm_true"], pred["cbm_pred"]),
            "energy_rmse": self._rmse(pred["energy_true"], pred["energy_pred"]),
            "vbm_bias": float(np.mean(pred["vbm_pred"] - pred["vbm_true"])),
            "cbm_bias": float(np.mean(pred["cbm_pred"] - pred["cbm_true"])),
            "energy_bias": float(np.mean(pred["energy_pred"] - pred["energy_true"])),
        }

        frames = {
            "vbm": pd.DataFrame({"true": pred["vbm_true"], "pred": pred["vbm_pred"]}),
            "cbm": pd.DataFrame({"true": pred["cbm_true"], "pred": pred["cbm_pred"]}),
            "energy": pd.DataFrame({"true": pred["energy_true"], "pred": pred["energy_pred"]}),
        }
        return metrics, frames
