"""Depth error metrics: MAE, RMSE and BAD-X over valid, non-occluded pixels."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist

PAPER_THRESHOLDS_MM = (2, 5, 10)


class DepthMetrics:
    """Accumulates per-pixel depth errors over many samples.

    Depths are in meters and errors are reported in millimeters. A pixel counts when
    it is lit by the projector in the ground truth (occlusion map) and, for classical
    baselines, when the method produced a depth for it.
    """

    def __init__(self, thresholds_mm=PAPER_THRESHOLDS_MM, half_precision_mm: bool = False):
        """``half_precision_mm`` reproduces the evaluation used for the paper's tables,
        which converted float16 network outputs to millimeters in float16. That rounds
        predictions to 0.5 mm steps near 1 m and mostly affects BAD-2.
        """
        self.thresholds_mm = tuple(thresholds_mm)
        self.half_precision_mm = half_precision_mm
        # abs-error sum, squared-error sum, valid pixels, total pixels, bad counts...
        self.sums = torch.zeros(4 + len(self.thresholds_mm), dtype=torch.float64)
        self.samples = 0

    def update(self, pred, gt, gt_valid, pred_valid=None) -> None:
        pred, gt, gt_valid = map(_as_tensor, (pred, gt, gt_valid))
        mask = gt_valid.bool()
        if pred_valid is not None:
            mask = mask & _as_tensor(pred_valid).bool()

        self.samples += pred.shape[0] if pred.ndim == 4 else 1
        pred, gt = pred[mask], gt[mask]
        if self.half_precision_mm:
            errors = torch.abs((pred.half() * 1000.0).double() - (gt.float() * 1000.0).double())
        else:
            errors = torch.abs(pred.double() - gt.double()) * 1000.0
        update = [
            errors.sum(),
            (errors**2).sum(),
            mask.sum(),
            torch.tensor(mask.numel(), device=mask.device),
        ]
        update += [(errors > t).sum() for t in self.thresholds_mm]
        self.sums += torch.stack([v.double() for v in update]).cpu()

    def all_reduce(self, device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        packed = torch.cat([self.sums, torch.tensor([self.samples], dtype=torch.float64)])
        packed = packed.to(device)
        dist.all_reduce(packed)
        self.sums = packed[:-1].cpu()
        self.samples = int(packed[-1].item())

    def compute(self) -> dict[str, float]:
        abs_sum, sq_sum, count, total, *bad = self.sums.tolist()
        if count == 0:
            raise ValueError("No valid pixels were accumulated")
        result = {
            "samples": self.samples,
            "mae_mm": abs_sum / count,
            "rmse_mm": math.sqrt(sq_sum / count),
        }
        for threshold, n in zip(self.thresholds_mm, bad):
            result[f"bad_{threshold}mm_pct"] = 100.0 * n / count
        result["counted_pixels_pct"] = 100.0 * count / total
        return result


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.array(x))
