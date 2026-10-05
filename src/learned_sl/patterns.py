"""Fixed sinusoidal patterns used by the 'Fixed' neural baseline, and pattern export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learned_sl.data import RESOLUTION

SINGLE_SHOT_PERIODS = 6


def fixed_patterns(count: int, resolution: tuple[int, int] = RESOLUTION) -> np.ndarray:
    """Fixed patterns as uint8 images, shape (count, height, width).

    One pattern: a vertical sine fringe with 6 periods across the projector.
    Three or more: ``count``-step phase shifting with one period across the projector,
    ``0.5 * (cos(2 pi x / width + 2 pi n / count) + 1)``.
    """
    width, height = resolution
    if count == 1:
        x = np.linspace(0, 2 * np.pi * SINGLE_SHOT_PERIODS, width, endpoint=False)
        rows = [127.5 * (np.sin(x) + 1)]
    elif count >= 3:
        x = np.arange(width)
        rows = [
            255 * 0.5 * (np.cos(2 * np.pi / width * x + 2 * np.pi * n / count) + 1)
            for n in range(count)
        ]
    else:
        raise ValueError(f"Fixed patterns exist for 1 or >= 3 patterns, got {count}")
    return np.stack([np.tile(row, (height, 1)) for row in rows]).astype(np.uint8)


def fixed_projection_matrix(count: int) -> torch.Tensor:
    """Fixed patterns as a (projector_pixels, count) matrix with values in [0, 1]."""
    patterns = fixed_patterns(count).astype(np.float32) / 255.0
    return torch.from_numpy(patterns.reshape(count, -1).T.copy())


def learned_patterns(state_dict: dict, resolution: tuple[int, int] = RESOLUTION):
    """Learned patterns sigmoid(P) from a checkpoint as uint8 images.

    Images are flipped vertically, as in the paper's pattern figures.
    """
    width, height = resolution
    patterns = torch.sigmoid(state_dict["projection_matrix"].float()).cpu().numpy()
    images = [np.flip(patterns[:, i].reshape(height, width), 0) for i in range(patterns.shape[1])]
    return (np.stack(images) * 255).astype(np.uint8)


def _save(images: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = out_dir / f"{prefix}_{i}.png"
        Image.fromarray(image).save(path)
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export projection patterns as PNGs.")
    sub = parser.add_subparsers(dest="command", required=True)
    fixed = sub.add_parser("fixed", help="Write the fixed sinusoidal patterns")
    fixed.add_argument("--count", type=int, choices=(1, 3, 5), required=True)
    fixed.add_argument("--out-dir", type=Path, required=True)
    learned = sub.add_parser("learned", help="Write the patterns of a learned model")
    learned.add_argument("checkpoint", type=Path)
    learned.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "fixed":
        _save(fixed_patterns(args.count), args.out_dir, "fixed")
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        _save(learned_patterns(state_dict), args.out_dir, "learned")
