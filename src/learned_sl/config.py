"""Experiment configuration, loaded from the TOML files in ``configs/``."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from learned_sl.data import RESOLUTION
from learned_sl.model import DepthNet, UNet
from learned_sl.patterns import fixed_projection_matrix


@dataclass(frozen=True)
class Config:
    name: str
    subset: Literal["diffuse", "metalic"]
    num_patterns: int
    patterns: Literal["learned", "fixed"]
    epochs: int = 80
    batch_size: int = 16  # total over all GPUs
    decoder_lr: float = 1e-5
    pattern_lr: float = 2e-4  # only used with learned patterns
    weight_decay: float = 1e-6  # decoder only
    ssim_weight: float = 0.74
    variance_weight: float = 1e-3  # only used with learned patterns
    grad_clip_norm: float | None = None
    base_channels: int = 32
    seed: int = 0

    @property
    def learn_patterns(self) -> bool:
        return self.patterns == "learned"

    def to_toml(self) -> str:
        lines = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            lines.append(f"{field.name} = {value!r}".replace("'", '"'))
        return "\n".join(lines) + "\n"


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        values = tomllib.load(f)
    names = {field.name for field in dataclasses.fields(Config)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
    config = Config(**values)
    if config.patterns not in ("learned", "fixed"):
        raise ValueError(f"patterns must be 'learned' or 'fixed', got {config.patterns!r}")
    return config


def build_model(config: Config, use_mixed_precision: bool = True) -> DepthNet:
    projection_matrix = (
        None if config.learn_patterns else fixed_projection_matrix(config.num_patterns)
    )
    decoder = UNet(config.num_patterns, 1, base_channels=config.base_channels)
    return DepthNet(
        RESOLUTION,
        RESOLUTION,
        decoder,
        n_channels=config.num_patterns,
        projection_matrix=projection_matrix,
        use_mixed_precision=use_mixed_precision,
    )


def load_model(
    config: Config, checkpoint: str | Path, device, use_mixed_precision: bool = True
) -> DepthNet:
    model = build_model(config, use_mixed_precision).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()
