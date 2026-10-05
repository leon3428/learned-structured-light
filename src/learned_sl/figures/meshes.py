"""Render depth maps as shaded off-axis meshes (paper Figs. 4 and 5).

sl-render-meshes --subset diffuse --samples 0 \\
    --runs runs/diffuse_fixed_1 runs/diffuse_learned_1 --baselines psp5

Writes ``<out-dir>/<subset>/<sample_id>/{gt,<run name>,<baseline>}.png``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from learned_sl import baselines
from learned_sl.config import load_config, load_model
from learned_sl.data import DEFAULT_ROOT, LtmSplit, collate
from learned_sl.geometry import FOV_DEG

HeightMode = Literal["relief", "depth", "metric"]


@dataclass(frozen=True)
class DepthRenderConfig:
    size: int = 800
    elev: float = 50.0
    azim: float = 0.0
    height_scale: float = 6.8
    max_mesh_size: int = 256
    stride: int = 1
    height_mode: HeightMode = "depth"
    fov_deg: float = FOV_DEG
    max_face_depth_delta: float | None = None
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    material_color: tuple[float, float, float] = (0.75, 0.76, 0.74)
    specular_strength: float = 0.1
    shininess: float = 8.0


DEFAULT_RENDER = DepthRenderConfig()


def _effective_stride(depth: np.ndarray, config: DepthRenderConfig) -> int:
    stride = max(1, int(config.stride))
    if config.max_mesh_size > 0:
        stride = max(stride, math.ceil(max(depth.shape) / config.max_mesh_size))
    return stride


def _depth_to_height(depth: np.ndarray, valid: np.ndarray, config: DepthRenderConfig):
    z = np.full(depth.shape, np.nan, dtype=np.float64)
    valid_depth = depth[valid]
    if valid_depth.size == 0:
        return z

    if config.height_mode in {"depth", "metric"}:
        z[valid] = -valid_depth * config.height_scale
        return z

    lo, hi = np.nanpercentile(valid_depth, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        z[valid] = 0.0
        return z
    z[valid] = np.clip((hi - valid_depth) / (hi - lo), 0.0, 1.0) * config.height_scale
    return z


def depth_to_mesh(
    depth: np.ndarray,
    valid: np.ndarray | None,
    config: DepthRenderConfig = DEFAULT_RENDER,
) -> trimesh.Trimesh:
    """Convert a depth map into a grid mesh over the valid pixels."""
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D after squeeze, got shape {depth.shape}")
    valid = np.isfinite(depth) if valid is None else np.asarray(valid, dtype=bool).squeeze()
    if valid.shape != depth.shape:
        raise ValueError(f"valid mask shape {valid.shape} does not match {depth.shape}")
    valid = valid & np.isfinite(depth)

    stride = _effective_stride(depth, config)
    original_height, original_width = depth.shape
    depth = depth[::stride, ::stride]
    valid = valid[::stride, ::stride]
    z = _depth_to_height(depth, valid, config)

    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    if config.height_mode == "metric":
        pixel_x = np.arange(0, original_width, stride, dtype=np.float64)[:width]
        pixel_y = np.arange(0, original_height, stride, dtype=np.float64)[:height]
        uu, vv = np.meshgrid(pixel_x, pixel_y)
        focal = original_width / (2.0 * np.tan(np.radians(config.fov_deg) / 2.0))
        metric_depth = depth * config.height_scale
        x = (uu - original_width / 2.0) * metric_depth / focal
        y = -(vv - original_height / 2.0) * metric_depth / focal
    else:
        aspect = width / height if height > 0 else 1.0
        x = (xx / max(width - 1, 1) - 0.5) * aspect
        y = 0.5 - yy / max(height - 1, 1)

    index = np.full((height, width), -1, dtype=np.int64)
    pixels = np.argwhere(valid)
    index[pixels[:, 0], pixels[:, 1]] = np.arange(len(pixels))
    vertices = np.column_stack([x[valid].ravel(), y[valid].ravel(), z[valid].ravel()]).reshape(
        -1, 3
    )

    tl, tr = index[:-1, :-1], index[:-1, 1:]
    bl, br = index[1:, :-1], index[1:, 1:]
    keep = (tl >= 0) & (tr >= 0) & (bl >= 0) & (br >= 0)
    if config.max_face_depth_delta is not None:
        corners = np.stack([depth[:-1, :-1], depth[:-1, 1:], depth[1:, :-1], depth[1:, 1:]])
        with np.errstate(invalid="ignore"):
            keep &= ~(corners.max(axis=0) - corners.min(axis=0) > config.max_face_depth_delta)
    tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
    faces = np.stack(
        [np.column_stack([tl, bl, tr]), np.column_stack([tr, bl, br])], axis=1
    ).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def face_colors(mesh: trimesh.Trimesh, config: DepthRenderConfig) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    light = np.array([0.45, 0.25, 0.85])
    light /= np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0])

    diffuse = 0.42 + 0.48 * np.clip(normals @ light, 0.0, 1.0)
    reflect = 2.0 * (normals @ light)[:, None] * normals - light
    reflect_lengths = np.linalg.norm(reflect, axis=1, keepdims=True)
    reflect = np.divide(
        reflect, reflect_lengths, out=np.zeros_like(reflect), where=reflect_lengths > 0
    )
    specular = np.clip(reflect @ view, 0.0, 1.0) ** config.shininess

    color = np.asarray(config.material_color)[None, :] * diffuse[:, None]
    color += config.specular_strength * specular[:, None]
    return np.clip(color, 0.0, 1.0)


def render_depth_mesh(
    depth: np.ndarray,
    valid: np.ndarray,
    output_path: str | Path,
    config: DepthRenderConfig = DEFAULT_RENDER,
) -> None:
    """Render a depth map as an off-axis shaded mesh PNG."""
    mesh = depth_to_mesh(depth, valid, config)
    if len(mesh.vertices) == 0:
        raise ValueError(f"No valid depth pixels to render for {output_path}")

    dpi = 100
    fig = plt.figure(
        figsize=(config.size / dpi, config.size / dpi),
        dpi=dpi,
        facecolor=config.background_color,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(config.background_color)
    ax.view_init(elev=config.elev, azim=config.azim)

    if len(mesh.faces) > 0:
        ax.add_collection3d(
            Poly3DCollection(
                mesh.vertices[mesh.faces],
                facecolors=face_colors(mesh, config),
                edgecolors="none",
                linewidths=0,
                antialiased=False,
            )
        )
    else:
        v = mesh.vertices
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c=[config.material_color])

    # Fixed framing shared by all renders: the table (z ~ -1 m * height_scale) and the
    # object fill the view, so meshes of the same sample line up across methods.
    center, radius = (0.0, 0.0, -6.3), 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _resolve_samples(split: LtmSplit, samples: str) -> list[str]:
    selected = []
    for token in (part.strip() for part in samples.split(",") if part.strip()):
        selected.append(split.sample_ids[int(token)] if token.isdigit() else token)
    if not selected:
        raise ValueError("At least one sample must be selected")
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subset", choices=("diffuse", "metalic"), required=True)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument(
        "--samples", default="0", help="comma-separated indices into the split, or sample ids"
    )
    parser.add_argument("--runs", nargs="*", type=Path, default=[], help="sl-train run directories")
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--baselines", nargs="*", choices=list(baselines.METHODS), default=[])
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/meshes"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial-dataset", action="store_true")
    parser.add_argument("--size", type=int, default=DepthRenderConfig.size)
    parser.add_argument("--elev", type=float, default=DepthRenderConfig.elev)
    parser.add_argument("--azim", type=float, default=DepthRenderConfig.azim)
    parser.add_argument("--height-scale", type=float, default=DepthRenderConfig.height_scale)
    parser.add_argument("--height-mode", choices=("relief", "depth", "metric"), default="depth")
    parser.add_argument("--max-mesh-size", type=int, default=DepthRenderConfig.max_mesh_size)
    parser.add_argument("--max-face-depth-delta", type=float, default=None)
    parser.add_argument(
        "--specular-strength", type=float, default=DepthRenderConfig.specular_strength
    )
    parser.add_argument("--shininess", type=float, default=DepthRenderConfig.shininess)
    args = parser.parse_args(argv)

    config = DepthRenderConfig(
        size=args.size,
        elev=args.elev,
        azim=args.azim,
        height_scale=args.height_scale,
        height_mode=args.height_mode,
        max_mesh_size=args.max_mesh_size,
        max_face_depth_delta=args.max_face_depth_delta,
        specular_strength=args.specular_strength,
        shininess=args.shininess,
    )
    split = LtmSplit(args.data_root, args.subset, args.split, args.allow_partial_dataset)
    samples = [split.sample(i) for i in _resolve_samples(split, args.samples)]
    device = torch.device(args.device)

    def out(sample, name):
        path = args.out_dir / args.subset / sample.sample_id / f"{name}.png"
        print(path)
        return path

    for sample in samples:
        render_depth_mesh(sample.depth, sample.occlusion, out(sample, "gt"), config)
        ltm = baselines.sample_ltm(sample)
        for method in args.baselines:
            rec = baselines.reconstruct(method, ltm)
            render_depth_mesh(rec.depth, rec.valid, out(sample, method), config)

    for run_dir in args.runs:
        run_config = load_config(run_dir / "config.toml")
        model = load_model(
            run_config, run_dir / args.checkpoint_name, device, use_mixed_precision=False
        )
        with torch.no_grad():
            for sample in samples:
                ltms, _, occlusions = collate([sample])
                depth = model(ltms.to(device))[0][0, 0].cpu().numpy()
                valid = occlusions[0, 0].numpy()
                render_depth_mesh(depth, valid, out(sample, run_config.name), config)


if __name__ == "__main__":
    main()
