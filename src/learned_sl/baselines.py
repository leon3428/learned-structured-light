"""Classical structured-light baselines: FTP (1 pattern) and PSP (3 and 5 patterns).

The camera images for the baseline patterns are simulated with the sample's LTM, decoded
to projector columns, and triangulated with the known synthetic geometry.

- ``ftp``:  one OpenCV FTP fringe, decoded by row-wise Fourier demodulation.
- ``psp3``: OpenCV's three-step PSP patterns, decoded by least-squares phase shifting.
- ``psp5``: five-step phase shifting with the same decoder.

sl-baseline --subset diffuse --methods ftp psp3 psp5 --results-csv results/my_results.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from learned_sl.data import DEFAULT_ROOT, LTM_SCALE, PIXELS, LtmSplit, Sample, read_row_group
from learned_sl.evaluate import append_results_row
from learned_sl.geometry import (
    intrinsic_matrix,
    point_cloud_to_depth_map,
    projector_pose,
    triangulate,
)
from learned_sl.metrics import DepthMetrics

METHODS = {"ftp": 1, "psp3": 3, "psp5": 5}  # method -> number of patterns
RESULT_NAMES = {"ftp": "FTP", "psp3": "PSP", "psp5": "PSP"}
MAX_DEPTH_M = 2.5  # camera far clip


@dataclass(frozen=True)
class Reconstruction:
    depth: np.ndarray
    valid: np.ndarray
    correspondence: np.ndarray


def sample_ltm(sample: Sample) -> coo_matrix:
    values = sample.values.astype(np.float32) / LTM_SCALE
    return coo_matrix(
        (values, (sample.rows.astype(np.int64), sample.cols.astype(np.int64))),
        shape=(PIXELS, PIXELS),
    )


def capture(ltm: coo_matrix, patterns: np.ndarray, camera_resolution) -> np.ndarray:
    """Simulated 8-bit camera images for uint8 patterns (count, height, width)."""
    width, height = camera_resolution
    vectors = patterns.reshape(len(patterns), -1).astype(np.float64) / 255.0
    images = (ltm @ vectors.T).T.reshape(len(patterns), height, width)
    return (np.minimum(images, 1.0) * 255).astype(np.uint8)


def opencv_sinusoidal_patterns(method_id: int, resolution) -> list[np.ndarray]:
    """One-period vertical fringes from OpenCV's structured_light module."""
    width, height = resolution
    params = cv2.structured_light.SinusoidalPattern.Params()
    params.width = width
    params.height = height
    params.nbrOfPeriods = 1
    params.setMarkers = False
    params.horizontal = False
    params.methodId = method_id
    params.shiftValue = 2.0 * np.pi / 3.0
    params.nbrOfPixelsBetweenMarkers = max(1, min(width, height) // 4)
    generated = cv2.structured_light.SinusoidalPattern.create(params).generate()
    if isinstance(generated, tuple):
        ok, generated = generated
        if not ok:
            raise RuntimeError("OpenCV failed to generate structured-light patterns")
    return [np.asarray(image, dtype=np.uint8) for image in generated]


def phase_shift_patterns(resolution, num_steps: int) -> np.ndarray:
    """N-step phase-shifting patterns with one period across the projector."""
    width, height = resolution
    shifts = 2.0 * np.pi * np.arange(num_steps) / num_steps
    phase = (2.0 * np.pi / width) * np.arange(width)
    rows = 0.5 * (np.cos(phase[None, :] + shifts[:, None]) + 1.0)
    patterns = np.repeat(rows[:, None, :], height, axis=1)
    return np.clip(np.rint(255.0 * patterns), 0, 255).astype(np.uint8)


def decode_phase_shift(captures: np.ndarray, projector_width: int) -> np.ndarray:
    """Least-squares N-step phase decoding to (wrapped) projector columns."""
    samples = np.asarray(captures, dtype=np.float64)
    num_steps = samples.shape[0]
    n = np.arange(num_steps, dtype=np.float64)
    basis = np.column_stack(
        [np.ones_like(n), np.cos(2.0 * np.pi * n / num_steps), -np.sin(2.0 * np.pi * n / num_steps)]
    )
    coeffs = np.linalg.pinv(basis) @ samples.reshape(num_steps, -1)
    _, cos_term, sin_term = coeffs.reshape(3, samples.shape[1], samples.shape[2])
    phase = np.mod(np.arctan2(sin_term, cos_term), 2.0 * np.pi)
    return phase * projector_width / (2.0 * np.pi)


def decode_ftp(capture: np.ndarray, projector_width: int, offset_px: float = 0.0):
    """Row-wise Fourier (Hilbert) demodulation of a single fringe image."""
    samples = np.asarray(capture, dtype=np.float64)
    centered = samples - np.mean(samples, axis=1, keepdims=True)
    spectrum = np.fft.fft(centered, axis=1)

    width = samples.shape[1]
    hilbert = np.zeros(width, dtype=np.float64)
    hilbert[0] = 1.0
    if width % 2 == 0:
        hilbert[1 : width // 2] = 2.0
        hilbert[width // 2] = 1.0
    else:
        hilbert[1 : (width + 1) // 2] = 2.0

    analytic = np.fft.ifft(spectrum * hilbert[None, :], axis=1)
    phase = np.mod(np.angle(analytic), 2.0 * np.pi)
    correspondence = np.mod(phase * projector_width / (2.0 * np.pi) - offset_px, projector_width)
    correspondence[np.abs(analytic) <= 1e-6] = np.nan
    return correspondence


def _wrapped_offset(decoded_row: np.ndarray, period_px: int) -> float:
    """Median offset between decoded and true columns, used to calibrate the phase origin."""
    expected = np.arange(period_px, dtype=np.float64)
    valid = np.isfinite(decoded_row)
    delta = np.mod(decoded_row[valid] - expected[valid] + period_px / 2, period_px)
    return float(np.median(delta - period_px / 2))


def _phase_shift_correspondence(ltm, patterns, resolution) -> np.ndarray:
    width = resolution[0]
    decoded = decode_phase_shift(capture(ltm, np.asarray(patterns), resolution), width)
    reference = decode_phase_shift(np.asarray(patterns), width)
    offset = _wrapped_offset(reference[reference.shape[0] // 2], width)
    return np.mod(decoded - offset, width)


def correspondence(method: str, ltm: coo_matrix, resolution) -> np.ndarray:
    """Projector column seen by each camera pixel (NaN where undecodable)."""
    width = resolution[0]
    if method == "ftp":
        pattern = opencv_sinusoidal_patterns(cv2.structured_light.FTP, resolution)[0]
        reference = decode_ftp(pattern, width)
        offset = _wrapped_offset(reference[pattern.shape[0] // 2], width)
        return decode_ftp(capture(ltm, pattern[None], resolution)[0], width, offset)
    if method == "psp3":
        patterns = opencv_sinusoidal_patterns(cv2.structured_light.PSP, resolution)
        return _phase_shift_correspondence(ltm, patterns, resolution)
    if method == "psp5":
        return _phase_shift_correspondence(ltm, phase_shift_patterns(resolution, 5), resolution)
    raise ValueError(f"Unknown method {method!r}; expected one of {list(METHODS)}")


def reconstruct(method: str, ltm: coo_matrix, resolution=(256, 256)) -> Reconstruction:
    columns = correspondence(method, ltm, resolution)
    y, x = np.where(np.isfinite(columns))
    K = intrinsic_matrix(resolution)
    R, t = projector_pose()
    points, valid = triangulate(
        K, K, R, t, x.astype(np.float64), y.astype(np.float64), columns[y, x]
    )

    width, height = resolution
    depth, hits = point_cloud_to_depth_map(points, np.column_stack([x, y])[valid], (height, width))
    is_valid = (hits > 0) & np.isfinite(depth) & (depth > 0.001) & (depth <= MAX_DEPTH_M)
    return Reconstruction(depth, is_valid, columns)


def _init_worker() -> None:
    # One process per CPU core; nested thread pools would oversubscribe the machine.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _evaluate_group(job) -> list[DepthMetrics]:
    group, methods, limit_ids = job
    results = [DepthMetrics() for _ in methods]
    for sample in read_row_group(group):
        if limit_ids is not None and sample.sample_id not in limit_ids:
            continue
        ltm = sample_ltm(sample)
        for metrics, method in zip(results, methods):
            rec = reconstruct(method, ltm)
            metrics.update(rec.depth, sample.depth, sample.occlusion, rec.valid)
    return results


def evaluate_baselines(split: LtmSplit, methods, workers: int = 8, limit: int | None = None):
    limit_ids = set(split.sample_ids[:limit]) if limit is not None else None
    groups = split.groups
    if limit_ids is not None:
        groups = [g for g in groups if any(i in limit_ids for i in _group_ids(g))]
    totals = [DepthMetrics() for _ in methods]
    jobs = [(group, tuple(methods), limit_ids) for group in groups]
    with Pool(workers, initializer=_init_worker) as pool:
        for partial in tqdm(pool.imap_unordered(_evaluate_group, jobs), total=len(jobs)):
            for total, part in zip(totals, partial):
                total.sums += part.sums
                total.samples += part.samples
    return {method: total.compute() for method, total in zip(methods, totals)}


def _group_ids(group) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(group.path).read_row_group(group.index, columns=["sample_id"])
    return [i for i, keep in zip(table["sample_id"].to_pylist(), group.keep) if keep]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subset", choices=("diffuse", "metalic"), required=True)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    parser.add_argument("--out", type=Path, default=None, help="write metrics as JSON")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--allow-partial-dataset", action="store_true")
    args = parser.parse_args()

    split = LtmSplit(args.data_root, args.subset, args.split, args.allow_partial_dataset)
    results = evaluate_baselines(split, args.methods, args.workers, args.limit)
    print(json.dumps(results, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if args.results_csv is not None:
        for method, result in results.items():
            append_results_row(
                args.results_csv, args.subset, METHODS[method], RESULT_NAMES[method], result
            )


if __name__ == "__main__":
    main()
