"""Evaluate a trained model on the test (or validation) split.

sl-evaluate runs/diffuse_learned_3                      # uses best.pth and config.toml
sl-evaluate --config configs/diffuse_learned_3.toml --checkpoint my.pth

Writes ``<split>_metrics.json`` next to the checkpoint and, with ``--results-csv``,
appends a row to a CSV that ``sl-plot-metrics`` reads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from learned_sl import distributed
from learned_sl.config import load_config, load_model
from learned_sl.data import DEFAULT_ROOT, LtmSplit, make_loader
from learned_sl.metrics import PAPER_THRESHOLDS_MM, DepthMetrics

RESULTS_COLUMNS = ["Dataset", "Pat", "Method", "RMSE_mm", "MAE_mm"] + [
    f"BAD-{t}_percent" for t in PAPER_THRESHOLDS_MM
]
SUBSET_NAMES = {"diffuse": "Diffuse", "metalic": "Metallic"}


def evaluate(
    model, loader, ctx, max_batches: int | None = None, half_precision_mm: bool = False
) -> dict[str, float]:
    """Depth metrics over the GT-lit pixels of every sample the loader yields."""
    model.eval()
    metrics = DepthMetrics(half_precision_mm=half_precision_mm)
    with torch.no_grad():
        for step, (ltms, depths, occlusions) in enumerate(
            tqdm(loader, desc="evaluate", disable=not ctx.is_main, leave=False)
        ):
            if step == max_batches:
                break
            pred, _ = model(ltms.to(ctx.device, non_blocking=True))
            metrics.update(pred, depths.to(ctx.device), occlusions.to(ctx.device))
    metrics.all_reduce(ctx.device)
    return metrics.compute()


def append_results_row(path: Path, subset: str, patterns: int, method: str, result: dict):
    row = {
        "Dataset": SUBSET_NAMES[subset],
        "Pat": patterns,
        "Method": method,
        "RMSE_mm": f"{result['rmse_mm']:.2f}",
        "MAE_mm": f"{result['mae_mm']:.2f}",
    }
    for t in PAPER_THRESHOLDS_MM:
        row[f"BAD-{t}_percent"] = f"{result[f'bad_{t}mm_pct']:.2f}"
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, nargs="?", help="directory written by sl-train")
    parser.add_argument("--config", type=Path, help="defaults to <run_dir>/config.toml")
    parser.add_argument("--checkpoint", type=Path, help="defaults to <run_dir>/best.pth")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32, help="per GPU")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--allow-partial-dataset", action="store_true")
    parser.add_argument(
        "--paper-rounding",
        action="store_true",
        help="convert float16 predictions to mm in float16, as for the paper's tables",
    )
    args = parser.parse_args()

    if args.run_dir is None and (args.config is None or args.checkpoint is None):
        parser.error("pass a run directory or both --config and --checkpoint")
    config = load_config(args.config or args.run_dir / "config.toml")
    checkpoint = args.checkpoint or args.run_dir / "best.pth"

    ctx = distributed.setup()
    split = LtmSplit(args.data_root, config.subset, args.split, args.allow_partial_dataset)
    loader = make_loader(
        split,
        args.batch_size,
        train=False,
        rank=ctx.rank,
        world_size=ctx.world_size,
        num_workers=args.num_workers,
    )
    model = load_model(config, checkpoint, ctx.device)
    result = evaluate(model, loader, ctx, half_precision_mm=args.paper_rounding)

    if ctx.is_main:
        result = {"name": config.name, "checkpoint": str(checkpoint), "split": args.split} | result
        print(json.dumps(result, indent=2))
        out = checkpoint.parent / f"{args.split}_metrics.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        if args.results_csv is not None:
            method = "Learned" if config.learn_patterns else "Fixed"
            append_results_row(args.results_csv, config.subset, config.num_patterns, method, result)
    distributed.teardown()


if __name__ == "__main__":
    main()
