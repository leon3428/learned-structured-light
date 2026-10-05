"""Plot MAE and RMSE against the number of patterns (paper Fig. 3).

sl-plot-metrics --input-csv results/my_results.csv --output-dir figures/metrics

The CSV is the one written by ``sl-evaluate`` and ``sl-baseline`` with ``--results-csv``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

FONT_SIZE = 24
METRICS = {"mae": ("MAE_mm", "MAE[mm]"), "rmse": ("RMSE_mm", "RMSE[mm]")}


def load_rows(path: Path) -> dict[str, dict[str, list[tuple[int, dict]]]]:
    """{dataset: {method: [(patterns, row), ...]}} for the Learned and Fixed rows."""
    data: dict = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            method = row["Method"].lower()
            if method in ("learned", "fixed"):
                data[row["Dataset"].lower()][method].append((int(row["Pat"]), row))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/metrics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, methods in load_rows(args.input_csv).items():
        for metric, (column, ylabel) in METRICS.items():
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
            for method, points in sorted(methods.items()):
                points.sort(key=lambda p: p[0])
                ax.plot(
                    [p for p, _ in points],
                    [float(row[column]) for _, row in points],
                    marker="o",
                    linewidth=5,
                    markersize=10,
                    label=method.capitalize(),
                )
            ax.set_xlabel("Number of patterns", fontsize=FONT_SIZE)
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
            ax.set_xticks([1, 3, 5])
            ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=FONT_SIZE - 1)
            fig.tight_layout()

            output_path = args.output_dir / f"{dataset}_{metric}.png"
            fig.savefig(output_path)
            plt.close(fig)
            print(output_path)


if __name__ == "__main__":
    main()
