import json
import sys

import pytest
import torch

from learned_sl import evaluate
from learned_sl.config import Config, load_config
from learned_sl.train import train


@pytest.mark.parametrize("patterns", ["learned", "fixed"])
def test_train_then_evaluate(tiny_dataset, tmp_path, monkeypatch, patterns):
    root, *_ = tiny_dataset
    config = Config(
        name=f"tiny_{patterns}",
        subset="diffuse",
        num_patterns=3,
        patterns=patterns,
        epochs=2,
        batch_size=4,
        base_channels=4,
        grad_clip_norm=None if patterns == "learned" else 1.0,
    )
    out = tmp_path / "runs"

    train(config, str(root), out, num_workers=0)

    run_dir = out / config.name
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]
    assert json.loads(lines[0])["val_samples"] == 6
    assert load_config(run_dir / "config.toml") == config
    state = torch.load(run_dir / "best.pth")
    assert ("projection_matrix" in state) and state["projection_matrix"].shape == (65536, 3)

    results_csv = tmp_path / "results.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sl-evaluate",
            str(run_dir),
            "--data-root",
            str(root),
            "--num-workers",
            "0",
            "--results-csv",
            str(results_csv),
        ],
    )
    evaluate.main()

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert metrics["samples"] == 10
    assert results_csv.read_text().splitlines()[1].startswith(f"Diffuse,3,{patterns.capitalize()},")
