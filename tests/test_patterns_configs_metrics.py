from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from learned_sl.config import Config, build_model, load_config
from learned_sl.metrics import DepthMetrics
from learned_sl.patterns import fixed_patterns, fixed_projection_matrix, learned_patterns

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "data" / "fixed_patterns"


@pytest.mark.parametrize("count", [1, 3, 5])
def test_fixed_patterns_match_the_pngs_used_for_the_paper(count):
    patterns = fixed_patterns(count)

    for i, pattern in enumerate(patterns):
        expected = np.asarray(Image.open(FIXTURES / f"{count}_fringe_{i}.png"))
        assert np.array_equal(pattern, expected)
    assert fixed_projection_matrix(count).shape == (256 * 256, count)


def test_learned_pattern_export():
    state = {"projection_matrix": torch.zeros(256 * 256, 2)}

    images = learned_patterns(state)

    assert images.shape == (2, 256, 256) and images.dtype == np.uint8
    assert np.all(images == 127)


def test_paper_configs():
    configs = {p.stem: load_config(p) for p in sorted((ROOT / "configs").glob("*.toml"))}

    assert len(configs) == 15
    for name, config in configs.items():
        assert config.name == name
        assert (config.epochs, config.batch_size, config.weight_decay) == (80, 16, 1e-6)
        if config.learn_patterns:
            assert (config.decoder_lr, config.pattern_lr, config.grad_clip_norm) == (
                1e-5,
                2e-4,
                None,
            )
        else:
            assert (config.decoder_lr, config.grad_clip_norm, config.ssim_weight) == (
                5e-6,
                1.0,
                0.74,
            )
    assert configs["diffuse_learned_3"].ssim_weight == 0.74
    assert configs["diffuse_learned_3"].variance_weight == 1e-3
    assert configs["ablation_no_ssim"].ssim_weight == 0.0
    assert configs["ablation_no_variance"].variance_weight == 0.0


def test_config_round_trips_through_toml(tmp_path):
    config = Config(
        name="x", subset="metalic", num_patterns=3, patterns="fixed", grad_clip_norm=1.0
    )
    path = tmp_path / "config.toml"
    path.write_text(config.to_toml())

    assert load_config(path) == config
    assert build_model(config).learn_patterns is False


def test_depth_metrics():
    metrics = DepthMetrics(thresholds_mm=(2, 5))
    gt = np.full((2, 2), 1.0, dtype=np.float32)
    pred = gt + np.array([[0.001, 0.003], [0.006, 1.0]], dtype=np.float32)
    gt_valid = np.array([[True, True], [True, False]])

    metrics.update(pred, gt, gt_valid)
    result = metrics.compute()

    assert result["mae_mm"] == pytest.approx(10 / 3, rel=1e-4)
    assert result["rmse_mm"] == pytest.approx(np.sqrt(46 / 3), rel=1e-4)
    assert result["bad_2mm_pct"] == pytest.approx(200 / 3)
    assert result["bad_5mm_pct"] == pytest.approx(100 / 3)

    with_pred_mask = DepthMetrics(thresholds_mm=(2,))
    with_pred_mask.update(pred, gt, gt_valid, pred_valid=np.array([[True, False], [True, True]]))
    assert with_pred_mask.compute()["mae_mm"] == pytest.approx(3.5, rel=1e-4)


def test_half_precision_metrics_round_to_half_millimeters():
    gt = torch.full((1, 1, 1, 1), 0.9)
    pred = gt + 0.0002  # 0.2 mm error
    exact = DepthMetrics()
    half = DepthMetrics(half_precision_mm=True)

    exact.update(pred, gt, torch.ones_like(gt, dtype=bool))
    half.update(pred, gt, torch.ones_like(gt, dtype=bool))

    assert exact.compute()["mae_mm"] == pytest.approx(0.2, abs=1e-3)
    assert half.compute()["mae_mm"] == pytest.approx(0.5, abs=1e-3)  # 900.5 mm in float16
