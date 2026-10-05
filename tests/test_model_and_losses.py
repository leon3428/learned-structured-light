import pytest
import torch
from torch import nn
from torch.testing import assert_close

from learned_sl.losses import SSIM, masked_rmse, masked_ssim_loss, pattern_variance_loss
from learned_sl.model import DepthNet, UNet


def test_learned_patterns_start_uniform_gray():
    model = DepthNet((4, 5), (3, 2), nn.Identity(), n_channels=3, use_mixed_precision=False)

    assert model.projection_matrix.shape == (20, 3)
    assert_close(model.patterns(), torch.full((20, 3), 0.5))


def test_forward_simulates_one_image_per_pattern():
    model = DepthNet(
        (4, 5),
        (3, 2),
        nn.Identity(),
        projection_matrix=torch.ones(20, 2),
        n_channels=2,
        use_mixed_precision=False,
    )
    model.norm_layer = nn.Identity()
    indices = torch.tensor([[0, 5, 6, 11], [0, 19, 0, 19]])
    ltms = torch.sparse_coo_tensor(indices, torch.ones(4), (12, 20)).coalesce()

    output, patterns = model(ltms)

    assert patterns.shape == (20, 2)
    assert output.shape == (2, 2, 2, 3)
    assert_close(output[0, :, 0, 0], torch.ones(2))
    assert_close(output[1, :, 1, 2], torch.ones(2))
    assert output.sum() == 8


def test_fixed_projection_matrix_shape_is_checked():
    with pytest.raises(ValueError, match="projection_matrix must have shape"):
        DepthNet((4, 5), (3, 2), nn.Identity(), projection_matrix=torch.ones(20, 2), n_channels=3)


def test_unet_keeps_resolution_and_checkpoint_keys():
    unet = UNet(3, 1, base_channels=8)

    assert unet(torch.rand(2, 3, 64, 64)).shape == (2, 1, 64, 64)
    # Parameter names must stay compatible with checkpoints of the paper's runs.
    keys = set(unet.state_dict())
    assert "inc.double_conv.0.weight" in keys
    assert "up4.conv.double_conv.4.running_var" in keys
    assert "outc.weight" in keys and "outc.bias" not in keys


def test_masked_rmse():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gt = torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    assert_close(masked_rmse(pred, gt, torch.ones(2, 2, dtype=bool)), torch.tensor(1.5).sqrt())
    assert_close(
        masked_rmse(pred, gt, torch.tensor([[True, False], [True, False]])),
        torch.tensor(0.5).sqrt(),
    )
    assert_close(masked_rmse(pred, gt, torch.zeros(2, 2, dtype=bool)), torch.tensor(0.0))


def test_ssim_loss_ignores_masked_out_pixels():
    gt = torch.rand(1, 1, 16, 16)
    pred = gt.clone()
    mask = torch.ones_like(gt, dtype=bool)
    mask[..., :4, :] = False
    pred[~mask] = 0.0

    assert SSIM()(gt, gt).item() == pytest.approx(1.0)
    assert masked_ssim_loss(SSIM(), pred, gt, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_pattern_variance_loss_is_negative_variance():
    patterns = torch.tensor([[0.0], [1.0]])

    assert pattern_variance_loss(patterns).item() == pytest.approx(-0.25)
