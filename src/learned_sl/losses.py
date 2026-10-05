"""Training losses: masked RMSE, masked SSIM and pattern variance (paper Eqs. 5-9)."""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RMSE over the pixels where ``mask`` is True (Eq. 5)."""
    num_valid = mask.sum()
    if num_valid == 0:
        return pred.sum() * 0.0
    mse = ((pred - gt) ** 2)[mask].sum() / (num_valid + 1e-8)
    return torch.sqrt(mse)


def pattern_variance_loss(patterns: torch.Tensor) -> torch.Tensor:
    """Negative variance of the projected intensities sigmoid(P) (Eq. 9)."""
    return -torch.var(patterns, unbiased=False)


# SSIM below is adapted from https://github.com/jorge-pessoa/pytorch-msssim (MIT).


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window, val_range: float) -> torch.Tensor:
    channel = img1.size(1)
    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim_map.mean()


class SSIM(torch.nn.Module):
    """Mean SSIM with an 11x11 Gaussian window and no padding."""

    def __init__(self, window_size: int = 11, val_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.val_range = val_range
        self.channel = 1
        self.register_buffer("window", _create_window(window_size))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, channel, height, width = img1.size()
        size = min(self.window_size, height, width)
        if not (
            channel == self.channel
            and self.window.dtype == img1.dtype
            and self.window.device == img1.device
            and self.window.size(-1) == size
        ):
            self.window = _create_window(size, channel).to(img1.device, img1.dtype)
            self.channel = channel
        return ssim(img1, img2, self.window, self.val_range)


def masked_ssim_loss(
    ssim_fn: SSIM, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """1 - SSIM after replacing predictions at masked-out pixels with GT (Eqs. 6-7)."""
    return 1 - ssim_fn(torch.where(mask, pred, gt), gt)
