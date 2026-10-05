"""Differentiable structured-light model: projected patterns -> LTM -> U-Net decoder."""

from __future__ import annotations

import torch
from torch import nn


class DepthNet(nn.Module):
    """Simulates the camera images for the current patterns and decodes depth.

    The input is a block-row sparse LTM of shape (batch * camera_pixels,
    projector_pixels) (see ``learned_sl.data.collate``). With ``projection_matrix=None``
    the patterns are learned: ``projection_matrix`` holds the unconstrained logits P,
    initialized to zero (uniform gray after the sigmoid). Otherwise it holds fixed
    patterns with intensities in [0, 1].
    """

    def __init__(
        self,
        projector_resolution: tuple[int, int],
        camera_resolution: tuple[int, int],
        decoder: nn.Module,
        n_channels: int = 1,
        projection_matrix: torch.Tensor | None = None,
        use_mixed_precision: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.use_mixed_precision = use_mixed_precision
        self.camera_resolution = camera_resolution
        self.projector_resolution = projector_resolution
        self.camera_pixel_count = camera_resolution[0] * camera_resolution[1]
        self.projector_pixel_count = projector_resolution[0] * projector_resolution[1]
        self.learn_patterns = projection_matrix is None
        self.n_channels = n_channels

        if self.learn_patterns:
            self.projection_matrix = nn.Parameter(
                torch.zeros(self.projector_pixel_count, n_channels)
            )
        else:
            if projection_matrix.shape != (self.projector_pixel_count, n_channels):
                raise ValueError(
                    "projection_matrix must have shape "
                    f"({self.projector_pixel_count}, {n_channels}), got "
                    f"{tuple(projection_matrix.shape)}"
                )
            self.register_buffer("projection_matrix", projection_matrix.float())
        self.norm_layer = nn.InstanceNorm2d(n_channels)

    def patterns(self) -> torch.Tensor:
        """Projected intensities in [0, 1], shape (projector_pixels, n_channels)."""
        if self.learn_patterns:
            return torch.sigmoid(self.projection_matrix)
        return self.projection_matrix

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.projector_pixel_count:
            raise ValueError(
                "LTM projector dimension does not match projector_resolution: "
                f"expected {self.projector_pixel_count}, got {x.shape[1]}"
            )
        if x.shape[0] % self.camera_pixel_count != 0:
            raise ValueError(
                "LTM camera dimension is not divisible by camera_resolution: "
                f"camera pixels={self.camera_pixel_count}, rows={x.shape[0]}"
            )

        batch_size = x.shape[0] // self.camera_pixel_count
        patterns = self.patterns()
        imgs = torch.sparse.mm(x, patterns)

        camera_width, camera_height = self.camera_resolution
        imgs = torch.reshape(imgs, (batch_size, camera_height, camera_width, self.n_channels))
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        imgs = self.norm_layer(imgs)

        if self.use_mixed_precision and imgs.is_cuda:
            with torch.autocast(device_type="cuda"):
                return self.decoder(imgs), patterns

        return self.decoder(imgs), patterns


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bilinear upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """Five-level U-Net with bilinear upsampling."""

    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 8)
        self.up1 = Up(c * 16, c * 4)
        self.up2 = Up(c * 8, c * 2)
        self.up3 = Up(c * 4, c)
        self.up4 = Up(c * 2, c)
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
