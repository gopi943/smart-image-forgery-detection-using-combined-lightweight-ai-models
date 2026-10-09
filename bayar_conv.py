"""
BayarConv — Learnable Constrained Convolution for Manipulation Detection.

Replaces fixed SRM filters with learned noise extraction filters.
Constraint: center weight = -1, all other weights sum to +1.
This forces every filter to act as a prediction residual filter,
suppressing image content and exposing manipulation artifacts.

Reference:
    Bayar & Stamm, "A Deep Learning Approach to Universal Image
    Manipulation Detection", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """
    Constrained convolutional layer for forensic noise extraction.

    Every filter computes: output = weighted_avg(neighbors) - center_pixel
    This is enforced by:
        - Fixing center weight to -1
        - Constraining all other weights to sum to +1

    Unlike SRM (3 fixed filters), BayarConv learns optimal filters for
    the specific manipulation types in the training data.

    Args:
        in_channels: Input channels (3 for RGB).
        out_channels: Number of learned noise filters (default: 3).
        kernel_size: Filter size (default: 5, same as SRM).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2

        # Learnable filter weights (initialized with Kaiming)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def _constrain_weights(self):
        """
        Build Bayar-constrained weights without mutating the learnable
        parameter tensor in-place:
            1. Set center weight to 0 temporarily
            2. Normalize non-center weights to sum to +1
            3. Set center weight to -1
        """
        weight = self.weight
        non_center = weight.clone()
        non_center[:, :, self.center, self.center] = 0

        # Normalize remaining weights to sum to 1 (per filter per channel).
        w_sum = non_center.sum(dim=(2, 3), keepdim=True)
        w_sum = torch.where(
            w_sum.abs() < 1e-8,
            torch.ones_like(w_sum),
            w_sum,
        )
        constrained = non_center / w_sum
        constrained[:, :, self.center, self.center] = -1
        return constrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Noise residual map (B, out_channels, H, W)
        """
        constrained_weight = self._constrain_weights()
        # Force FP32 to prevent AMP FP16 instability from -1 center weight
        return F.conv2d(
            x.float(),
            constrained_weight.float(),
            padding=self.center,
        )


class BayarFilter(nn.Module):
    """
    Drop-in replacement for SRMFilter.

    Uses BayarConv for learnable noise extraction, followed by
    batch normalization for stable training. Output has the same
    shape as input (B, 3, H, W) for compatibility with TextureEncoder.

    Args:
        num_filters: Number of BayarConv filter banks (default: 3).
        kernel_size: Spatial size of each filter (default: 5).
    """

    def __init__(self, num_filters: int = 3, kernel_size: int = 5):
        super().__init__()
        self.bayar = BayarConv2d(
            in_channels=3, out_channels=num_filters, kernel_size=kernel_size
        )
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image (B, 3, H, W)
        Returns:
            Noise residual (B, 3, H, W) — same shape for TextureEncoder compat
        """
        residual = self.bayar(x)
        return self.bn(residual)
