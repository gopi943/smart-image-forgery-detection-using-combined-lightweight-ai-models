"""
CBAM — Convolutional Block Attention Module.

Two sequential sub-modules:
1. Channel Attention: learns WHICH feature channels are important.
2. Spatial Attention: learns WHERE to focus spatially.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention: GlobalAvgPool + GlobalMaxPool → SharedMLP → Sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Global average pooling
        avg_pool = x.mean(dim=[2, 3])  # (B, C)
        # Global max pooling
        max_pool = x.amax(dim=[2, 3])  # (B, C)
        # Shared MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        # Combine + sigmoid
        attn = torch.sigmoid(avg_out + max_out)  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial Attention: ChannelMax + ChannelAvg → 7×7 Conv → Sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(concat))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention then Spatial Attention.
    Input/Output: (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
