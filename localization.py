"""
Localization Decoders for pixel-level forgery localization.

LiteUNetDecoder:    Original bilinear upsampling decoder.
ImprovedDecoder:    Learned deconvolution + dropout + edge supervision.

Both take fused feature maps and upsample to input resolution,
producing a single-channel tamper probability mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """Single upsample block: Bilinear 2× → Conv3×3 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class LiteUNetDecoder(nn.Module):
    """
    Lightweight U-Net decoder: 256-ch 7×7 → 224×224 single-channel mask.

    Architecture:
        256 → 128 → 64 → 32 → 16 → 1 (sigmoid)
    Each stage doubles spatial resolution via bilinear upsampling.
    """

    def __init__(self, in_channels: int = 256, target_size: int = 224):
        super().__init__()
        self.target_size = target_size

        self.decoder = nn.Sequential(
            UpBlock(in_channels, 128),  # 7→14
            UpBlock(128, 64),  # 14→28
            UpBlock(64, 32),  # 28→56
            UpBlock(32, 16),  # 56→112
            UpBlock(16, 16),  # 112→224
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Fused features (B, 256, H, W) where H,W are small (e.g. 7×7)
        Returns:
            Tamper probability mask (B, 1, target_size, target_size)
        """
        out = self.decoder(x)
        out = self.head(out)
        # Ensure exact target size (handles rounding from odd feature maps)
        if out.shape[-1] != self.target_size or out.shape[-2] != self.target_size:
            out = nn.functional.interpolate(
                out,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return out


# ── Improved Decoder (Base Paper Techniques) ──────────────────────


class DeconvUpBlock(nn.Module):
    """
    Learned deconvolution block: ConvTranspose2d 2× → Conv3×3 → BN → ReLU → Dropout.

    Uses ConvTranspose2d instead of bilinear upsampling for sharper boundaries.
    Dropout2d between layers prevents decoder overfitting (from base paper).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.conv(self.up(x)))


def sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    """
    Extract edges from binary mask using Sobel operator.

    Used to generate edge supervision targets from GT masks —
    no extra annotation required.

    Args:
        mask: Binary mask (B, 1, H, W)
    Returns:
        Edge map (B, 1, H, W) with values in [0, 1]
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=mask.dtype, device=mask.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=mask.dtype, device=mask.device,
    ).view(1, 1, 3, 3)

    # Apply Sobel
    edge_x = F.conv2d(mask.float(), sobel_x, padding=1)
    edge_y = F.conv2d(mask.float(), sobel_y, padding=1)

    # Magnitude + normalize to [0, 1]
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge = torch.clamp(edge / (edge.amax(dim=[2, 3], keepdim=True) + 1e-8), 0, 1)

    return edge


class ImprovedDecoder(nn.Module):
    """
    Decoder with learned deconvolution, dropout, and edge supervision.

    Inspired by the base paper's decoder which uses deconvolution layers
    and dropout to produce sharper tamper masks.

    Architecture:
        256 → 128 → 64 → 32 → 16 → 16  (5 DeconvUpBlocks)
        mask_head: Conv2d(16, 1) + Sigmoid  → tamper mask
        edge_head: Conv2d(16, 1) + Sigmoid  → edge prediction

    The edge_head is trained with BCE loss against Sobel edges
    extracted from GT masks (free signal, no extra annotation).
    """

    def __init__(
        self,
        in_channels: int = 256,
        target_size: int = 384,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.target_size = target_size

        self.decoder = nn.Sequential(
            DeconvUpBlock(in_channels, 128, dropout=dropout),
            DeconvUpBlock(128, 64, dropout=dropout),
            DeconvUpBlock(64, 32, dropout=dropout),
            DeconvUpBlock(32, 16, dropout=dropout),
            DeconvUpBlock(16, 16, dropout=dropout),
        )

        # Mask head — tamper probability
        self.mask_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Edge head — boundary prediction (trained with Sobel edge targets)
        self.edge_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Fused features (B, 256, H, W) where H,W are small (e.g. 12×12)
        Returns:
            Tamper probability mask (B, 1, target_size, target_size)
        """
        features = self.decoder(x)  # (B, 16, ~384, ~384)

        mask = self.mask_head(features)

        # Ensure exact target size
        if mask.shape[-1] != self.target_size or mask.shape[-2] != self.target_size:
            mask = F.interpolate(
                mask,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return mask

    def predict_edge(self, x):
        """
        Predict edges from decoder features. Called separately during training.

        Args:
            x: Fused features (B, 256, H, W)
        Returns:
            Edge prediction (B, 1, target_size, target_size)
        """
        features = self.decoder(x)
        edge = self.edge_head(features)

        if edge.shape[-1] != self.target_size or edge.shape[-2] != self.target_size:
            edge = F.interpolate(
                edge,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return edge

    def forward_with_edge(self, x):
        """
        Forward pass returning both mask and edge predictions.
        Used during training with edge supervision.

        Args:
            x: Fused features (B, 256, H, W)
        Returns:
            mask: (B, 1, target_size, target_size)
            edge: (B, 1, target_size, target_size)
        """
        features = self.decoder(x)

        mask = self.mask_head(features)
        edge = self.edge_head(features)

        # Ensure exact target size for both
        if mask.shape[-1] != self.target_size or mask.shape[-2] != self.target_size:
            size = (self.target_size, self.target_size)
            mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
            edge = F.interpolate(edge, size=size, mode="bilinear", align_corners=False)

        return mask, edge
