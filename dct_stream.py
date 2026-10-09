"""
DCT Frequency Stream — Direct JPEG Compression Artifact Analysis.

Converts images to DCT (Discrete Cosine Transform) coefficients and
processes them through a lightweight CNN. This directly exposes
double-compression artifacts that ELA only approximates.

The key insight: when you splice image A (JPEG compressed 3x) into
image B (JPEG compressed 1x) and re-save, the DCT coefficient
distribution in the spliced region has a fundamentally different
statistical signature. ELA converts back to pixels, losing this info.
DCT keeps the raw frequency data.

Reference:
    Kwon et al., "CAT-Net: Compression Artifact Tracing Network",
    IJCV 2022.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dct_basis(block_size: int = 8) -> torch.Tensor:
    """
    Build the 2D DCT-II basis matrix for an NxN block.

    Returns:
        Tensor of shape (N*N, 1, N, N) — each slice is one basis pattern.
    """
    n = block_size
    basis = torch.zeros(n, n, n, n)  # (u, v, x, y)
    for u in range(n):
        for v in range(n):
            alpha_u = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
            alpha_v = math.sqrt(1.0 / n) if v == 0 else math.sqrt(2.0 / n)
            for x in range(n):
                for y in range(n):
                    basis[u, v, x, y] = (
                        alpha_u * alpha_v
                        * math.cos((2 * x + 1) * u * math.pi / (2 * n))
                        * math.cos((2 * y + 1) * v * math.pi / (2 * n))
                    )
    # Reshape to (N*N, 1, N, N) for use as conv2d filters
    return basis.reshape(n * n, 1, n, n)


class DCTEncoder(nn.Module):
    """
    Converts an image to DCT coefficients per 8×8 block, then processes
    through a lightweight CNN to extract compression artifact features.

    Pipeline:
        RGB → Grayscale → 8×8 DCT (64 channels) → CNN → Feature map

    Args:
        out_channels: Output feature dimension (default: 128).
        block_size: DCT block size, must be 8 for JPEG analysis.
    """

    def __init__(self, out_channels: int = 128, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        num_dct = block_size * block_size  # 64 coefficients per block

        # Fixed DCT basis filters (not trainable — mathematical constants)
        self.register_buffer('dct_basis', _build_dct_basis(block_size))

        # Lightweight CNN to process 64-channel DCT coefficient maps
        self.encoder = nn.Sequential(
            # 64 → 128 channels
            nn.Conv2d(num_dct, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 → 128, downsample 2×
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 → out_channels, downsample 2×
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image tensor (B, 3, H, W), values in [0, 1].
        Returns:
            DCT feature map (B, out_channels, H', W') where
            H' ≈ H / (block_size * 4), W' ≈ W / (block_size * 4).
        """
        b, c, h, w = x.shape

        # Convert to grayscale (Y channel — where JPEG artifacts are strongest)
        # ITU-R BT.601 weights
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Pad to multiple of block_size
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        if pad_h > 0 or pad_w > 0:
            gray = F.pad(gray, (0, pad_w, 0, pad_h), mode='reflect')

        # Apply DCT basis as convolution (stride=block_size → non-overlapping blocks)
        # Force FP32 to prevent AMP FP16 overflow on DCT coefficients
        gray_f32 = gray.float()
        dct_coeffs = F.conv2d(
            gray_f32, self.dct_basis.float(), stride=self.block_size
        )  # (B, 64, H/8, W/8)

        # Normalize DCT coefficients — per-instance standardization
        # (zero-mean, unit-variance per sample → stable BN + dense gradients)
        mean = dct_coeffs.mean(dim=(1, 2, 3), keepdim=True)
        std = dct_coeffs.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        dct_coeffs = (dct_coeffs - mean) / std

        # Process through CNN
        features = self.encoder(dct_coeffs)
        return features
