"""
SRM (Spatial Rich Model) High-Pass Filter Module.

Extracts noise residuals using 6 fixed forensic kernels, then projects
through a learnable 1×1 Conv + BatchNorm to normalize for MobileNetV3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilter(nn.Module):
    """
    6 canonical SRM kernels (non-trainable) + learnable projection.

    Output is normalized via BatchNorm to be compatible with pretrained
    MobileNetV3-Small's internal BatchNorm running statistics.
    """

    def __init__(self):
        super().__init__()
        kernels = self._build_kernels()  # (6, 1, 5, 5)
        self.register_buffer("kernels", kernels)

        # Learnable projection: 6 SRM channels → 3 channels (RGB-like)
        self.project = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )

    def _build_kernels(self) -> torch.Tensor:
        """Build 6 canonical 5×5 SRM kernels."""
        # K1: 1st-order horizontal edge
        k1 = torch.zeros(5, 5)
        k1[2, 1] = 1.0
        k1[2, 2] = -1.0

        # K2: 1st-order vertical edge
        k2 = torch.zeros(5, 5)
        k2[1, 2] = 1.0
        k2[2, 2] = -1.0

        # K3: 2nd-order Laplacian
        k3 = torch.zeros(5, 5)
        k3[1, 2] = 1.0
        k3[2, 1] = 1.0
        k3[2, 2] = -4.0
        k3[2, 3] = 1.0
        k3[3, 2] = 1.0

        # K4: 3rd-order square
        k4 = torch.zeros(5, 5)
        k4[1, 1] = -1.0
        k4[1, 2] = 2.0
        k4[1, 3] = -1.0
        k4[2, 1] = 2.0
        k4[2, 2] = -4.0
        k4[2, 3] = 2.0
        k4[3, 1] = -1.0
        k4[3, 2] = 2.0
        k4[3, 3] = -1.0

        # K5: SPAM horizontal (pixel adjacency)
        k5 = torch.zeros(5, 5)
        k5[2, 0] = 0.0
        k5[2, 1] = -1.0
        k5[2, 2] = 3.0
        k5[2, 3] = -3.0
        k5[2, 4] = 1.0

        # K6: 5×5 high-pass filter (normalized)
        k6 = -torch.ones(5, 5)
        k6[2, 2] = 24.0
        k6 = k6 / 8.0

        kernels = torch.stack([k1, k2, k3, k4, k5, k6])  # (6, 5, 5)
        return kernels.unsqueeze(1)  # (6, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W), typically C=3 RGB.
        Returns:
            Normalized residual tensor (B, 3, H, W) ready for MobileNetV3.
        """
        B, C, H, W = x.shape
        residuals = []
        for ch in range(C):
            single_ch = x[:, ch : ch + 1, :, :]  # (B, 1, H, W)
            out = F.conv2d(single_ch, self.kernels, padding=2)  # (B, 6, H, W)
            residuals.append(out)

        # Average across input channels → (B, 6, H, W)
        residual = torch.stack(residuals, dim=0).mean(dim=0)
        # Project + normalize → (B, 3, H, W)
        return self.project(residual)
