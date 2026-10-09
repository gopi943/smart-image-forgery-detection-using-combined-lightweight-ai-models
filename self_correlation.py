"""
Self-Correlation Module for Copy-Move Forgery Detection.

Detects duplicated regions by computing within-image feature similarity.

Pipeline:
    1. L2-normalize feature map along channel dimension
    2. Compute NxN cosine similarity matrix (N = H*W spatial positions)
    3. Percentile pooling: keep only top-k% similarities per position
    4. Reshape correlation to spatial feature map (B, N, H, W)
    5. Refine with 2 conv layers + BN + ReLU + Dropout

Inspired by the CNN-CenSurE base paper's Channel-2 self-correlation
module, which computes similarity between spatial regions in feature
maps to detect duplicated (copy-moved) content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCorrelation(nn.Module):
    """
    Within-image self-correlation for copy-move detection.

    For each spatial position in the feature map, computes cosine
    similarity with every other position. Duplicated regions produce
    high correlation peaks. Percentile pooling suppresses noise.

    Input:  (B, C, H, W) — encoder feature map (e.g. 320ch at 12×12)
    Output: (B, out_channels, H, W) — correlation features (e.g. 128ch at 12×12)
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 128,
        n_spatial: int = 144,
        topk_pct: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.topk_pct = topk_pct
        self.out_channels = out_channels

        # Conv refinement: n_spatial → mid → out_channels
        # n_spatial = H*W of input feature map (default 144 for 12×12 from 384×384 input)
        mid = 64
        self.refine = nn.Sequential(
            nn.Conv2d(n_spatial, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Feature map (B, C, H, W) from an encoder.
        Returns:
            Correlation features (B, out_channels, H, W).
        """
        B, C, H, W = feat.shape
        N = H * W

        # 1. L2-normalize along channel dimension
        # Force FP32 to prevent AMP FP16 overflow in bmm
        feat_flat = feat.float().view(B, C, N)  # (B, C, N)
        feat_norm = F.normalize(feat_flat, dim=1)  # (B, C, N)

        # 2. Cosine similarity: (B, N, N)
        corr = torch.bmm(feat_norm.permute(0, 2, 1), feat_norm)  # (B, N, N)

        # 3. Percentile pooling — zero out bottom (1 - topk_pct)
        k = max(2, int(N * self.topk_pct))  # at least 2 to avoid empty
        topk_vals = corr.topk(k, dim=-1).values  # (B, N, k)
        threshold = topk_vals[:, :, -1:]  # (B, N, 1) — k-th largest value
        corr = corr * (corr >= threshold).float()

        # 4. Reshape to spatial: each position's correlation vector → spatial map
        corr = corr.view(B, N, H, W)  # (B, N, H, W) e.g. (B, 144, 12, 12)

        # 5. Conv refinement
        return self.refine(corr)  # (B, out_channels, H, W)
