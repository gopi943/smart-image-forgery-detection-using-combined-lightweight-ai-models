"""
Classification Head with Attention Pooling.

AttentionPool → FC → ReLU → Dropout → FC → 3-class logits.
Softmax is NOT applied inside — CrossEntropyLoss includes it.
"""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    Learnable spatial attention pooling.

    Replaces simple GAP with a learned attention map that weights
    discriminative regions (e.g. forgery boundaries) higher.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Feature maps (B, C, H, W)
        Returns:
            Pooled features (B, C)
        """
        w = self.attn(x)                        # (B, 1, H, W)
        weighted = x * w                         # (B, C, H, W)
        return weighted.mean(dim=[2, 3])          # (B, C)


class ClassificationHead(nn.Module):
    """
    3-class classification head: Authentic / Copy-Move / Splicing.

    Takes fused feature maps and outputs raw logits.
    Supports both GAP and AttentionPool modes.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_attention_pool: bool = False,
    ):
        super().__init__()

        if use_attention_pool:
            self.pool = AttentionPool(in_channels)
        else:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # NOTE: nn.Flatten() kept at index 0 for checkpoint compatibility
        # with Phase 1 (where Flatten was inside Sequential). This ensures
        # Linear layers stay at indices 1 and 4, matching saved keys.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Fused feature maps (B, C, H, W)
        Returns:
            Raw logits (B, num_classes) — no softmax applied.
        """
        pooled = self.pool(x)  # (B, C)
        return self.classifier(pooled)
