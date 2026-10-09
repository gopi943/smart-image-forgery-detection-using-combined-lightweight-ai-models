"""
Feature Fusion Module with BN-Inception.

Aligns multi-stream encoder features to a shared dimension via 1×1 convolutions,
concatenates, applies CBAM attention, and reduces with BN-Inception block.

Supports 2-stream (semantic + texture), 3-stream (+ ELA), 4-stream
(+ self-correlation), or 5-stream (+ DCT) modes. BN-Inception fusion is toggleable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CBAM


class BNInceptionReduce(nn.Module):
    """
    BN-Inception block that replaces single 1×1 reduce.

    Three parallel conv branches at different kernel sizes capture
    both local and contextual cross-stream relationships:
        - 1×1: point-wise feature mixing
        - 3×3: local spatial context
        - 5×5: wider contextual relationships

    Inspired by the base paper's BN-Inception fusion classifier.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        ch1 = out_channels // 3           # 85
        ch2 = out_channels // 3           # 85
        ch3 = out_channels - ch1 - ch2    # 86

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch_1x1(x),
            self.branch_3x3(x),
            self.branch_5x5(x),
        ], dim=1)  # (B, out_channels, H, W)


class FeatureFusion(nn.Module):
    """
    Attention-based multi-stream feature fusion with residual connection.

    Pipeline:
        1. Project each stream → fusion_dim channels via 1×1 Conv+BN+ReLU
        2. Concatenate → num_streams × fusion_dim channels
        3. CBAM attention on concatenated features
        4. Reduce back to fusion_dim via BN-Inception (or 1×1 Conv+BN)
        5. Add residual (average of all projected streams) for gradient stability

    Supports 2-5 streams: semantic, texture, ELA, correlation, DCT.
    """

    def __init__(
        self,
        semantic_ch: int = 320,
        texture_ch: int = 576,
        ela_ch: int = 0,
        correlation_ch: int = 0,
        dct_ch: int = 0,
        fusion_dim: int = 256,
        cbam_reduction: int = 16,
        use_bn_inception: bool = False,
    ):
        super().__init__()

        self.has_ela = ela_ch > 0
        self.has_corr = correlation_ch > 0
        self.has_dct = dct_ch > 0
        self.num_streams = (
            2
            + (1 if self.has_ela else 0)
            + (1 if self.has_corr else 0)
            + (1 if self.has_dct else 0)
        )

        # 1×1 projection to shared dimension
        self.proj_semantic = nn.Sequential(
            nn.Conv2d(semantic_ch, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_texture = nn.Sequential(
            nn.Conv2d(texture_ch, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Optional ELA projection (3rd stream)
        if self.has_ela:
            self.proj_ela = nn.Sequential(
                nn.Conv2d(ela_ch, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )

        # Optional correlation projection (4th stream)
        if self.has_corr:
            self.proj_correlation = nn.Sequential(
                nn.Conv2d(correlation_ch, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )

        # Optional DCT projection (5th stream)
        if self.has_dct:
            self.proj_dct = nn.Sequential(
                nn.Conv2d(dct_ch, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )

        concat_ch = fusion_dim * self.num_streams

        # Attention on concatenated features
        self.cbam = CBAM(concat_ch, reduction=cbam_reduction)

        # Reduce back to fusion_dim
        if use_bn_inception:
            self.reduce = BNInceptionReduce(concat_ch, fusion_dim)
        else:
            self.reduce = nn.Sequential(
                nn.Conv2d(concat_ch, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
            )

    def forward(
        self,
        feat_semantic: torch.Tensor,
        feat_texture: torch.Tensor,
        feat_ela: torch.Tensor = None,
        feat_corr: torch.Tensor = None,
        feat_dct: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_semantic: (B, semantic_ch, H, W) from EfficientNet-B0
            feat_texture:  (B, texture_ch, H, W)  from MobileNetV3-Small
            feat_ela:      (B, ela_ch, H, W)      from ELA encoder (optional)
            feat_corr:     (B, corr_ch, H, W)     from SelfCorrelation (optional)
            feat_dct:      (B, dct_ch, H, W)      from DCTEncoder (optional)
        Returns:
            Fused features (B, fusion_dim, H, W)
        """
        target_size = feat_semantic.shape[2:]  # (H, W) reference for interpolation
        projections = []

        s = self.proj_semantic(feat_semantic)
        projections.append(s)

        t = self.proj_texture(feat_texture)
        if t.shape[2:] != target_size:
            t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
        projections.append(t)

        if self.has_ela and feat_ela is not None:
            e = self.proj_ela(feat_ela)
            if e.shape[2:] != target_size:
                e = F.interpolate(e, size=target_size, mode='bilinear', align_corners=False)
            projections.append(e)

        if self.has_corr and feat_corr is not None:
            c = self.proj_correlation(feat_corr)
            if c.shape[2:] != target_size:
                c = F.interpolate(c, size=target_size, mode='bilinear', align_corners=False)
            projections.append(c)

        if self.has_dct and feat_dct is not None:
            # DCT has different spatial size — resize to match others
            d = self.proj_dct(feat_dct)
            if d.shape[2:] != target_size:
                d = F.interpolate(d, size=target_size, mode='bilinear', align_corners=False)
            projections.append(d)

        # Concatenate all projections
        concat = torch.cat(projections, dim=1)

        # Residual = mean of all projections
        residual = sum(projections) / len(projections)

        # CBAM attention
        attended = self.cbam(concat)

        # Reduce to fusion_dim
        fused = self.reduce(attended)

        return F.relu(fused + residual, inplace=True)
