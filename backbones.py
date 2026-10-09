"""
Multi-Encoder Backbone Feature Extractors.

Stream 1: EfficientNet (configurable B0/B2) — global semantic analysis.
Stream 2: MobileNetV3-Small — fine-grained texture/noise analysis.
Stream 3: EfficientNet (same as Stream 1) — ELA analysis.

Phase 2 additions:
- Configurable backbone (B0/B2 via cfg.backbone_name)
- Stochastic Depth (drop_path) for regularization
"""

import torch.nn as nn
import torch.nn.functional as F
import timm


class SemanticEncoder(nn.Module):
    """
    EfficientNet feature extractor for global semantic analysis.
    Backbone is configurable via model_name parameter.
    """

    def __init__(self, pretrained: bool = True, model_name: str = "efficientnet_b0",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        features = self.backbone(x)
        return features[-1]  # Last stage feature map


class MultiScaleSemanticEncoder(nn.Module):
    """
    EfficientNet-B0 with FPN-style multi-scale feature fusion.

    Uses stages 2 (48×48×40), 3 (24×24×112), 4 (12×12×320).
    Lateral 1×1 convolutions align channel dimensions, then features
    are downsampled to 12×12 and summed (top-down FPN addition).

    Output shape is identical to SemanticEncoder: (B, 320, 12, 12).
    This enables drop-in replacement with full backward compatibility.
    """

    def __init__(self, pretrained: bool = True, model_name: str = "efficientnet_b0",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            drop_path_rate=drop_path_rate,
        )
        channels = self.backbone.feature_info.channels()  # e.g. [16, 24, 40, 112, 320]
        self.out_channels = channels[-1]  # 320

        # Lateral projections to align channels from earlier stages
        self.lateral_2 = nn.Sequential(
            nn.Conv2d(channels[2], self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.lateral_3 = nn.Sequential(
            nn.Conv2d(channels[3], self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        # Stage 4 already has out_channels — no projection needed

    def forward(self, x):
        features = self.backbone(x)  # 5 stages
        feat_4 = features[4]  # (B, 320, 12, 12)
        target_size = feat_4.shape[-1]  # 12

        # Lateral projections + downsample to stage 4 spatial size
        feat_3 = self.lateral_3(features[3])  # (B, 320, 24, 24)
        feat_3 = F.adaptive_avg_pool2d(feat_3, target_size)  # (B, 320, 12, 12)

        feat_2 = self.lateral_2(features[2])  # (B, 320, 48, 48)
        feat_2 = F.adaptive_avg_pool2d(feat_2, target_size)  # (B, 320, 12, 12)

        # FPN top-down sum
        return feat_4 + feat_3 + feat_2  # (B, 320, 12, 12)


class TextureEncoder(nn.Module):
    """
    MobileNetV3-Small feature extractor for texture/noise analysis.
    Output: (B, 576, H', W').
    """

    def __init__(self, pretrained: bool = True, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, features_only=True,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]  # 576

    def forward(self, x):
        features = self.backbone(x)
        return features[-1]  # Last stage feature map


class ELAEncoder(nn.Module):
    """
    EfficientNet feature extractor for ELA (Error Level Analysis) images.

    Uses the same backbone as SemanticEncoder (configurable via model_name).
    """

    def __init__(self, pretrained: bool = True, model_name: str = "efficientnet_b0",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        features = self.backbone(x)
        return features[-1]  # Last stage feature map
