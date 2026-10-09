"""
StudentNet — Lightweight Single-Encoder Model for Mobile Deployment.

Architecture: MobileNetV3-Small → GAP → FC → 3 classes
~2.5M params, runs in ~30ms on mobile GPU.

Trained via knowledge distillation from ForensicNet teacher.
Input: single RGB image. The teacher's multi-stream forensic evidence
is distilled into the student's learned features.
"""

import torch
import torch.nn as nn
import timm


class StudentNet(nn.Module):
    """
    Lightweight mobile-optimized forgery detection model.

    Single MobileNetV3-Small backbone — 10× smaller and 5× faster
    than the full 3-stream ForensicNet teacher.

    Designed to be trained via knowledge distillation where the
    teacher's multi-stream and CLIP-assisted knowledge gets compressed
    into this single-stream student.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, num_classes=0,
        )
        self.feature_dim = self.backbone.num_features  # 576

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: RGB image (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone(x)  # (B, 576)
        return self.head(features)
