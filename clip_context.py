"""
FrozenCLIPContext — Frozen CLIP Feature Extractor for ForensicLens v2.1.

Uses OpenCLIP to load a pretrained CLIP vision encoder. The encoder is fully
frozen (no gradients). A small trainable projection head maps CLIP features
to a compact vector used by the classification evidence heads.

Usage:
    ctx = FrozenCLIPContext(model_name='ViT-B-16', pretrained='openai')
    preprocess = ctx.preprocess  # Use this in the dataloader for x_clip
    clip_vec = ctx(x_clip)       # [B, out_dim]

IMPORTANT: x_clip MUST be preprocessed with ctx.preprocess, NOT with
ImageNet normalization. CLIP uses different pixel statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCLIPContext(nn.Module):
    """
    Frozen CLIP visual encoder + trainable projection.

    The CLIP model is loaded once and fully frozen. Only the projection
    head (Linear → LayerNorm → GELU) receives gradients during training.

    Attributes:
        preprocess: The torchvision transform returned by OpenCLIP that
                    MUST be used to preprocess images before encode_image().
        clip_dim:   Output dimension of the CLIP visual encoder (512 for ViT-B/16).
        out_dim:    Output dimension after projection (default 128).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        out_dim: int = 128,
    ):
        super().__init__()

        # Late import to avoid hard dependency at module level
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for CLIP integration. "
                "Install with: pip install open-clip-torch>=2.24.0"
            )

        # Prefer the configured pretrained weights, but fall back to an
        # uninitialized local graph so checkpoint weights can still restore the
        # encoder during offline inference.
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=None
            )

        # Freeze ALL CLIP parameters
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get CLIP visual output dimension
        self.clip_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/16
        self.out_dim = out_dim

        # Trainable projection: maps CLIP features to compact vector
        self.proj = nn.Sequential(
            nn.Linear(self.clip_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x_clip: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features and project to out_dim.

        Args:
            x_clip: Images preprocessed with self.preprocess [B, 3, 224, 224]

        Returns:
            Projected CLIP features [B, out_dim]
        """
        # CLIP inference with no gradients (frozen)
        with torch.no_grad():
            z = self.clip_model.encode_image(x_clip)  # [B, clip_dim]
            z = F.normalize(z, dim=-1)                 # L2 normalize

        # Trainable projection (gradients flow here)
        return self.proj(z.float())  # [B, out_dim]

    def train(self, mode: bool = True):
        """Override train() to keep CLIP always in eval mode."""
        super().train(mode)
        self.clip_model.eval()  # CLIP stays eval regardless
        return self

    @property
    def num_trainable_params(self) -> int:
        """Count trainable parameters (projection only)."""
        return sum(p.numel() for p in self.proj.parameters() if p.requires_grad)

    @property
    def num_frozen_params(self) -> int:
        """Count frozen CLIP parameters."""
        return sum(p.numel() for p in self.clip_model.parameters())
