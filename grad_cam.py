"""
Forensic Attribution Engine — Multi-method localization for image forgery detection.

Uses the industry-standard `pytorch-grad-cam` library with multiple attribution
methods (EigenCAM, HiResCAM) targeting the fusion attention and backbone layers,
producing sharper and more accurate localization than a single Grad-CAM.

Also includes occlusion sensitivity for causal verification.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple

# ── pytorch-grad-cam library ──────────────────────────────────
from pytorch_grad_cam import (
    EigenCAM,
    HiResCAM,
    GradCAMPlusPlus,
    AblationCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ═══════════════════════════════════════════════════════════════
#  Custom Target for Two-Head Forensic Model
# ═══════════════════════════════════════════════════════════════


class ForensicTarget:
    """
    Custom target for pytorch-grad-cam that extracts the correct
    head logit from our two-head ForensicNet model.

    Since ForensicNetWrapper now returns a single [B, 1] tensor,
    this just returns the scalar from position [0, 0].
    """

    def __init__(self, head: str = "cm"):
        self.head = head

    def __call__(self, model_output):
        # model_output is now a [B, 1] tensor from our wrapper
        if model_output.dim() >= 2:
            return model_output[0, 0]
        elif model_output.dim() == 1:
            return model_output[0]
        return model_output


# ═══════════════════════════════════════════════════════════════
#  Wrapper to make ForensicNet compatible with pytorch-grad-cam
# ═══════════════════════════════════════════════════════════════


class ForensicNetWrapper(torch.nn.Module):
    """
    Wraps ForensicNet to work with pytorch-grad-cam.

    pytorch-grad-cam expects the model to return a simple tensor,
    NOT a tuple. This wrapper calls the raw model and returns
    only the target head logit as a [B, 1] tensor.
    """

    def __init__(self, model, x_clip: Optional[torch.Tensor] = None, head: str = "cm"):
        super().__init__()
        self.model = model
        self.x_clip = x_clip
        self.head = head

    def forward(self, x_forensic: torch.Tensor):
        out = self.model(x_forensic, self.x_clip)
        # out = (cm_logit, sp_logit, tamper_logit, mask, extras)
        if self.head == "cm":
            logit = out[0]
        else:
            logit = out[1]
        # Return as [B, 1] so pytorch-grad-cam can process it
        return logit.reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════
#  Utility: Percentile Normalization
# ═══════════════════════════════════════════════════════════════


def _percentile_normalize(
    cam: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0
) -> np.ndarray:
    """
    Robust percentile normalization instead of fragile min-max.
    Prevents noise amplification that plagues standard Grad-CAM.
    """
    cam = np.nan_to_num(cam.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.clip(cam, 0.0, None)

    lo = float(np.percentile(cam, low_pct))
    hi = float(np.percentile(cam, high_pct))

    if hi - lo > 1e-8:
        cam = np.clip((cam - lo) / (hi - lo), 0.0, 1.0)
    elif cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    else:
        return np.zeros_like(cam, dtype=np.float32)

    return cam.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  Main Engine: ForensicAttributionEngine
# ═══════════════════════════════════════════════════════════════


def _resolve_target_layers(model) -> List[torch.nn.Module]:
    """
    Automatically select target layers for attribution.

    Returns layers in order:
    [0] fusion.cbam — high-level attended features (coarse)
    [1] backbone last block — semantic features (medium)
    [2] backbone mid block — spatial features (fine-grained)
    """
    layers = []

    # Level 1: CBAM attention (attended cross-stream features)
    if hasattr(model, "fusion") and hasattr(model.fusion, "cbam"):
        layers.append(model.fusion.cbam)

    # Level 2 & 3: semantic encoder blocks
    if hasattr(model, "stream1"):
        stream1 = model.stream1
        backbone = getattr(stream1, "backbone", stream1)
        if hasattr(backbone, "features"):
            features = backbone.features
            layers.append(features[-1])   # Last block (high-level)
            if len(features) >= 4:
                layers.append(features[-3])  # Mid block (spatially rich)
        elif hasattr(backbone, "blocks"):
            blocks = backbone.blocks
            layers.append(blocks[-1])
            if len(blocks) >= 3:
                layers.append(blocks[-3])

    if not layers:
        if hasattr(model, "fusion") and hasattr(model.fusion, "reduce"):
            layers.append(model.fusion.reduce)

    return layers


class ForensicAttributionEngine:
    """
    Production-grade attribution engine for forensic localization.

    Strategy:
    1. HiResCAM on fusion.cbam — pixel-precise gradient × activation (primary)
    2. GradCAM++ on fusion.cbam — better small-region localization
    3. EigenCAM on mid-level backbone — fine-grained spatial features
    4. Morphological post-processing for clean boundaries

    Fusion: weighted ensemble with percentile-normalization per-method.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List[torch.nn.Module]] = None,
    ):
        self.model = model
        self.target_layers = target_layers or _resolve_target_layers(model)
        self._wrapper = None

    def generate(
        self,
        x_forensic: torch.Tensor,
        x_clip: Optional[torch.Tensor] = None,
        target_head: str = "cm",
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a fused attribution map using multiple methods.

        Args:
            x_forensic: Input tensor [1, 3, H, W]
            x_clip: Optional CLIP tensor [1, 3, 224, 224]
            target_head: "cm" or "sp"
            image_size: Output size (H, W). Defaults to input spatial dims.

        Returns:
            Attribution heatmap (H, W) in [0, 1], float32.
        """
        if image_size is None:
            image_size = (x_forensic.shape[2], x_forensic.shape[3])

        self.model.eval()
        wrapper = ForensicNetWrapper(self.model, x_clip, head=target_head)
        targets = [ForensicTarget(head=target_head)]

        cams = []
        weights = []

        # The primary layer for gradient-based methods (fusion.cbam or best available)
        primary_layer = self.target_layers[:1]
        # Mid-level layer for spatial precision (if available)
        mid_layer = [self.target_layers[2]] if len(self.target_layers) > 2 else primary_layer

        # ── Method 1: HiResCAM — pixel-precise, gradient-based (primary) ──
        try:
            with HiResCAM(
                model=wrapper,
                target_layers=primary_layer,
            ) as cam_engine:
                cam = cam_engine(
                    input_tensor=x_forensic,
                    targets=targets,
                )
            cam = cam[0]
            cam = cv2.resize(cam, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)
            cam = _percentile_normalize(cam)
            cams.append(cam)
            weights.append(0.40)
        except Exception as e:
            print(f"[DEBUG] HiResCAM failed: {e}")

        # ── Method 2: GradCAM++ — better for small/localized regions ──
        try:
            with GradCAMPlusPlus(
                model=wrapper,
                target_layers=primary_layer,
            ) as cam_engine:
                cam = cam_engine(
                    input_tensor=x_forensic,
                    targets=targets,
                )
            cam = cam[0]
            cam = cv2.resize(cam, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)
            cam = _percentile_normalize(cam)
            cams.append(cam)
            weights.append(0.30)
        except Exception as e:
            print(f"[DEBUG] GradCAM++ failed: {e}")

        # ── Method 3: EigenCAM on mid-level backbone (spatial detail) ──
        try:
            with EigenCAM(
                model=wrapper,
                target_layers=mid_layer,
            ) as cam_engine:
                cam = cam_engine(
                    input_tensor=x_forensic,
                    targets=None,  # EigenCAM is gradient-free
                    eigen_smooth=True,
                )
            cam = cam[0]
            cam = cv2.resize(cam, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)
            cam = _percentile_normalize(cam)
            cams.append(cam)
            weights.append(0.30)
        except Exception as e:
            print(f"[DEBUG] EigenCAM-mid failed: {e}")

        if not cams:
            return np.zeros(image_size, dtype=np.float32)

        # ── Fuse all methods ──
        weight_sum = sum(weights)
        fused = np.zeros(image_size, dtype=np.float32)
        for cam, w in zip(cams, weights):
            fused += cam * (w / weight_sum)

        # ── Post-processing for cleaner boundaries ──
        fused = _percentile_normalize(fused, low_pct=5.0, high_pct=95.0)

        # Morphological cleanup: remove noise and smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = (fused > 0.3).astype(np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        fused = fused * mask_binary.astype(np.float32)

        # Smooth with Gaussian blur
        fused = cv2.GaussianBlur(fused, (7, 7), 0)
        fused = _percentile_normalize(fused, low_pct=2.0, high_pct=98.0)

        return fused


    def generate_guided(
        self,
        x_forensic: torch.Tensor,
        x_clip: Optional[torch.Tensor] = None,
        target_head: str = "cm",
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a guided attribution by combining GradCAM++ with
        input gradient magnitude for sharper boundaries.
        """
        if image_size is None:
            image_size = (x_forensic.shape[2], x_forensic.shape[3])

        self.model.eval()
        wrapper = ForensicNetWrapper(self.model, x_clip, head=target_head)
        targets = [ForensicTarget(head=target_head)]

        # GradCAM++ for the base map
        try:
            with GradCAMPlusPlus(
                model=wrapper,
                target_layers=self.target_layers[:1],
            ) as cam_engine:
                cam = cam_engine(
                    input_tensor=x_forensic,
                    targets=targets,
                    aug_smooth=True,
                )
            cam = cam[0]
            cam = cv2.resize(cam, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)
            cam = _percentile_normalize(cam)
        except Exception:
            cam = np.zeros(image_size, dtype=np.float32)

        # Input gradient magnitude for sharpening
        # Use wrapper (returns [B,1] tensor) so ForensicTarget works correctly
        x_grad = x_forensic.detach().clone().requires_grad_(True)
        wrapper_output = wrapper(x_grad)
        target = ForensicTarget(head=target_head)
        score = target(wrapper_output)
        wrapper.zero_grad(set_to_none=True)
        score.backward()

        if x_grad.grad is not None:
            grad = x_grad.grad.detach().abs().mean(dim=1, keepdim=True)
            grad = F.interpolate(
                grad, size=image_size, mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            grad = _percentile_normalize(grad, low_pct=40.0, high_pct=99.5)
        else:
            grad = np.zeros(image_size, dtype=np.float32)

        # Fuse: CAM provides region, gradient provides boundaries
        fused = cam * 0.6 + (cam * grad) * 0.4
        fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=1.5)
        fused = _percentile_normalize(fused, low_pct=2.0, high_pct=99.0)

        return fused

    def generate_occlusion(
        self,
        x_forensic: torch.Tensor,
        x_clip: Optional[torch.Tensor] = None,
        target_head: str = "cm",
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 12,
    ) -> np.ndarray:
        """
        Batched occlusion sensitivity map — causally tests each region's
        contribution by masking it out and measuring score drop.

        Slower than CAM methods but most trustworthy for forensic evidence.
        """
        self.model.eval()

        if image_size is None:
            image_size = (x_forensic.shape[2], x_forensic.shape[3])

        # Use wrapper so output is [B, 1] — compatible with ForensicTarget
        wrapper = ForensicNetWrapper(self.model, x_clip, head=target_head)
        target = ForensicTarget(head=target_head)

        with torch.no_grad():
            base_output = wrapper(x_forensic)
            base_score = target(base_output).float()

        _, _, h, w = x_forensic.shape
        windows = [
            (max(56, min(h, w) // 4), max(32, min(h, w) // 7)),
            (max(88, min(h, w) // 3), max(44, min(h, w) // 6)),
        ]

        score_map = torch.zeros((h, w), dtype=torch.float32, device=x_forensic.device)
        count_map = torch.zeros((h, w), dtype=torch.float32, device=x_forensic.device)
        patch_batches = []

        def flush_batch():
            nonlocal patch_batches, score_map, count_map
            if not patch_batches:
                return

            forensic_batch = torch.cat([item[0] for item in patch_batches], dim=0)
            # Create a fresh wrapper for the batch (x_clip stays the same)
            batch_wrapper = ForensicNetWrapper(self.model, x_clip, head=target_head)
            output = batch_wrapper(forensic_batch)

            # output is [B, 1] — extract scores for each sample
            scores = output.squeeze(1).float()
            drops = torch.clamp(base_score - scores, min=0.0).detach()

            for i, (_, _, y0, y1, x0, x1) in enumerate(patch_batches):
                score_map[y0:y1, x0:x1] += float(drops[i].item())
                count_map[y0:y1, x0:x1] += 1.0
            patch_batches = []

        for window, stride in windows:
            y_positions = list(range(0, max(h - window, 0) + 1, stride))
            x_positions = list(range(0, max(w - window, 0) + 1, stride))
            if not y_positions or y_positions[-1] != h - window:
                y_positions.append(max(h - window, 0))
            if not x_positions or x_positions[-1] != w - window:
                x_positions.append(max(w - window, 0))

            for y0 in y_positions:
                for x0 in x_positions:
                    y1 = min(y0 + window, h)
                    x1 = min(x0 + window, w)

                    masked_f = x_forensic.detach().clone()
                    masked_f[:, :, y0:y1, x0:x1] = 0.0

                    patch_batches.append((masked_f, None, y0, y1, x0, x1))
                    if len(patch_batches) >= max(int(batch_size), 1):
                        flush_batch()

        flush_batch()

        count_map = torch.clamp(count_map, min=1.0)
        occ = (score_map / count_map).detach().cpu().numpy()
        occ = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), sigmaX=3.5, sigmaY=3.5)
        occ = _percentile_normalize(occ, low_pct=40.0, high_pct=99.5)

        if occ.shape != image_size:
            occ = cv2.resize(
                occ, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC
            )
        return occ.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  Backward-Compatible GradCAM Wrapper
# ═══════════════════════════════════════════════════════════════


class GradCAM:
    """
    Backward-compatible wrapper that delegates to ForensicAttributionEngine.

    Maintains the same interface as the original GradCAM class so existing
    code in predict.py and api_server.py continues to work.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        target_layers = _resolve_target_layers(model)
        self._engine = ForensicAttributionEngine(model, target_layers)
        # These exist for compatibility — hooks are managed by pytorch-grad-cam
        self._hooks_active = True

    def generate(
        self,
        input_tensor,
        x_clip=None,
        target_head=None,
        target_class=None,
        image_size=None,
    ) -> np.ndarray:
        """Generate attribution map (backward-compatible API)."""
        if target_head is None:
            # Auto-select based on forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor, x_clip) if x_clip is not None else self.model(input_tensor)
                cm_logit, sp_logit = outputs[0], outputs[1]
                target_head = "cm" if cm_logit.max().item() >= sp_logit.max().item() else "sp"

        return self._engine.generate(
            input_tensor,
            x_clip=x_clip,
            target_head=target_head,
            image_size=image_size,
        )

    def generate_guided(
        self,
        input_tensor,
        x_clip=None,
        target_head=None,
        target_class=None,
        image_size=None,
        smooth_samples: int = 3,
        noise_std: float = 0.03,
    ) -> np.ndarray:
        """Generate guided attribution (backward-compatible API)."""
        if target_head is None:
            target_head = "cm"

        return self._engine.generate_guided(
            input_tensor,
            x_clip=x_clip,
            target_head=target_head,
            image_size=image_size,
        )

    def generate_occlusion(
        self,
        input_tensor,
        x_clip=None,
        target_head=None,
        target_class=None,
        image_size=None,
        batch_size: int = 12,
    ) -> np.ndarray:
        """Generate occlusion sensitivity map (backward-compatible API)."""
        if target_head is None:
            target_head = "cm"

        return self._engine.generate_occlusion(
            input_tensor,
            x_clip=x_clip,
            target_head=target_head,
            image_size=image_size,
            batch_size=batch_size,
        )

    def remove_hooks(self):
        """No-op for backward compatibility — pytorch-grad-cam manages hooks."""
        self._hooks_active = False

    def __del__(self):
        pass
