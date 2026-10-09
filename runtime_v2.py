"""
Shared runtime helpers for the ForensicLens v2.1 pipeline.

Keeps checkpoint loading, calibration, and decision policy aligned across
CLI inference, deployment, and export utilities.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import torch

from config import Config
from models.forensic_net import ForensicNet


CHECKPOINT_CONFIG_KEYS = [
    "backbone_name",
    "fusion_dim",
    "dropout_rate",
    "use_bayar",
    "use_self_correlation",
    "use_dct",
    "use_clip",
    "clip_model",
    "clip_pretrained",
    "clip_proj_dim",
    "use_multiscale",
    "use_bn_inception",
    "use_improved_decoder",
    "use_localization",
    "use_tamper_head",
    "use_reliability_head",
    "use_corr_consistency",
]

DEFAULT_THRESHOLDS = {
    "cm_threshold": 0.5,
    "sp_threshold": 0.5,
    "cm_delta": 0.0,
    "sp_delta": 0.0,
}


def apply_model_config(cfg: Config, config_dict: Dict[str, Any]) -> None:
    """Restore architecture-critical config flags from a checkpoint."""
    for key in CHECKPOINT_CONFIG_KEYS:
        if key in config_dict:
            setattr(cfg, key, config_dict[key])


def resolve_checkpoint_path(cfg: Config, checkpoint_path: Optional[str] = None) -> str:
    """Resolve an explicit checkpoint path or fall back to common v2 names."""
    if checkpoint_path:
        return checkpoint_path

    candidates = [
        os.path.join(cfg.get_path("checkpoint_dir"), "best_model_v2.pth"),
        os.path.join(cfg.get_path("checkpoint_dir"), "best_384.pth"),
        os.path.join(cfg.get_path("checkpoint_dir"), "best_320.pth"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a v2 checkpoint. Checked: " + ", ".join(candidates)
    )


def load_model_bundle(
    checkpoint_path: Optional[str] = None,
    cfg: Optional[Config] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ForensicNet, Config, torch.device, Dict[str, Any], str]:
    """Load a v2 checkpoint and rebuild the matching model/config pair."""
    cfg = cfg or Config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_path = resolve_checkpoint_path(cfg, checkpoint_path)

    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
    apply_model_config(cfg, ckpt.get("config", {}))
    if "resolution" in ckpt:
        cfg.image_size = int(ckpt["resolution"])

    model = ForensicNet(cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, cfg, device, ckpt, resolved_path


def load_calibration_artifacts(
    checkpoint_path: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    """Load calibrators.pkl and thresholds.json next to a checkpoint if present."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    calibrators = None
    thresholds = dict(DEFAULT_THRESHOLDS)

    calibrator_path = os.path.join(checkpoint_dir, "calibrators.pkl")
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as handle:
            calibrators = pickle.load(handle)

    threshold_path = os.path.join(checkpoint_dir, "thresholds.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as handle:
            thresholds.update(json.load(handle))

    return calibrators, thresholds


def calibrate_prob(
    raw_logit: float,
    calibrators: Optional[Dict[str, Any]],
    head: str,
) -> float:
    """Calibrate a raw head logit with isotonic/Platt models when available."""
    raw_prob = float(torch.sigmoid(torch.tensor(raw_logit)).item())
    if calibrators is None:
        return raw_prob

    calibrator = calibrators.get(f"{head}_calibrator")
    if calibrator is None:
        return raw_prob

    method = calibrators.get(f"{head}_method", "isotonic")
    try:
        if method == "isotonic":
            return float(calibrator.predict([raw_prob])[0])
        return float(calibrator.predict_proba([[raw_logit]])[0, 1])
    except Exception:
        return raw_prob


def decision_policy(
    cm_prob: float,
    sp_prob: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[str, float]:
    """Apply the calibrated two-head decision policy used in deployment."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    cm_margin = cm_prob - sp_prob
    sp_margin = sp_prob - cm_prob

    cm_pos = (
        cm_prob >= thresholds["cm_threshold"]
        and cm_margin >= thresholds["cm_delta"]
    )
    sp_pos = (
        sp_prob >= thresholds["sp_threshold"]
        and sp_margin >= thresholds["sp_delta"]
    )

    if not cm_pos and not sp_pos:
        return "Authentic", max(1.0 - cm_prob, 1.0 - sp_prob)
    if cm_pos and not sp_pos:
        return "CopyMove", cm_prob
    if sp_pos and not cm_pos:
        return "Splicing", sp_prob
    if cm_prob >= sp_prob:
        return "CopyMove", cm_prob
    return "Splicing", sp_prob


def authentic_probability(cm_prob: float, sp_prob: float) -> float:
    """Conservative authentic probability derived from subtype evidence."""
    return max(0.0, 1.0 - max(cm_prob, sp_prob))


def compute_confidence(
    pred_label: str,
    cm_prob: float,
    sp_prob: float,
    thresholds: Optional[Dict[str, float]] = None,
    reliability: Optional[float] = None,
) -> float:
    """
    Compute a meaningful confidence score that reflects *how certain*
    we are about the prediction.

    For authentic predictions:
      - Primarily driven by how LOW the manipulation signals are
      - When both CM and SP are under 10%, confidence is high
      - Only penalized when signals are genuinely ambiguous (near 50%)

    For tampered predictions:
      - Based on probability, margin above threshold, and head separation

    Args:
        pred_label: "Authentic", "CopyMove", or "Splicing"
        cm_prob: Calibrated copy-move probability
        sp_prob: Calibrated splicing probability
        thresholds: Dict with cm_threshold, sp_threshold, cm_delta, sp_delta
        reliability: Optional reliability head output [0, 1]

    Returns:
        Confidence score in [0, 1]
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Reliability head is NOT used for confidence because the checkpoint
    # analysis proved ALL decoder BatchNorms are NaN (only 2 epochs of
    # Stage 2 training). The reliability head output is garbage noise.
    # When the model is retrained with a proper Stage 2, re-enable this:
    #   reliability_factor = min(max(reliability, 0.4), 1.0) if reliability is not None else 0.92
    reliability_factor = 1.0

    if pred_label == "Authentic":
        max_prob = max(cm_prob, sp_prob)

        # Core: how clean is this image? (1.0 when both signals are 0)
        base = max(0.0, 1.0 - max_prob)

        # Clarity factor: how far from the ambiguous zone (0.3-0.5)?
        # max_prob < 0.15 → very clear (1.0)
        # max_prob ~ 0.3  → moderate (0.75)
        # max_prob ~ 0.5  → uncertain (0.5)
        if max_prob < 0.15:
            clarity = 1.0
        elif max_prob < 0.35:
            clarity = 1.0 - (max_prob - 0.15) * 1.25  # 1.0 → 0.75
        else:
            clarity = max(0.4, 0.75 - (max_prob - 0.35) * 1.5)

        return float(min(1.0, max(0.0, base * clarity * reliability_factor)))

    else:
        # Tampered prediction
        if pred_label == "CopyMove":
            prob = cm_prob
            threshold = thresholds["cm_threshold"]
        else:
            prob = sp_prob
            threshold = thresholds["sp_threshold"]

        # How far above threshold? (higher = more certain)
        margin = max(prob - threshold, 0.0)
        margin_factor = min(margin / max(1.0 - threshold, 0.1), 1.0) * 0.35 + 0.65

        # How separated are the two heads? (higher = more certain which type)
        separation = abs(cm_prob - sp_prob)
        separation_factor = min(separation / 0.25, 1.0) * 0.25 + 0.75

        return float(min(1.0, max(0.0, prob * margin_factor * separation_factor * reliability_factor)))

