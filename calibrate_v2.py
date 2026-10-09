"""
ForensicLens v2.1 — Calibration Script.

Fits isotonic regression and Platt scaling calibrators on the calibration split,
then picks whichever has lower ECE. Saves calibrators + optimized thresholds.

Usage:
    python scripts/calibrate_v2.py \\
        --checkpoint /root/checkpoints/best_384.pth \\
        --cal-data /root/dataset_v2/metadata_cal.csv \\
        --output /root/checkpoints

Outputs:
    calibrators.pkl  — fitted calibrator objects
    thresholds.json  — optimal thresholds + ECE metrics
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.forensic_net import ForensicNet
from data.dataset_v2 import ForensicDatasetV2
from torch.utils.data import DataLoader

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


def apply_model_config(cfg, config_dict):
    """Restore architecture-critical flags from a checkpoint."""
    for key in CHECKPOINT_CONFIG_KEYS:
        if key in config_dict:
            setattr(cfg, key, config_dict[key])


def decision_policy(cm_prob, sp_prob, cm_threshold, sp_threshold, cm_delta=0.0, sp_delta=0.0):
    """Apply the calibrated two-head policy with per-head margin gates."""
    cm_margin = cm_prob - sp_prob
    sp_margin = sp_prob - cm_prob

    cm_call = (cm_prob >= cm_threshold) & (cm_margin >= cm_delta)
    sp_call = (sp_prob >= sp_threshold) & (sp_margin >= sp_delta)

    pred = np.zeros_like(cm_prob, dtype=np.int64)
    pred[cm_call] = 1
    pred[sp_call] = 2

    conflict = cm_call & sp_call
    if np.any(conflict):
        pred[conflict] = np.where(cm_prob[conflict] >= sp_prob[conflict], 1, 2)

    return pred


def evaluate_policy(cm_prob, sp_prob, cm_targets, sp_targets, cm_threshold, sp_threshold,
                    cm_delta=0.0, sp_delta=0.0):
    """Evaluate the plan's business-weighted decision score."""
    pred = decision_policy(cm_prob, sp_prob, cm_threshold, sp_threshold, cm_delta, sp_delta)
    true = np.zeros_like(pred)
    true[cm_targets == 1] = 1
    true[sp_targets == 1] = 2

    auth_mask = true == 0
    cm_mask = true == 1
    sp_mask = true == 2

    auth_fpr = float((pred[auth_mask] != 0).mean()) if np.any(auth_mask) else 0.0
    cm_recall = float((pred[cm_mask] == 1).mean()) if np.any(cm_mask) else 0.0
    sp_recall = float((pred[sp_mask] == 2).mean()) if np.any(sp_mask) else 0.0

    return {
        "auth_fpr": auth_fpr,
        "cm_recall": cm_recall,
        "sp_recall": sp_recall,
        "score": 0.5 * (1.0 - auth_fpr) + 0.25 * cm_recall + 0.25 * sp_recall,
    }


def search_decision_policy(cm_prob, sp_prob, cm_targets, sp_targets):
    """Jointly search thresholds and margins using the plan's scoring rule."""
    threshold_grid = np.arange(0.2, 0.81, 0.02)
    delta_grid = np.arange(0.0, 0.26, 0.05)

    best = {
        "cm_threshold": 0.5,
        "sp_threshold": 0.5,
        "cm_delta": 0.0,
        "sp_delta": 0.0,
        "score": -np.inf,
        "auth_fpr": 1.0,
        "cm_recall": 0.0,
        "sp_recall": 0.0,
    }

    for cm_threshold in threshold_grid:
        for sp_threshold in threshold_grid:
            for cm_delta in delta_grid:
                for sp_delta in delta_grid:
                    metrics = evaluate_policy(
                        cm_prob,
                        sp_prob,
                        cm_targets,
                        sp_targets,
                        cm_threshold,
                        sp_threshold,
                        cm_delta,
                        sp_delta,
                    )
                    if metrics["score"] > best["score"]:
                        best = {
                            "cm_threshold": float(cm_threshold),
                            "sp_threshold": float(sp_threshold),
                            "cm_delta": float(cm_delta),
                            "sp_delta": float(sp_delta),
                            **metrics,
                        }

    return best


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE).

    Lower = better. ECE = 0 means perfectly calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += mask.sum() * abs(bin_conf - bin_acc)
    return ece / len(y_true)


def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        mce = max(mce, abs(bin_conf - bin_acc))
    return mce


def collect_predictions(model, cal_metadata, cfg, device):
    """Run inference on calibration split, collect raw logits."""
    dataset = ForensicDatasetV2(
        metadata=cal_metadata,
        image_size=cfg.image_size,
        use_clip=cfg.use_clip,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    all_cm_logits = []
    all_sp_logits = []
    all_cm_targets = []
    all_sp_targets = []
    amp_enabled = cfg.use_amp and device.type == "cuda"

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_forensic = batch["x_forensic"].to(device, non_blocking=True)
            x_clip = batch["x_clip"].to(device, non_blocking=True) if cfg.use_clip else None
            known_mask = batch["subtype_known"] > 0

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                cm_logit, sp_logit, _, _, _ = model(x_forensic, x_clip)

            if known_mask.any():
                keep = known_mask.nonzero(as_tuple=False).squeeze(1)
                all_cm_logits.extend(cm_logit.squeeze(1)[keep].cpu().tolist())
                all_sp_logits.extend(sp_logit.squeeze(1)[keep].cpu().tolist())
                all_cm_targets.extend(batch["cm_target"][keep].tolist())
                all_sp_targets.extend(batch["sp_target"][keep].tolist())

    return (
        np.array(all_cm_logits), np.array(all_sp_logits),
        np.array(all_cm_targets), np.array(all_sp_targets),
    )


def fit_calibrators(y_true, y_prob_raw):
    """Fit both isotonic and Platt, return whichever has lower ECE."""
    y_prob = 1 / (1 + np.exp(-y_prob_raw))  # sigmoid

    if len(np.unique(y_true)) < 2:
        return None, "identity", 1.0, y_prob

    # Isotonic regression
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    iso_probs = iso.predict(y_prob)
    iso_ece = compute_ece(y_true, iso_probs)

    # Platt scaling (logistic regression on logits)
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(y_prob_raw.reshape(-1, 1), y_true)
    platt_probs = platt.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
    platt_ece = compute_ece(y_true, platt_probs)

    print(f"    Isotonic ECE={iso_ece:.4f}  |  Platt ECE={platt_ece:.4f}")

    if iso_ece <= platt_ece:
        return iso, "isotonic", iso_ece, iso_probs
    else:
        return platt, "platt", platt_ece, platt_probs


def main():
    parser = argparse.ArgumentParser(description="ForensicLens v2.1 Calibration")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--cal-data", required=True, help="Calibration CSV")
    parser.add_argument("--output", default="checkpoints", help="Output directory")
    args = parser.parse_args()
    cal_data_path = Path(args.cal_data)
    if cal_data_path.is_dir():
        cal_data_path = cal_data_path / "metadata_cal.csv"

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    apply_model_config(cfg, ckpt.get("config", {}))
    if "resolution" in ckpt:
        cfg.image_size = int(ckpt["resolution"])

    # Load model
    model = ForensicNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # Load calibration metadata
    cal_meta = []
    with open(cal_data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cal_meta.append(row)
    print(f"Calibration images: {len(cal_meta)}")

    # Collect predictions
    print("\nCollecting predictions on calibration split...")
    cm_logits, sp_logits, cm_targets, sp_targets = collect_predictions(
        model, cal_meta, cfg, device
    )
    print(f"  CM positives: {cm_targets.sum():.0f}/{len(cm_targets)}")
    print(f"  SP positives: {sp_targets.sum():.0f}/{len(sp_targets)}")

    # Fit calibrators
    print("\nFitting CM calibrator...")
    cm_cal, cm_method, cm_ece, cm_cal_probs = fit_calibrators(cm_targets, cm_logits)

    print("\nFitting SP calibrator...")
    sp_cal, sp_method, sp_ece, sp_cal_probs = fit_calibrators(sp_targets, sp_logits)

    # Joint threshold + margin search
    print("\nSearching joint decision policy...")
    policy = search_decision_policy(cm_cal_probs, sp_cal_probs, cm_targets, sp_targets)
    print(
        "  Policy:"
        f" CM>={policy['cm_threshold']:.3f} Δ>={policy['cm_delta']:.3f}"
        f" | SP>={policy['sp_threshold']:.3f} Δ>={policy['sp_delta']:.3f}"
        f" | score={policy['score']:.4f}"
    )

    # Compute additional metrics
    cm_brier = brier_score_loss(cm_targets, cm_cal_probs)
    sp_brier = brier_score_loss(sp_targets, sp_cal_probs)
    cm_mce = compute_mce(cm_targets, cm_cal_probs)
    sp_mce = compute_mce(sp_targets, sp_cal_probs)

    # Save calibrators
    os.makedirs(args.output, exist_ok=True)

    cal_path = os.path.join(args.output, "calibrators.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump({
            "cm_calibrator": cm_cal,
            "sp_calibrator": sp_cal,
            "cm_method": cm_method,
            "sp_method": sp_method,
        }, f)
    print(f"\nCalibrators saved to: {cal_path}")

    # Save thresholds
    thresholds = {
        "cm_threshold": float(policy["cm_threshold"]),
        "sp_threshold": float(policy["sp_threshold"]),
        "cm_method": cm_method,
        "sp_method": sp_method,
        "cm_delta": float(policy["cm_delta"]),
        "sp_delta": float(policy["sp_delta"]),
        "decision_score": float(policy["score"]),
        "decision_auth_fpr": float(policy["auth_fpr"]),
        "decision_cm_recall": float(policy["cm_recall"]),
        "decision_sp_recall": float(policy["sp_recall"]),
        "cm_ece": float(cm_ece),
        "sp_ece": float(sp_ece),
        "cm_mce": float(cm_mce),
        "sp_mce": float(sp_mce),
        "cm_brier": float(cm_brier),
        "sp_brier": float(sp_brier),
        "cal_samples": len(cal_meta),
    }

    thresh_path = os.path.join(args.output, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to: {thresh_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE")
    print(f"  CM: {cm_method} | ECE={cm_ece:.4f} | MCE={cm_mce:.4f} | Brier={cm_brier:.4f}")
    print(f"  SP: {sp_method} | ECE={sp_ece:.4f} | MCE={sp_mce:.4f} | Brier={sp_brier:.4f}")
    print(
        "  Policy:"
        f" CM≥{policy['cm_threshold']:.3f} with Δ≥{policy['cm_delta']:.3f}"
        f" | SP≥{policy['sp_threshold']:.3f} with Δ≥{policy['sp_delta']:.3f}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
