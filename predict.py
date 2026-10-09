"""
Production v2.1 inference CLI.

Validates image inputs, runs the two-head forensic model, applies
checkpoint-aligned calibration and decision thresholds, and optionally
generates Grad-CAM forensic reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.transforms import get_val_transforms
from data.validation import ImageValidator
from utils.grad_cam import GradCAM
from utils.runtime_v2 import (
    authentic_probability,
    calibrate_prob,
    compute_confidence,
    decision_policy,
    load_calibration_artifacts,
    load_model_bundle,
)
from utils.tta import TTAPredictor
from utils.visualization import save_forensic_report


def _prepare_inputs(transformed, device: torch.device, channels_last: bool):
    x_forensic, x_clip = transformed
    x_forensic = x_forensic.unsqueeze(0).to(device)
    if x_clip is not None:
        x_clip = x_clip.unsqueeze(0).to(device)

    if channels_last:
        x_forensic = x_forensic.contiguous(memory_format=torch.channels_last)
        if x_clip is not None:
            x_clip = x_clip.contiguous(memory_format=torch.channels_last)

    return x_forensic, x_clip


def _iter_image_paths(root: str, supported_formats: List[str]) -> List[str]:
    if os.path.isfile(root):
        return [root]

    image_paths = []
    for current_root, _, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in supported_formats:
                image_paths.append(os.path.join(current_root, name))
    return image_paths


def predict_single(
    image_path: str,
    model,
    cfg,
    device: torch.device,
    validator: ImageValidator,
    transform,
    calibrators=None,
    thresholds=None,
    generate_cam: bool = True,
    save_report: bool = True,
    output_dir: str | None = None,
    use_tta: bool = False,
) -> Dict[str, object]:
    """Run a single-image v2.1 inference pass."""
    result = {
        "image_path": image_path,
        "prediction": None,
        "probabilities": {},
        "confidence": 0.0,
        "calibrated": calibrators is not None,
        "inference_time_ms": 0.0,
        "errors": [],
    }

    validation = validator.validate(image_path)
    if not validation.valid:
        result["errors"] = validation.errors
        return result

    t0 = time.perf_counter()
    x_forensic, x_clip = _prepare_inputs(
        transform(validation.image),
        device=device,
        channels_last=cfg.use_channels_last,
    )

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=cfg.use_amp and device.type == "cuda",
        ):
            if use_tta:
                cm_logit, sp_logit, tamper_logit = TTAPredictor(
                    num_views=cfg.tta_num_views
                ).predict(model, x_forensic, x_clip)
                mask_pred = None
            else:
                cm_logit, sp_logit, tamper_logit, mask_pred, _ = model(
                    x_forensic, x_clip
                )

    cm_raw = float(cm_logit.squeeze().item())
    sp_raw = float(sp_logit.squeeze().item())
    tamper_raw = (
        float(tamper_logit.squeeze().item()) if tamper_logit is not None else None
    )

    cm_prob = calibrate_prob(cm_raw, calibrators, "cm")
    sp_prob = calibrate_prob(sp_raw, calibrators, "sp")
    pred_label, raw_confidence = decision_policy(cm_prob, sp_prob, thresholds)
    # Compute meaningful confidence instead of using raw probability
    confidence = compute_confidence(pred_label, cm_prob, sp_prob, thresholds=thresholds)
    authentic_prob = authentic_probability(cm_prob, sp_prob)
    tamper_prob = (
        float(torch.sigmoid(torch.tensor(tamper_raw)).item())
        if tamper_raw is not None
        else max(cm_prob, sp_prob)
    )

    inference_time = (time.perf_counter() - t0) * 1000

    result["prediction"] = pred_label
    result["probabilities"] = {
        "Authentic": round(authentic_prob, 4),
        "CopyMove": round(cm_prob, 4),
        "Splicing": round(sp_prob, 4),
        "Tampered": round(tamper_prob, 4),
    }
    result["confidence"] = float(confidence)
    result["inference_time_ms"] = round(inference_time, 1)
    result["evidence"] = {
        "cm_probability": round(cm_prob, 4),
        "sp_probability": round(sp_prob, 4),
        "cm_raw_logit": round(cm_raw, 4),
        "sp_raw_logit": round(sp_raw, 4),
        "tamper_raw_logit": round(tamper_raw, 4) if tamper_raw is not None else None,
        "cm_threshold": thresholds.get("cm_threshold", 0.5) if thresholds else 0.5,
        "sp_threshold": thresholds.get("sp_threshold", 0.5) if thresholds else 0.5,
        "cm_delta": thresholds.get("cm_delta", 0.0) if thresholds else 0.0,
        "sp_delta": thresholds.get("sp_delta", 0.0) if thresholds else 0.0,
    }

    if generate_cam and save_report and pred_label != "Authentic":
        cam_generator = GradCAM(model)  # Auto-selects optimal layers
        try:
            cam = cam_generator.generate(
                x_forensic,
                x_clip=x_clip,
                target_head="cm" if pred_label == "CopyMove" else "sp",
                image_size=(cfg.image_size, cfg.image_size),
            )
        except Exception:
            cam = np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32)
        finally:
            cam_generator.remove_hooks()

        orig_np = np.array(validation.image.resize((cfg.image_size, cfg.image_size)))
        mask_np = mask_pred.squeeze().cpu().numpy() if mask_pred is not None else None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            report_path = os.path.join(output_dir, f"{stem}_forensic_report.png")
            save_forensic_report(
                orig_np,
                cam,
                mask_np,
                pred_label,
                float(confidence),
                result["probabilities"],
                report_path,
                inference_time,
            )
            result["report_path"] = report_path

    return result


def main():
    parser = argparse.ArgumentParser(description="ForensicLens v2.1 inference")
    parser.add_argument("--image", type=str, required=True, help="Image path or directory")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--output", type=str, default="predictions", help="Output directory"
    )
    parser.add_argument("--no-cam", action="store_true", help="Disable Grad-CAM reports")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    model, cfg, device, _, checkpoint_path = load_model_bundle(args.checkpoint)
    calibrators, thresholds = load_calibration_artifacts(checkpoint_path)

    validator = ImageValidator(
        supported_formats=cfg.supported_formats,
        max_resolution=cfg.max_resolution,
    )
    transform = get_val_transforms(cfg.image_size, use_clip=cfg.use_clip)

    image_files = _iter_image_paths(args.image, cfg.supported_formats)
    if not image_files:
        raise FileNotFoundError(f"No supported images found at: {args.image}")

    results = []
    for image_path in image_files:
        result = predict_single(
            image_path=image_path,
            model=model,
            cfg=cfg,
            device=device,
            validator=validator,
            transform=transform,
            calibrators=calibrators,
            thresholds=thresholds,
            generate_cam=not args.no_cam,
            save_report=not args.json,
            output_dir=args.output,
            use_tta=args.tta,
        )
        results.append(result)

        if not args.json:
            if result["errors"]:
                print(f"  x {os.path.basename(image_path)}: {result['errors']}")
            else:
                pred = result["prediction"]
                conf = result["confidence"]
                t_ms = result["inference_time_ms"]
                print(
                    f"  OK {os.path.basename(image_path)}: "
                    f"{pred} ({conf:.1%}) [{t_ms:.0f}ms]"
                )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if len(results) > 1:
        valid = [item for item in results if item["prediction"] is not None]
        print(f"\nProcessed {len(results)} images ({len(valid)} valid)")
        if valid:
            avg_time = np.mean([item["inference_time_ms"] for item in valid])
            print(f"Average inference: {avg_time:.0f}ms")


if __name__ == "__main__":
    main()
