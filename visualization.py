"""Visualization helpers for forensic evidence overlays."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


def _resize_map(arr: np.ndarray, image: np.ndarray) -> np.ndarray:
    if arr.shape[:2] != image.shape[:2]:
        arr = cv2.resize(arr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    return arr.astype(np.float32)


def _normalize_map(arr: np.ndarray, image: np.ndarray, blur_sigma: float = 0.0) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = _resize_map(arr, image)
    arr = np.clip(arr, 0.0, None)

    if blur_sigma > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    hi = float(np.percentile(arr, 99.2))
    lo = float(np.percentile(arr, 12.0))
    if hi - lo > 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    elif arr.max() > 0:
        arr = arr / (arr.max() + 1e-8)

    return arr


def _clean_binary_mask(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(int(binary.shape[0] * binary.shape[1] * 0.002), 96)
    cleaned = np.zeros_like(binary)

    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 1

    if cleaned.sum() == 0 and num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned[labels == largest] = 1

    return cleaned


def _safe_grabcut_mask(
    image: np.ndarray,
    support: np.ndarray,
    prediction: str | None = None,
) -> np.ndarray:
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    if prediction == "CopyMove":
        mask[support < 0.14] = cv2.GC_BGD
        mask[support > 0.58] = cv2.GC_FGD
        mask[(support > 0.34) & (support <= 0.58)] = cv2.GC_PR_FGD
    else:
        mask[support < 0.10] = cv2.GC_BGD
        mask[support > 0.50] = cv2.GC_FGD
        mask[(support > 0.26) & (support <= 0.50)] = cv2.GC_PR_FGD

    if not np.any(mask == cv2.GC_FGD):
        fallback = support > (0.42 if prediction == "CopyMove" else 0.34)
        return fallback.astype(np.uint8)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
        binary = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)
        return binary
    except Exception:
        fallback = support > (0.42 if prediction == "CopyMove" else 0.34)
        return fallback.astype(np.uint8)
def _normalize_scalar(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    peak = float(arr.max())
    if peak <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, 0.0, 1.0)


def refine_evidence_region(
    image: np.ndarray,
    support: np.ndarray | None,
    prediction: str | None = None,
) -> np.ndarray | None:
    if support is None:
        return None

    support = _normalize_scalar(support)
    if float(np.percentile(support, 99.0)) < 0.16:
        return None

    binary = _safe_grabcut_mask(image, support, prediction=prediction)
    binary = _clean_binary_mask(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(np.float32)

    total_area = int(binary.shape[0] * binary.shape[1])
    max_regions = 2 if prediction == "CopyMove" else 3
    scored: list[tuple[float, np.ndarray]] = []

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < max(int(total_area * 0.001), 120):
            continue

        x, y, w, h = stats[idx, :4]
        component = (labels == idx)
        mean_support = float(support[component].mean())
        aspect = max(w / max(h, 1), h / max(w, 1))
        fill_ratio = area / max(w * h, 1)
        touches = sum(
            [
                x <= 1,
                y <= 1,
                x + w >= binary.shape[1] - 1,
                y + h >= binary.shape[0] - 1,
            ]
        )
        area_ratio = area / max(total_area, 1)
        score = mean_support * 0.72 + min(area_ratio / 0.12, 1.0) * 0.12 + fill_ratio * 0.12 - touches * 0.08
        if prediction == "CopyMove":
            score -= max(aspect - 4.0, 0.0) * 0.10
            score -= max(0.20 - fill_ratio, 0.0) * 0.20
        else:
            score -= max(aspect - 5.5, 0.0) * 0.08
            score -= max(0.18 - fill_ratio, 0.0) * 0.35
        scored.append((score, component.astype(np.uint8)))

    if not scored:
        return binary.astype(np.float32)

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    kept = np.zeros_like(binary)
    kept_count = 0
    for score, component in scored:
        if kept_count >= max_regions:
            break
        if score < max(0.18, best_score * 0.38):
            continue
        kept = np.maximum(kept, component)
        kept_count += 1

    if kept.sum() == 0:
        kept = binary

    kept = _clean_binary_mask(kept)
    return kept.astype(np.float32)


def _component_score(primary: np.ndarray, support: np.ndarray | None, component: np.ndarray, total_area: int) -> float:
    area = max(int(component.sum()), 1)
    primary_mean = float(primary[component > 0].mean()) if area else 0.0
    support_mean = float(support[component > 0].mean()) if support is not None and area else 0.0
    ys, xs = np.where(component > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    touches = sum(
        [
            y0 <= 1,
            x0 <= 1,
            y1 >= component.shape[0] - 2,
            x1 >= component.shape[1] - 2,
        ]
    )
    area_ratio = area / max(total_area, 1)
    area_bonus = min(area / max(total_area * 0.045, 1), 1.0) * 0.06
    large_penalty = max(area_ratio - 0.12, 0.0) * 0.42
    border_penalty = touches * 0.1
    return primary_mean * 0.7 + support_mean * 0.24 + area_bonus - large_penalty - border_penalty


def _extract_region_support(
    image: np.ndarray,
    primary: np.ndarray,
    support: np.ndarray | None = None,
    prediction: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    primary = _normalize_scalar(primary)
    support = _normalize_scalar(support) if support is not None else None
    total_area = int(primary.shape[0] * primary.shape[1])

    high_thresh = float(np.clip(np.percentile(primary, 97 if prediction == "CopyMove" else 96), 0.52, 0.88))
    mid_floor = max(high_thresh - (0.1 if prediction == "CopyMove" else 0.13), 0.34 if prediction == "CopyMove" else 0.32)
    mid_thresh = float(np.clip(np.percentile(primary, 92 if prediction == "CopyMove" else 89), mid_floor, max(high_thresh - 0.04, mid_floor)))

    high_binary = _clean_binary_mask((primary >= high_thresh).astype(np.uint8))
    mid_binary = _clean_binary_mask((primary >= mid_thresh).astype(np.uint8))
    guided_binary = _safe_grabcut_mask(
        image,
        np.maximum(primary, support * 0.82) if support is not None else primary,
        prediction=prediction,
    )
    guided_binary = _clean_binary_mask(guided_binary)

    if mid_binary.sum() == 0:
        mid_binary = high_binary.copy()
    if high_binary.sum() == 0:
        high_binary = mid_binary.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mid_binary, connectivity=8)
    scored: list[tuple[float, int, int, float, np.ndarray]] = []
    max_regions = 1 if prediction == "CopyMove" else 2
    min_area = max(int(total_area * 0.0008), 64)

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = (labels == idx).astype(np.uint8)
        ys, xs = np.where(component > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        width = max(x1 - x0 + 1, 1)
        height = max(y1 - y0 + 1, 1)
        box_area = max(width * height, 1)
        fill_ratio = area / box_area
        aspect = max(width / max(height, 1), height / max(width, 1))
        touches = sum(
            [
                y0 <= 1,
                x0 <= 1,
                y1 >= component.shape[0] - 2,
                x1 >= component.shape[1] - 2,
            ]
        )
        if prediction == "Splicing" and touches >= 2 and (aspect > 5.5 or fill_ratio < 0.16):
            continue
        area_ratio = area / max(total_area, 1)
        score = _component_score(primary, support, component, total_area)
        high_overlap = float((high_binary * component).sum()) / max(area, 1)
        score += high_overlap * 0.18
        if prediction == "Splicing":
            score -= max(aspect - 3.6, 0.0) * 0.06
            score -= max(0.22 - fill_ratio, 0.0) * 0.5
        else:
            score -= max(aspect - 6.0, 0.0) * 0.02
            score -= max(0.12 - fill_ratio, 0.0) * 0.18
        scored.append((score, area, touches, area_ratio, component))

    if not scored:
        fallback = high_binary if high_binary.sum() > 0 else mid_binary
        shell = fallback.astype(np.uint8)
        core = _clean_binary_mask(cv2.erode(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))
        return core, shell

    if prediction == "CopyMove":
        interior_scored = [item for item in scored if item[2] == 0 and item[3] <= 0.14]
        if interior_scored:
            scored = interior_scored

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = scored[0][0]
    shell = np.zeros_like(mid_binary)

    kept = 0
    for score, _, _, _, component in scored:
        if kept >= max_regions:
            break
        score_floor = 0.22 if prediction == "CopyMove" else 0.16
        ratio_floor = 0.52 if prediction == "CopyMove" else 0.34
        if score < max(score_floor, best_score * ratio_floor):
            continue
        shell = np.maximum(shell, component)
        kept += 1

    if shell.sum() == 0:
        shell = scored[0][4].copy()

    shell = _clean_binary_mask(shell)
    core = _clean_binary_mask((high_binary * shell).astype(np.uint8))
    if core.sum() == 0:
        core = _clean_binary_mask(cv2.erode(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))

    return core, shell


def _soft_region(binary: np.ndarray, sigma: float) -> np.ndarray:
    if binary.sum() == 0:
        return np.zeros(binary.shape, dtype=np.float32)
    soft = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return _normalize_scalar(soft)


def _build_evidence_state(
    cam: np.ndarray | None,
    mask: np.ndarray | None,
    image: np.ndarray,
    structure: np.ndarray | None = None,
    prediction: str | None = None,
) -> dict[str, np.ndarray]:
    cam_map = _normalize_map(cam, image, blur_sigma=8.0) if cam is not None else None
    mask_map = _normalize_map(mask, image, blur_sigma=3.2) if mask is not None else None
    structure_map = _normalize_map(structure, image, blur_sigma=5.0) if structure is not None else None
    has_mask = mask_map is not None and float(np.percentile(mask_map, 97)) >= 0.12

    if prediction == "CopyMove" and structure_map is not None:
        if has_mask:
            primary = np.maximum(mask_map * 0.82, structure_map * 0.94)
        else:
            primary = np.maximum(structure_map * 0.96, (cam_map if cam_map is not None else 0.0) * 0.22)

        support = structure_map * 0.72
        if cam_map is not None:
            support = np.maximum(support, cam_map * 0.34)

        core, shell = _extract_region_support(image, primary, support, prediction=prediction)
        shell_soft = _soft_region(
            cv2.dilate(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
            sigma=8.0,
        )
        core_soft = _soft_region(core, sigma=4.2)

        localized_cam = np.zeros_like(primary, dtype=np.float32)
        if cam_map is not None:
            localized_cam = cam_map * np.clip(0.1 + shell_soft * 0.92, 0.0, 1.0)

        localized_structure = structure_map * np.clip(0.28 + shell_soft * 0.94, 0.0, 1.0)
        evidence = primary * 0.72 + localized_structure * 0.2 + localized_cam * 0.08
        evidence = np.maximum(evidence, primary * 0.95)
        evidence *= np.clip(0.34 + shell_soft * 0.96, 0.0, 1.0)
        evidence = _normalize_scalar(cv2.GaussianBlur(evidence, (0, 0), sigmaX=2.0, sigmaY=2.0))
        mask_map = primary
    elif prediction == "Splicing" and structure_map is not None:
        primary = structure_map * 0.88
        if cam_map is not None:
            primary = np.maximum(primary, cam_map * 0.58)

        support = structure_map
        if cam_map is not None:
            support = np.maximum(support, cam_map * 0.34)

        core, shell = _extract_region_support(image, primary, support, prediction=prediction)
        shell_soft = _soft_region(
            cv2.dilate(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
            sigma=7.4,
        )
        core_soft = _soft_region(core, sigma=4.0)

        localized_cam = np.zeros_like(primary, dtype=np.float32)
        if cam_map is not None:
            localized_cam = cam_map * np.clip(0.2 + shell_soft * 0.96, 0.0, 1.0)

        localized_structure = structure_map * np.clip(0.3 + shell_soft * 0.9, 0.0, 1.0)
        evidence = primary * 0.56 + localized_structure * 0.28 + localized_cam * 0.16
        evidence = np.maximum(evidence, primary * 0.92)
        evidence *= np.clip(0.34 + shell_soft * 0.92, 0.0, 1.0)
        evidence = _normalize_scalar(cv2.GaussianBlur(evidence, (0, 0), sigmaX=1.8, sigmaY=1.8))
        mask_map = primary
    elif has_mask:
        core, shell = _extract_region_support(image, mask_map, cam_map, prediction=prediction)
        shell_soft = _soft_region(
            cv2.dilate(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
            sigma=8.0,
        )
        core_soft = _soft_region(core, sigma=4.2)

        localized_cam = np.zeros_like(mask_map, dtype=np.float32)
        if cam_map is not None:
            localized_cam = cam_map * np.clip(0.18 + shell_soft * 1.08, 0.0, 1.0)

        evidence = mask_map * 0.84 + localized_cam * 0.16
        evidence = np.maximum(evidence, mask_map * 0.94)
        evidence *= np.clip(0.36 + shell_soft * 0.92, 0.0, 1.0)
        evidence = _normalize_scalar(cv2.GaussianBlur(evidence, (0, 0), sigmaX=2.0, sigmaY=2.0))
    else:
        fallback = cam_map if cam_map is not None else (
            structure_map if structure_map is not None else np.zeros(image.shape[:2], dtype=np.float32)
        )
        core, shell = _extract_region_support(image, fallback, None, prediction=prediction)
        shell_soft = _soft_region(
            cv2.dilate(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))),
            sigma=9.0,
        )
        core_soft = _soft_region(core, sigma=4.6)
        evidence = fallback
        mask_map = np.zeros_like(evidence, dtype=np.float32) if mask_map is None else mask_map
        localized_cam = fallback

    shell_area = float(shell.sum())
    core_area = float(core.sum())
    shell_ratio = shell_area / max(float(shell.size), 1.0)
    if core_area > 0 and shell_area > 0 and (shell_area > core_area * 3.5 or shell_ratio > 0.18):
        shell = _clean_binary_mask(
            cv2.dilate(shell if core.sum() == 0 else core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        )
        shell_soft = _soft_region(
            cv2.dilate(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
            sigma=7.4,
        )
        if core.sum() == 0:
            core = _clean_binary_mask(cv2.erode(shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))
        core_soft = _soft_region(core, sigma=4.2)

    return {
        "cam": np.zeros(image.shape[:2], dtype=np.float32) if cam_map is None else cam_map,
        "mask": np.zeros(image.shape[:2], dtype=np.float32) if mask_map is None else mask_map,
        "structure": np.zeros(image.shape[:2], dtype=np.float32) if structure_map is None else structure_map,
        "localized_cam": localized_cam,
        "evidence": evidence,
        "core": core.astype(np.uint8),
        "shell": shell.astype(np.uint8),
        "core_soft": core_soft,
        "shell_soft": shell_soft,
    }


def _draw_contours(image: np.ndarray, source: np.ndarray, color, levels=(0.46, 0.64, 0.82), thickness=2) -> np.ndarray:
    result = image.copy()
    for level in levels:
        binary = (source >= level).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(result, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return result


def fuse_evidence_maps(
    cam: np.ndarray | None,
    mask: np.ndarray | None,
    image: np.ndarray,
    structure: np.ndarray | None = None,
    prediction: str | None = None,
) -> np.ndarray:
    return _build_evidence_state(cam, mask, image, structure=structure, prediction=prediction)["evidence"]


def generate_spotlight_cam(
    image: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray | None = None,
    structure: np.ndarray | None = None,
    prediction: str | None = None,
) -> np.ndarray:
    """Create a cleaner evidence heatmap that preserves the original frame."""
    state = _build_evidence_state(cam, mask, image, structure=structure, prediction=prediction)
    evidence = state["evidence"]
    shell_soft = state["shell_soft"]
    core_soft = state["core_soft"]
    heatmap = cv2.applyColorMap(np.uint8(evidence * 255), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = image.astype(np.float32)
    alpha = np.clip((evidence - 0.1) / 0.9, 0.0, 1.0) ** 1.4
    alpha *= np.clip(0.34 + shell_soft * 0.9, 0.0, 1.0)
    alpha = alpha[..., np.newaxis] * 0.58
    base *= (1.0 - alpha * 0.12)

    blended = base * (1.0 - alpha) + heatmap * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended = _draw_contours(blended, shell_soft, (113, 231, 255), levels=(0.3, 0.52), thickness=1)
    blended = _draw_contours(blended, core_soft, (246, 240, 220), levels=(0.5,), thickness=1)
    return blended


def draw_forensic_bounding_boxes(
    image: np.ndarray,
    cam: np.ndarray,
    label: str = "",
    mask: np.ndarray | None = None,
    structure: np.ndarray | None = None,
    prediction: str | None = None,
) -> np.ndarray:
    """
    Create a real localization overlay from the model mask, optionally refined by Grad-CAM.
    Keeps the suspicious region readable while darkening the surrounding frame.
    """
    state = _build_evidence_state(cam, mask, image, structure=structure, prediction=prediction)
    shell = state["shell"]
    core = state["core"]
    shell_soft = state["shell_soft"]
    core_soft = state["core_soft"]

    overlay = image.astype(np.float32)
    if prediction == "Splicing":
        shell_color = np.full_like(overlay, (255, 188, 96))
        core_color = np.full_like(overlay, (255, 232, 196))
        shell_line = (255, 190, 92)
        core_line = (255, 236, 205)
    else:
        shell_color = np.full_like(overlay, (94, 220, 255))
        core_color = np.full_like(overlay, (176, 243, 255))
        shell_line = (118, 233, 255)
        core_line = (246, 240, 220)

    dim_factor = (0.34 + (1.0 - shell_soft[..., np.newaxis]) * 0.34).astype(np.float32)
    overlay = overlay * dim_factor

    shell_alpha = (shell_soft[..., np.newaxis] * 0.26).astype(np.float32)
    core_alpha = (core_soft[..., np.newaxis] * 0.46).astype(np.float32)

    overlay = overlay * (1.0 - shell_alpha) + shell_color * shell_alpha
    overlay = overlay * (1.0 - core_alpha) + core_color * core_alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    shell_contours, _ = cv2.findContours(shell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    core_contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if shell_contours:
        cv2.drawContours(overlay, shell_contours, -1, shell_line, 2, lineType=cv2.LINE_AA)
    if core_contours:
        cv2.drawContours(overlay, core_contours, -1, core_line, 2, lineType=cv2.LINE_AA)
    if core_contours:
        if label:
            largest = max(core_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            text_y = max(y - 10, 18)
            cv2.putText(
                overlay,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                core_line,
                1,
                cv2.LINE_AA,
            )

    return overlay

def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Fallback if used elsewhere"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != cam.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return np.uint8(alpha * heatmap + (1.0 - alpha) * image)


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay predicted tamper mask on original image.

    Args:
        image: Original image (H, W, 3) uint8
        mask: Binary mask (H, W) in [0, 1]
        color: Overlay color for tampered regions
        alpha: Blend factor

    Returns:
        Overlaid image (H, W, 3) uint8
    """
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    overlay = image.copy()
    color_mask = np.zeros_like(image)
    color_mask[:] = color
    mask_binary = (mask > 0.5).astype(np.float32)

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_binary > 0,
            np.uint8(alpha * color_mask[:, :, c] + (1 - alpha) * image[:, :, c]),
            image[:, :, c],
        )

    return overlay


def save_forensic_report(
    image: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray,
    prediction: str,
    confidence: float,
    probabilities: dict,
    save_path: str,
    inference_time_ms: float = None,
):
    """
    Generate and save composite forensic analysis figure.

    Shows: Original | Grad-CAM | Tamper Mask | Prediction details.
    """
    has_mask = mask is not None

    ncols = 3 if has_mask else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM overlay
    cam_overlay = overlay_heatmap(image, cam)
    axes[1].imshow(cam_overlay)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Mask overlay (if available)
    if has_mask:
        mask_overlay = overlay_mask(image, mask)
        axes[2].imshow(mask_overlay)
        axes[2].set_title("Tamper Mask", fontsize=14, fontweight="bold")
        axes[2].axis("off")

    # Prediction text
    color_map = {"Authentic": "green", "CopyMove": "red", "Splicing": "orange"}
    pred_color = color_map.get(prediction, "blue")

    prob_text = "\n".join(
        [f"  {k}: {v:.1%}" for k, v in probabilities.items()]
    )
    info_text = (
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Probabilities:\n{prob_text}"
    )
    if inference_time_ms:
        info_text += f"\n\nInference: {inference_time_ms:.0f}ms"

    fig.text(
        0.5, 0.02, info_text,
        ha="center", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round", facecolor=pred_color, alpha=0.2),
        family="monospace",
    )

    fig.suptitle(
        f"Forensic Analysis — {prediction}",
        fontsize=16, fontweight="bold", color=pred_color, y=0.98,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
