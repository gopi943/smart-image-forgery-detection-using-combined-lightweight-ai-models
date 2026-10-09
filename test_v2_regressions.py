"""Focused regression tests for the v2.1 training/inference pipeline."""

from pathlib import Path

import numpy as np

from data.dataset_v2 import ForensicDatasetV2
from data.prepare_v2 import collect_casia
from scripts.calibrate_v2 import decision_policy
from train_v2 import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_auth_fpr_is_image_level():
    metrics = compute_metrics(
        all_cm_logits=[10.0, -10.0],
        all_sp_logits=[10.0, -10.0],
        all_cm_targets=[0.0, 1.0],
        all_sp_targets=[0.0, 0.0],
    )

    assert metrics["auth_fpr"] == 1.0


def test_synthetic_copy_move_returns_resized_mask():
    image_path = next((REPO_ROOT / "datasets" / "casia_v2" / "authentic" / "Au").glob("*.jpg"))

    dataset = ForensicDatasetV2(
        metadata=[{"path": str(image_path), "label": "authentic", "source": "casia_v2"}],
        image_size=64,
        use_clip=False,
        is_train=True,
        synthetic_cm_prob=1.0,
    )

    sample = dataset[0]
    assert sample["cm_target"].item() == 1.0
    assert sample["mask"].shape == (1, 64, 64)
    assert sample["mask"].sum().item() > 0


def test_collect_casia_parses_m_and_s_and_skips_masks():
    entries = collect_casia(REPO_ROOT / "datasets" / "casia_v2")
    labels = [entry["label"] for entry in entries]
    tampered_with_masks = [
        entry for entry in entries
        if entry["label"] in {"copymove", "splicing"} and entry["mask_path"]
    ]

    assert "authentic" in labels
    assert "copymove" in labels
    assert "splicing" in labels
    assert tampered_with_masks


def test_decision_policy_respects_margin():
    cm_prob = np.array([0.65, 0.75, 0.20])
    sp_prob = np.array([0.63, 0.20, 0.72])

    pred = decision_policy(
        cm_prob=cm_prob,
        sp_prob=sp_prob,
        cm_threshold=0.6,
        sp_threshold=0.6,
        cm_delta=0.10,
        sp_delta=0.10,
    )

    assert pred.tolist() == [0, 1, 2]
