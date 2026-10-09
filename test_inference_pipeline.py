"""End-to-end v2.1 inference pipeline test."""

from pathlib import Path

import torch
from PIL import Image

from config import Config
from data.transforms import get_val_transforms
from data.validation import ImageValidator
from models.forensic_net import ForensicNet
from utils.runtime_v2 import decision_policy


def test_full_pipeline():
    cfg = Config()
    cfg.image_size = 128
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = False

    model = ForensicNet(cfg)
    model.eval()

    validator = ImageValidator()
    transform = get_val_transforms(cfg.image_size, use_clip=cfg.use_clip)

    test_path = Path("tests") / "_runtime_tmp_pipeline.jpg"
    Image.new("RGB", (300, 300), color="green").save(test_path)

    try:
        result = validator.validate(str(test_path))
        assert result.valid

        x_forensic, x_clip = transform(result.image)
        assert x_forensic.shape == (3, cfg.image_size, cfg.image_size)
        assert x_clip is None

        with torch.no_grad():
            cm_logit, sp_logit, tamper_logit, mask, extras = model(
                x_forensic.unsqueeze(0), x_clip
            )

        assert cm_logit.shape == (1, 1)
        assert sp_logit.shape == (1, 1)
        assert tamper_logit is not None and tamper_logit.shape == (1, 1)
        assert mask is None
        assert "reliability" in extras

        cm_prob = torch.sigmoid(cm_logit).item()
        sp_prob = torch.sigmoid(sp_logit).item()
        pred_label, confidence = decision_policy(cm_prob, sp_prob)

        assert 0.0 <= cm_prob <= 1.0
        assert 0.0 <= sp_prob <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert pred_label in ["Authentic", "CopyMove", "Splicing"]
    finally:
        test_path.unlink(missing_ok=True)
