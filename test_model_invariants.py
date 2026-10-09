"""Model invariants and component tests for the v2.1 path."""

import torch

from config import Config
from models.attention import CBAM
from models.forensic_net import ForensicNet
from models.localization import LiteUNetDecoder
from models.srm_filter import SRMFilter
from utils.grad_cam import GradCAM
from utils.tta import TTAPredictor


def make_cfg():
    cfg = Config()
    cfg.image_size = 96
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    return cfg


def test_srm_kernels_are_frozen_buffers():
    srm = SRMFilter()
    assert not srm.kernels.requires_grad
    assert "kernels" not in {name for name, _ in srm.named_parameters()}


def test_cbam_preserves_shape():
    cbam = CBAM(channels=128, reduction=8, spatial_kernel=7)
    x = torch.randn(2, 128, 16, 16)
    out = cbam(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_decoder_output_is_bounded():
    decoder = LiteUNetDecoder(in_channels=256, target_size=96)
    x = torch.randn(1, 256, 3, 3)
    out = decoder(x)
    assert out.shape == (1, 1, 96, 96)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_joint_dim_matches_head_input():
    cfg = make_cfg()
    model = ForensicNet(cfg)
    first_linear = model.cm_head[0]
    assert first_linear.in_features == cfg.fusion_dim


def test_batch_size_invariance():
    cfg = make_cfg()
    model = ForensicNet(cfg)
    model.eval()

    x_single = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    x_batch = x_single.repeat(4, 1, 1, 1)

    with torch.no_grad():
        single_outputs = model(x_single)
        batch_outputs = model(x_batch)

    assert torch.allclose(single_outputs[0], batch_outputs[0][0:1], atol=1e-5)
    assert torch.allclose(single_outputs[1], batch_outputs[1][0:1], atol=1e-5)


def test_tta_single_view_equals_forward():
    cfg = make_cfg()
    model = ForensicNet(cfg)
    model.eval()
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        direct_cm, direct_sp, direct_tamper, _, _ = model(x)
    tta_cm, tta_sp, tta_tamper = TTAPredictor(num_views=1).predict(model, x)

    assert torch.allclose(direct_cm, tta_cm, atol=1e-5)
    assert torch.allclose(direct_sp, tta_sp, atol=1e-5)
    assert torch.allclose(direct_tamper, tta_tamper, atol=1e-5)


def test_gradcam_output_shape():
    cfg = make_cfg()
    model = ForensicNet(cfg)
    model.eval()
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size, requires_grad=True)

    grad_cam = GradCAM(model, model.fusion.reduce)
    try:
        cam = grad_cam.generate(x, target_head="cm")
    finally:
        grad_cam.remove_hooks()

    assert cam.shape == (cfg.image_size, cfg.image_size)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
