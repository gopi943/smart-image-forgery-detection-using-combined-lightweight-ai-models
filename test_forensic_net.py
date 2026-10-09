"""Tests for the v2.1 ForensicNet interface."""

import torch

from config import Config
from models.forensic_net import ForensicNet


def make_cfg(localization: bool = False, self_correlation: bool = False):
    cfg = Config()
    cfg.image_size = 128
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = localization
    cfg.use_self_correlation = self_correlation
    return cfg


def test_forward_pass_without_localization():
    cfg = make_cfg(localization=False)
    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, tamper_logit, mask, extras = model(x)

    assert cm_logit.shape == (2, 1)
    assert sp_logit.shape == (2, 1)
    assert tamper_logit is not None and tamper_logit.shape == (2, 1)
    assert mask is None
    assert "reliability" in extras


def test_forward_pass_with_localization():
    cfg = make_cfg(localization=True)
    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, tamper_logit, mask, extras = model(x)

    assert cm_logit.shape == (2, 1)
    assert sp_logit.shape == (2, 1)
    assert tamper_logit is not None and tamper_logit.shape == (2, 1)
    assert mask is not None
    assert mask.shape == (2, 1, cfg.image_size, cfg.image_size)
    assert "reliability" in extras


def test_backward_pass_through_primary_heads():
    cfg = make_cfg(localization=True)
    model = ForensicNet(cfg)
    model.train()

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, tamper_logit, mask, _ = model(x)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        cm_logit, torch.tensor([[0.0], [1.0]])
    )
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        sp_logit, torch.tensor([[1.0], [0.0]])
    )
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        tamper_logit, torch.tensor([[1.0], [1.0]])
    )
    loss = loss + torch.nn.functional.binary_cross_entropy(
        mask, torch.zeros_like(mask)
    )
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_self_correlation_emits_consistency_signal():
    cfg = make_cfg(localization=True, self_correlation=True)
    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    _, _, _, _, extras = model(x)

    assert "corr_consistency" in extras
    assert extras["corr_consistency"].shape == (2,)
