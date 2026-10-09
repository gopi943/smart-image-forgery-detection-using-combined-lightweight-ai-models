"""Numerical stability tests for the v2.1 evidence heads."""

import torch
import pytest

from config import Config
from models.forensic_net import ForensicNet


@pytest.fixture(scope="module")
def model_and_cfg():
    cfg = Config()
    cfg.image_size = 96
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = False
    model = ForensicNet(cfg)
    model.eval()
    return model, cfg


def _assert_finite_tensors(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


def test_zero_input(model_and_cfg):
    model, cfg = model_and_cfg
    x = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        cm_logit, sp_logit, tamper_logit, mask, _ = model(x)
    _assert_finite_tensors(cm_logit, sp_logit, tamper_logit, mask)


def test_ones_input(model_and_cfg):
    model, cfg = model_and_cfg
    x = torch.ones(1, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        cm_logit, sp_logit, tamper_logit, mask, _ = model(x)
    _assert_finite_tensors(cm_logit, sp_logit, tamper_logit, mask)


def test_huge_noise_input(model_and_cfg):
    model, cfg = model_and_cfg
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size) * 1e4
    with torch.no_grad():
        cm_logit, sp_logit, tamper_logit, mask, _ = model(x)
    _assert_finite_tensors(cm_logit, sp_logit, tamper_logit, mask)


def test_sigmoid_probabilities_bounded(model_and_cfg):
    model, cfg = model_and_cfg
    x = torch.randn(4, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        cm_logit, sp_logit, _, _, _ = model(x)

    cm_prob = torch.sigmoid(cm_logit)
    sp_prob = torch.sigmoid(sp_logit)
    assert ((cm_prob >= 0.0) & (cm_prob <= 1.0)).all()
    assert ((sp_prob >= 0.0) & (sp_prob <= 1.0)).all()


def test_backward_gradients_remain_finite():
    cfg = Config()
    cfg.image_size = 96
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = True
    model = ForensicNet(cfg)
    model.train()

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, tamper_logit, mask, _ = model(x)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        cm_logit, torch.ones_like(cm_logit)
    )
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        sp_logit, torch.zeros_like(sp_logit)
    )
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        tamper_logit, torch.ones_like(tamper_logit)
    )
    loss = loss + torch.nn.functional.binary_cross_entropy(mask, torch.zeros_like(mask))
    loss.backward()

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert not torch.isnan(parameter.grad).any()
            assert not torch.isinf(parameter.grad).any()
