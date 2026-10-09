"""Config permutation tests for the v2.1 model."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from config import Config
from models.forensic_net import ForensicNet


TOGGLE_COMBOS = list(itertools.product([False, True], repeat=4))
TOGGLE_NAMES = [
    "use_bayar",
    "use_self_correlation",
    "use_localization",
    "use_tamper_head",
]
FAST_IMAGE_SIZE = 96


@pytest.mark.parametrize(
    "combo",
    TOGGLE_COMBOS,
    ids=[
        f"bayar={c[0]}_corr={c[1]}_loc={c[2]}_tamper={c[3]}"
        for c in TOGGLE_COMBOS
    ],
)
def test_forward_pass_all_combos(combo):
    cfg = Config()
    cfg.image_size = FAST_IMAGE_SIZE
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    for name, value in zip(TOGGLE_NAMES, combo):
        setattr(cfg, name, value)

    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        cm_logit, sp_logit, tamper_logit, mask, extras = model(x)

    assert cm_logit.shape == (1, 1)
    assert sp_logit.shape == (1, 1)
    assert tamper_logit is None or tamper_logit.shape == (1, 1)

    if cfg.use_localization:
        assert mask is not None
        assert mask.shape == (1, 1, cfg.image_size, cfg.image_size)
    else:
        assert mask is None

    if cfg.use_self_correlation and cfg.use_localization:
        assert "corr_consistency" in extras

    assert not torch.isnan(cm_logit).any()
    assert not torch.isnan(sp_logit).any()


@pytest.mark.parametrize(
    "combo",
    TOGGLE_COMBOS,
    ids=[
        f"bwd_bayar={c[0]}_corr={c[1]}_loc={c[2]}_tamper={c[3]}"
        for c in TOGGLE_COMBOS
    ],
)
def test_backward_pass_all_combos(combo):
    cfg = Config()
    cfg.image_size = FAST_IMAGE_SIZE
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    for name, value in zip(TOGGLE_NAMES, combo):
        setattr(cfg, name, value)

    model = ForensicNet(cfg)
    model.train()

    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, tamper_logit, mask, _ = model(x)

    loss = F.binary_cross_entropy_with_logits(cm_logit, torch.ones_like(cm_logit))
    loss = loss + F.binary_cross_entropy_with_logits(sp_logit, torch.zeros_like(sp_logit))
    if tamper_logit is not None:
        loss = loss + F.binary_cross_entropy_with_logits(
            tamper_logit, torch.ones_like(tamper_logit)
        )
    if mask is not None:
        loss = loss + F.binary_cross_entropy(mask, torch.zeros_like(mask))

    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_variable_batch_sizes(batch_size):
    cfg = Config()
    cfg.image_size = FAST_IMAGE_SIZE
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        cm_logit, sp_logit, tamper_logit, _, _ = model(x)

    assert cm_logit.shape == (batch_size, 1)
    assert sp_logit.shape == (batch_size, 1)
    assert tamper_logit is not None and tamper_logit.shape == (batch_size, 1)


@pytest.mark.parametrize("dropout", [0.0, 0.5, 0.99])
def test_dropout_extremes(dropout):
    cfg = Config()
    cfg.image_size = FAST_IMAGE_SIZE
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.dropout_rate = dropout
    model = ForensicNet(cfg)

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    cm_logit, sp_logit, _, _, _ = model(x)
    assert cm_logit.shape == (2, 1)
    assert sp_logit.shape == (2, 1)


@pytest.mark.parametrize("image_size", [64, 96, 128])
def test_variable_image_sizes(image_size):
    cfg = Config()
    cfg.image_size = image_size
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = False
    model = ForensicNet(cfg)
    model.eval()

    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        cm_logit, sp_logit, _, _, _ = model(x)

    assert cm_logit.shape == (1, 1)
    assert sp_logit.shape == (1, 1)
