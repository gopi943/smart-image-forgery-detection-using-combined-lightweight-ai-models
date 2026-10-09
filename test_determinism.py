"""Determinism and reproducibility tests for the v2.1 path."""

import random

import pytest
import torch
from PIL import Image

from config import Config
from data.transforms import get_train_transforms, get_val_transforms
from models.forensic_net import ForensicNet
from utils.tta import TTAPredictor


@pytest.fixture(scope="module")
def deterministic_model():
    cfg = Config()
    cfg.image_size = 96
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    model = ForensicNet(cfg)
    model.eval()
    return model, cfg


def test_eval_mode_deterministic(deterministic_model):
    model, cfg = deterministic_model
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)

    results = []
    for _ in range(5):
        with torch.no_grad():
            outputs = model(x)
        results.append([tensor.clone() if tensor is not None else None for tensor in outputs[:4]])

    for idx in range(1, len(results)):
        for tensor_a, tensor_b in zip(results[0], results[idx]):
                if tensor_a is None:
                    assert tensor_b is None
                else:
                    assert torch.allclose(tensor_a, tensor_b, atol=1e-7, rtol=0.0)


def test_val_transform_deterministic_without_clip():
    transform = get_val_transforms(224, use_clip=False)
    image = Image.new("RGB", (300, 300), color="red")

    first_forensic, first_clip = transform(image)
    second_forensic, second_clip = transform(image)

    assert first_clip is None and second_clip is None
    assert torch.equal(first_forensic, second_forensic)


def test_val_transform_deterministic_with_clip():
    transform = get_val_transforms(224, use_clip=True)
    image = Image.new("RGB", (300, 300), color="blue")

    first_forensic, first_clip = transform(image)
    second_forensic, second_clip = transform(image)

    assert torch.equal(first_forensic, second_forensic)
    assert torch.equal(first_clip, second_clip)


def test_seeded_train_transform_reproducible():
    transform = get_train_transforms(224, use_clip=False)
    image = Image.new("RGB", (300, 300), color="green")

    outputs = []
    for _ in range(3):
        torch.manual_seed(42)
        random.seed(42)
        outputs.append(transform(image)[0].clone())

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_seeded_dual_train_transform_reproducible():
    transform = get_train_transforms(224, use_clip=True)
    image = Image.new("RGB", (300, 300), color="yellow")

    forensic_outputs = []
    clip_outputs = []
    for _ in range(3):
        torch.manual_seed(42)
        random.seed(42)
        x_forensic, x_clip = transform(image)
        forensic_outputs.append(x_forensic.clone())
        clip_outputs.append(x_clip.clone())

    assert torch.equal(forensic_outputs[0], forensic_outputs[1])
    assert torch.equal(clip_outputs[0], clip_outputs[1])


def test_tta_deterministic(deterministic_model):
    model, cfg = deterministic_model
    tta = TTAPredictor(num_views=5)
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    results = []
    for _ in range(3):
        results.append(tta.predict(model, x))

    for idx in range(1, len(results)):
        for tensor_a, tensor_b in zip(results[0], results[idx]):
                if tensor_a is None:
                    assert tensor_b is None
                else:
                    assert torch.allclose(tensor_a, tensor_b, atol=1e-7, rtol=0.0)


def test_different_inputs_yield_different_outputs(deterministic_model):
    model, cfg = deterministic_model
    x_zero = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
    x_one = torch.ones(1, 3, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        zero_outputs = model(x_zero)
        one_outputs = model(x_one)

    assert not torch.equal(zero_outputs[0], one_outputs[0]) or not torch.equal(
        zero_outputs[1], one_outputs[1]
    )
