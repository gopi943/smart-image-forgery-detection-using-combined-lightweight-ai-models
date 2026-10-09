"""Data pipeline integrity tests for the v2.1 transform stack."""

import numpy as np
import torch
from PIL import Image

from data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OSNCurriculum,
    RandomJPEGCompression,
    denormalize,
    get_mask_transforms,
    get_train_transforms,
    get_val_transforms,
)


def test_val_transform_without_clip():
    transform = get_val_transforms(224, use_clip=False)
    image = Image.new("RGB", (300, 300), color=(128, 128, 128))
    x_forensic, x_clip = transform(image)

    assert x_forensic.shape == (3, 224, 224)
    assert x_forensic.dtype == torch.float32
    assert x_clip is None


def test_train_transform_with_clip():
    transform = get_train_transforms(224, use_clip=True)
    image = Image.new("RGB", (300, 300), color=(64, 128, 192))
    x_forensic, x_clip = transform(image)

    assert x_forensic.shape == (3, 224, 224)
    assert x_clip.shape == (3, 224, 224)
    assert not torch.isnan(x_forensic).any()
    assert not torch.isnan(x_clip).any()


def test_denormalize_roundtrip():
    from torchvision.transforms import Normalize, ToTensor

    image = Image.new("RGB", (50, 50), color=(100, 150, 200))
    original = ToTensor()(image)
    normalized = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(original.clone())
    recovered = denormalize(normalized.clone())

    assert torch.allclose(original, recovered, atol=1e-5)


def test_mask_transform_preserves_binary_values():
    transform = get_mask_transforms(224)
    mask_np = np.zeros((100, 100), dtype=np.uint8)
    mask_np[30:70, 30:70] = 255
    mask = Image.fromarray(mask_np, mode="L")

    torch.manual_seed(42)
    mask_tensor = transform(mask)

    assert mask_tensor.shape == (1, 224, 224)
    assert set(torch.unique(mask_tensor).tolist()).issubset({0.0, 1.0})


def test_jpeg_compression_identity_at_zero_probability():
    jpeg = RandomJPEGCompression(quality_range=(10, 100), p=0.0)
    image = Image.new("RGB", (100, 100), color="purple")
    result = jpeg(image)
    assert result is image


def test_osn_curriculum_returns_rgb_image():
    osn = OSNCurriculum(stage=2, p=1.0)
    image = Image.new("RGB", (120, 120), color="orange")
    result = osn(image)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == image.size
