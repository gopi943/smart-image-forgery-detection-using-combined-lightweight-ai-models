"""Tests for U-Net localization decoder."""
import torch
from models.localization import LiteUNetDecoder


def test_output_shape_224():
    decoder = LiteUNetDecoder(in_channels=256, target_size=224)
    x = torch.randn(2, 256, 7, 7)
    out = decoder(x)
    assert out.shape == (2, 1, 224, 224)


def test_output_range():
    """Output should be in [0, 1] due to sigmoid."""
    decoder = LiteUNetDecoder(in_channels=256, target_size=224)
    x = torch.randn(2, 256, 7, 7)
    out = decoder(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_different_target_size():
    decoder = LiteUNetDecoder(in_channels=256, target_size=256)
    x = torch.randn(1, 256, 8, 8)
    out = decoder(x)
    assert out.shape == (1, 1, 256, 256)
