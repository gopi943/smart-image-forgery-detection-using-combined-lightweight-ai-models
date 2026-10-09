"""Tests for backbone encoder modules."""
import torch
from models.backbones import SemanticEncoder, TextureEncoder


def test_semantic_encoder_shape():
    enc = SemanticEncoder(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape[0] == 2
    assert out.shape[1] == enc.out_channels  # EfficientNet-B0 last stage
    assert out.shape[2] == 7
    assert out.shape[3] == 7


def test_texture_encoder_shape():
    enc = TextureEncoder(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 576  # MobileNetV3-Small last stage channels
    assert out.shape[2] == 7
    assert out.shape[3] == 7


def test_out_channels_attribute():
    assert SemanticEncoder(pretrained=False).out_channels == 320
    assert TextureEncoder(pretrained=False).out_channels == 576
