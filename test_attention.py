"""Tests for CBAM attention module."""
import torch
from models.attention import CBAM, ChannelAttention, SpatialAttention


def test_channel_attention_shape():
    ca = ChannelAttention(256, reduction=16)
    x = torch.randn(2, 256, 7, 7)
    out = ca(x)
    assert out.shape == x.shape


def test_spatial_attention_shape():
    sa = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 256, 7, 7)
    out = sa(x)
    assert out.shape == x.shape


def test_cbam_shape_preserved():
    cbam = CBAM(512, reduction=16)
    x = torch.randn(2, 512, 7, 7)
    out = cbam(x)
    assert out.shape == x.shape


def test_cbam_different_channels():
    for ch in [64, 128, 256, 512]:
        cbam = CBAM(ch)
        x = torch.randn(1, ch, 14, 14)
        out = cbam(x)
        assert out.shape == x.shape
