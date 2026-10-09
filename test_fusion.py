"""Tests for feature fusion module."""
import torch
from models.fusion import FeatureFusion


def test_fusion_output_shape():
    fusion = FeatureFusion(semantic_ch=320, texture_ch=576, fusion_dim=256)
    feat_s = torch.randn(2, 320, 7, 7)
    feat_t = torch.randn(2, 576, 7, 7)
    out = fusion(feat_s, feat_t)
    assert out.shape == (2, 256, 7, 7)


def test_fusion_gradient_flow():
    fusion = FeatureFusion(semantic_ch=320, texture_ch=576, fusion_dim=256)
    feat_s = torch.randn(2, 320, 7, 7, requires_grad=True)
    feat_t = torch.randn(2, 576, 7, 7, requires_grad=True)
    out = fusion(feat_s, feat_t)
    loss = out.sum()
    loss.backward()
    assert feat_s.grad is not None
    assert feat_t.grad is not None


def test_3stream_fusion_output_shape():
    """Test 3-stream fusion with ELA channel."""
    fusion = FeatureFusion(
        semantic_ch=320, texture_ch=576, ela_ch=320, fusion_dim=256
    )
    feat_s = torch.randn(2, 320, 7, 7)
    feat_t = torch.randn(2, 576, 7, 7)
    feat_e = torch.randn(2, 320, 7, 7)
    out = fusion(feat_s, feat_t, feat_e)
    assert out.shape == (2, 256, 7, 7)


def test_3stream_fusion_gradient_flow():
    """Verify gradients flow through all 3 stream projections."""
    fusion = FeatureFusion(
        semantic_ch=320, texture_ch=576, ela_ch=320, fusion_dim=256
    )
    feat_s = torch.randn(2, 320, 7, 7, requires_grad=True)
    feat_t = torch.randn(2, 576, 7, 7, requires_grad=True)
    feat_e = torch.randn(2, 320, 7, 7, requires_grad=True)
    out = fusion(feat_s, feat_t, feat_e)
    loss = out.sum()
    loss.backward()
    assert feat_s.grad is not None
    assert feat_t.grad is not None
    assert feat_e.grad is not None
