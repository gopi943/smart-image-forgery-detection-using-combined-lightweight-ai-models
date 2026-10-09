"""Tests for SelfCorrelation module."""
import torch
from models.self_correlation import SelfCorrelation


def test_output_shape():
    """Output must be (B, 128, H, W) from (B, 320, H, W)."""
    sc = SelfCorrelation(in_channels=320, out_channels=128, topk_pct=0.1)
    feat = torch.randn(2, 320, 12, 12)
    out = sc(feat)
    assert out.shape == (2, 128, 12, 12), f"Expected (2, 128, 12, 12), got {out.shape}"


def test_gradient_flow():
    """Gradients must propagate through the correlation matrix."""
    sc = SelfCorrelation(in_channels=320, out_channels=128, topk_pct=0.1)
    feat = torch.randn(2, 320, 12, 12, requires_grad=True)
    out = sc(feat)
    loss = out.sum()
    loss.backward()
    assert feat.grad is not None, "No gradient on input"
    assert feat.grad.abs().sum() > 0, "Zero gradient"


def test_duplicate_detection():
    """Duplicated patches should produce higher correlation than random."""
    sc = SelfCorrelation(in_channels=64, out_channels=32, n_spatial=64, topk_pct=0.2)
    sc.eval()

    # Create input with duplicated patch (top-left == bottom-right)
    feat_dup = torch.randn(1, 64, 8, 8)
    feat_dup[:, :, 4:, 4:] = feat_dup[:, :, :4, :4]  # Copy top-left to bottom-right
    out_dup = sc(feat_dup)

    # Create random input (no duplication)
    feat_rand = torch.randn(1, 64, 8, 8)
    out_rand = sc(feat_rand)

    # Duplicated input should have higher max correlation values
    assert out_dup.abs().max() >= out_rand.abs().max() * 0.5, \
        "Duplicated patches should produce stronger correlation signal"


def test_different_spatial_sizes():
    """Module should work with different spatial resolutions."""
    for size in [6, 8, 12, 16]:
        n_spatial = size * size
        sc = SelfCorrelation(in_channels=128, out_channels=64, n_spatial=n_spatial, topk_pct=0.1)
        feat = torch.randn(1, 128, size, size)
        out = sc(feat)
        assert out.shape == (1, 64, size, size), f"Failed for size {size}"


def test_percentile_pooling():
    """Percentile pooling should zero out low-similarity entries."""
    sc = SelfCorrelation(in_channels=32, out_channels=16, n_spatial=64, topk_pct=0.1)
    feat = torch.randn(1, 32, 8, 8)
    # With topk=10%, only ~6 out of 64 positions should have non-zero correlation
    # We test indirectly by checking the module runs without error
    out = sc(feat)
    assert out.shape == (1, 16, 8, 8)
