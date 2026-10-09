"""Tests for SRM Filter module."""
import torch
from models.srm_filter import SRMFilter


def test_output_shape():
    srm = SRMFilter()
    x = torch.randn(2, 3, 224, 224)
    out = srm(x)
    assert out.shape == (2, 3, 224, 224)


def test_different_sizes():
    srm = SRMFilter()
    for size in [128, 224, 256]:
        x = torch.randn(1, 3, size, size)
        out = srm(x)
        assert out.shape == (1, 3, size, size)


def test_kernels_are_buffers():
    srm = SRMFilter()
    param_names = [n for n, _ in srm.named_parameters()]
    assert not any("kernels" in n for n in param_names), "Kernels should be buffers"


def test_projection_is_trainable():
    srm = SRMFilter()
    trainable = [n for n, p in srm.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "Projection layer should be trainable"


def test_six_kernels():
    srm = SRMFilter()
    assert srm.kernels.shape[0] == 6, "Should have 6 SRM kernels"
    assert srm.kernels.shape[1] == 1
    assert srm.kernels.shape[2] == 5
    assert srm.kernels.shape[3] == 5
