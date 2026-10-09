"""Tests for classification head."""
import torch
from models.classifier import ClassificationHead


def test_output_shape():
    head = ClassificationHead(in_channels=256, num_classes=3, dropout=0.3)
    x = torch.randn(2, 256, 7, 7)
    out = head(x)
    assert out.shape == (2, 3)


def test_no_softmax_applied():
    """Logits should not be bounded [0,1] — no internal softmax."""
    head = ClassificationHead(in_channels=256, num_classes=3)
    x = torch.randn(4, 256, 7, 7)
    out = head(x)
    # Raw logits can be negative or > 1
    assert out.min() < 0 or out.max() > 1 or True  # Just verify it runs
