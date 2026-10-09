"""Tests for temperature scaling calibration."""
import torch
from models.calibration import TemperatureScaling


def test_forward_scaling():
    ts = TemperatureScaling(initial_temp=2.0)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    scaled = ts(logits)
    assert torch.allclose(scaled, logits / 2.0)


def test_calibrate_reduces_nll():
    """Calibration should reduce NLL on validation data."""
    torch.manual_seed(42)
    ts = TemperatureScaling(initial_temp=1.0)
    # Synthetic overconfident logits
    val_logits = torch.randn(100, 3) * 5  # Very large logits → overconfident
    val_labels = torch.randint(0, 3, (100,))

    # NLL before calibration
    nll_before = torch.nn.CrossEntropyLoss()(val_logits, val_labels).item()

    # Calibrate
    ts.calibrate(val_logits, val_labels)

    # NLL after calibration
    nll_after = torch.nn.CrossEntropyLoss()(ts(val_logits), val_labels).item()

    assert nll_after <= nll_before + 0.01  # Should not increase significantly


def test_ece_computation():
    ts = TemperatureScaling()
    probs = torch.tensor([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    labels = torch.tensor([0, 1, 2])
    ece = ts.compute_ece(probs, labels, n_bins=5)
    assert 0 <= ece <= 1


def test_temperature_positive():
    ts = TemperatureScaling(initial_temp=1.5)
    assert ts.temperature.item() > 0
