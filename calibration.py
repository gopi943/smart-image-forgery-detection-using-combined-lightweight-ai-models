"""
Temperature Scaling — Post-hoc calibration for reliable confidence scores.

Learns a single temperature parameter on the validation set to transform
raw softmax outputs into calibrated probabilities that match empirical frequencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """
    Single-parameter temperature scaling for calibration.

    After training, optimize temperature T on the validation set so that
    softmax(logits / T) produces well-calibrated probabilities.
    """

    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated softmax probabilities."""
        return F.softmax(self.forward(logits), dim=-1)

    @torch.enable_grad()
    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Optimize temperature on validation set using NLL loss (LBFGS).

        Args:
            val_logits: Raw logits from model on validation set (N, C)
            val_labels: Ground-truth labels (N,)
            lr: LBFGS learning rate
            max_iter: Maximum LBFGS iterations

        Returns:
            Optimized temperature value.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()

    def compute_ece(
        self, probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            probs: Calibrated probabilities (N, C)
            labels: Ground-truth labels (N,)
            n_bins: Number of confidence bins

        Returns:
            ECE value (lower is better calibrated).
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].float().mean()
                bin_confidence = confidences[in_bin].mean()
                bin_weight = in_bin.float().sum() / len(labels)
                ece += bin_weight * (bin_accuracy - bin_confidence).abs()

        return ece.item()
