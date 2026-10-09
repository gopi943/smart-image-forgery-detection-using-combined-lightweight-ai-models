"""
Drift Detection Module.

Monitors prediction distributions, confidence, and latency for production drift.
Uses KL divergence and chi-squared tests to detect distribution shifts.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


class DriftDetector:
    """Detects input/output distribution drift in production."""

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        confidence_threshold: float = 0.1,
        latency_threshold_ms: float = 300.0,
    ):
        """
        Args:
            baseline_path: Path to baseline stats JSON (from validation set)
            confidence_threshold: Max KL divergence before alerting
            latency_threshold_ms: 95th percentile latency threshold
        """
        self.confidence_threshold = confidence_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.baseline = None

        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path) as f:
                self.baseline = json.load(f)

    @staticmethod
    def compute_prediction_distribution(predictions: List[str]) -> Dict[str, float]:
        """Compute frequency distribution of predictions."""
        counts = {}
        for p in predictions:
            counts[p] = counts.get(p, 0) + 1
        total = len(predictions)
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Compute KL divergence KL(p || q).
        p = observed distribution, q = baseline distribution.
        """
        all_keys = set(list(p.keys()) + list(q.keys()))
        eps = 1e-10
        kl = 0.0
        for k in all_keys:
            p_val = p.get(k, eps)
            q_val = q.get(k, eps)
            kl += p_val * np.log(p_val / q_val)
        return kl

    @staticmethod
    def chi_squared_test(observed: Dict[str, float], expected: Dict[str, float], n: int) -> float:
        """
        Chi-squared test statistic.
        Returns p-value (< 0.05 indicates significant drift).
        """
        from scipy import stats

        all_keys = sorted(set(list(observed.keys()) + list(expected.keys())))
        obs = np.array([observed.get(k, 0) * n for k in all_keys])
        exp = np.array([expected.get(k, 1e-10) * n for k in all_keys])

        # Avoid division by zero
        exp = np.maximum(exp, 1e-10)

        chi2_stat = np.sum((obs - exp) ** 2 / exp)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(all_keys) - 1)
        return p_value

    def check_drift(self, inference_logs: List[dict]) -> Dict[str, any]:
        """
        Analyze recent inference logs for drift signals.

        Args:
            inference_logs: List of logged inference entries

        Returns:
            Drift report with alerts.
        """
        if not inference_logs:
            return {"status": "no_data", "alerts": []}

        alerts = []
        report = {"status": "ok", "n_samples": len(inference_logs), "alerts": alerts}

        # Extract fields
        predictions = [log["prediction"] for log in inference_logs]
        confidences = [log["confidence"] for log in inference_logs]
        latencies = [log["inference_time_ms"] for log in inference_logs]

        # 1. Prediction distribution
        pred_dist = self.compute_prediction_distribution(predictions)
        report["prediction_distribution"] = pred_dist

        if self.baseline and "prediction_distribution" in self.baseline:
            kl = self.kl_divergence(pred_dist, self.baseline["prediction_distribution"])
            report["prediction_kl_divergence"] = round(kl, 4)
            if kl > self.confidence_threshold:
                alerts.append(
                    f"PREDICTION DRIFT: KL divergence {kl:.4f} > threshold {self.confidence_threshold}"
                )

        # 2. Confidence distribution
        avg_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        report["confidence_mean"] = round(avg_conf, 4)
        report["confidence_std"] = round(std_conf, 4)

        if self.baseline and "confidence_mean" in self.baseline:
            baseline_mean = self.baseline["confidence_mean"]
            diff = abs(avg_conf - baseline_mean)
            if diff > 0.1:
                alerts.append(
                    f"CONFIDENCE DRIFT: mean {avg_conf:.4f} vs baseline {baseline_mean:.4f}"
                )

        # 3. Latency
        p95_latency = np.percentile(latencies, 95)
        report["latency_p95_ms"] = round(p95_latency, 1)
        if p95_latency > self.latency_threshold_ms:
            alerts.append(
                f"LATENCY ALERT: P95 {p95_latency:.0f}ms > threshold {self.latency_threshold_ms:.0f}ms"
            )

        if alerts:
            report["status"] = "drift_detected"

        return report

    @staticmethod
    def save_baseline(
        predictions: List[str],
        confidences: List[float],
        save_path: str,
    ):
        """Save validation set statistics as drift baseline."""
        baseline = {
            "prediction_distribution": DriftDetector.compute_prediction_distribution(predictions),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "n_samples": len(predictions),
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(baseline, f, indent=2)
        return baseline
