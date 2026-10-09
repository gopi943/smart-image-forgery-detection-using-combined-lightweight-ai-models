"""
Inference Logger — Structured JSON logging for every analysis.

Enables drift detection, latency monitoring, and usage analytics.
"""

import json
import os
import time
from datetime import datetime, timezone


class InferenceLogger:
    """Append-only JSON logger for inference events."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(
            log_dir, f"inference_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    def log_inference(
        self,
        analysis_id: str = None,
        prediction: str = None,
        confidence: float = None,
        confidence_tier: str = None,
        probabilities: list = None,
        inference_time_ms: float = None,
        input_resolution: tuple = None,
        input_format: str = None,
        input_size_bytes: int = None,
        sha256: str = None,
        tta_enabled: bool = False,
        calibrated: bool = False,
        model_version: str = "1.0.0",
        error: str = None,
    ):
        """Log a single inference event as a JSON line."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_id": analysis_id,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_tier": confidence_tier,
            "inference_time_ms": inference_time_ms,
            "input_resolution": list(input_resolution) if input_resolution else None,
            "input_format": input_format,
            "input_size_bytes": input_size_bytes,
            "sha256": sha256,
            "tta_enabled": tta_enabled,
            "calibrated": calibrated,
            "model_version": model_version,
            "error": error,
        }

        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Logging should never crash the server

    def get_recent_logs(self, n: int = 100) -> list:
        """Read the last N log entries."""
        if not os.path.exists(self._log_file):
            return []
        try:
            with open(self._log_file) as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-n:]]
        except Exception:
            return []
