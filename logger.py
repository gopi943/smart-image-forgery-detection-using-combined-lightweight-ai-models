"""
Structured Inference Logger.

Logs every inference as structured JSON for monitoring and drift analysis.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional


class InferenceLogger:
    """Structured JSON logger for production inference monitoring."""

    def __init__(self, log_dir: str = "logs", model_version: str = "1.0.0"):
        self.log_dir = log_dir
        self.model_version = model_version
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(
            log_dir, f"inference_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    def log_inference(
        self,
        prediction: str,
        confidence: float,
        probabilities: List[float],
        inference_time_ms: float,
        input_resolution: tuple = None,
        input_format: str = None,
    ):
        """Log a single inference event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
            "inference_time_ms": round(inference_time_ms, 1),
            "input_resolution": list(input_resolution) if input_resolution else None,
            "input_format": input_format,
            "model_version": self.model_version,
        }

        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_logs(self, max_entries: int = 1000) -> List[dict]:
        """Read recent inference logs."""
        entries = []
        if os.path.exists(self._log_file):
            with open(self._log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
                        if len(entries) >= max_entries:
                            break
        return entries
