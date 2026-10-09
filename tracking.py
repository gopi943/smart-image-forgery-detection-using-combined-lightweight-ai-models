"""
Experiment Tracking Wrapper.

Provides a unified interface for MLflow or Weights & Biases.
Falls back to local file logging if neither is available.
"""

import json
import os
from datetime import datetime


class ExperimentTracker:
    """Simple experiment tracking — logs metrics and hyperparams."""

    def __init__(self, experiment_name: str, log_dir: str = "logs", backend: str = "local"):
        """
        Args:
            experiment_name: Name of the experiment run
            log_dir: Directory for local logs
            backend: "local", "mlflow", or "wandb"
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.backend = backend
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history = []

        os.makedirs(log_dir, exist_ok=True)

        if backend == "mlflow":
            try:
                import mlflow

                mlflow.set_experiment(experiment_name)
                self._mlflow_run = mlflow.start_run(run_name=self.run_id)
                self._mlflow = mlflow
            except ImportError:
                print("[WARN] mlflow not installed, falling back to local logging")
                self.backend = "local"
        elif backend == "wandb":
            try:
                import wandb

                wandb.init(project=experiment_name, name=self.run_id)
                self._wandb = wandb
            except ImportError:
                print("[WARN] wandb not installed, falling back to local logging")
                self.backend = "local"

    def log_params(self, params: dict):
        """Log hyperparameters."""
        if self.backend == "mlflow":
            self._mlflow.log_params(params)
        elif self.backend == "wandb":
            self._wandb.config.update(params)

        # Always save locally
        path = os.path.join(self.log_dir, f"{self.run_id}_params.json")
        with open(path, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics for a training step/epoch."""
        entry = {"step": step, **metrics}
        self.metrics_history.append(entry)

        if self.backend == "mlflow":
            self._mlflow.log_metrics(metrics, step=step)
        elif self.backend == "wandb":
            self._wandb.log(metrics, step=step)

    def save_history(self):
        """Save full metrics history to JSON."""
        path = os.path.join(self.log_dir, f"{self.run_id}_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def finish(self):
        """End the tracking session."""
        self.save_history()
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend == "wandb":
            self._wandb.finish()
