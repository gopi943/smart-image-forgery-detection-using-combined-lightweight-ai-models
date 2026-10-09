"""
Evaluation Metrics for Forensic Detection.

AUROC, F1, confusion matrix, reliability diagram, per-dataset breakdown.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_probs=None, num_classes=3):
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground-truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N, C) for AUROC
        num_classes: Number of classes

    Returns:
        Dictionary of metrics.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Authentic", "CopyMove", "Splicing"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            results["auroc_macro"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["auroc_macro"] = None

    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_reliability_diagram(y_true, y_probs, n_bins=10, save_path=None):
    """
    Plot reliability diagram for calibration assessment.

    Compares predicted confidence vs. actual accuracy per bin.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Predicted Confidence")
    ax1.set_ylabel("Actual Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of confidences
    ax2.bar(
        bin_confs, bin_counts, width=1.0 / n_bins, alpha=0.7, edgecolor="black", color="orange"
    )
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def per_dataset_report(y_true, y_pred, sources, class_names=None):
    """Generate per-dataset performance breakdown."""
    if class_names is None:
        class_names = ["Authentic", "CopyMove", "Splicing"]

    report = {}
    unique_sources = sorted(set(sources))

    for src in unique_sources:
        mask = [s == src for s in sources]
        src_true = [y_true[i] for i, m in enumerate(mask) if m]
        src_pred = [y_pred[i] for i, m in enumerate(mask) if m]

        if len(src_true) == 0:
            continue

        report[src] = {
            "n_samples": len(src_true),
            "accuracy": accuracy_score(src_true, src_pred),
            "f1_macro": f1_score(src_true, src_pred, average="macro", zero_division=0),
        }

    return report
