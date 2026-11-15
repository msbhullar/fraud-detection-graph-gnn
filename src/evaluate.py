"""Shared evaluation: PR-AUC, validation threshold, metrics, plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def best_f1_threshold(true_labels, fraud_scores) -> float:
    """Pick the cutoff with the highest F1. Call this on VALIDATION only."""
    precision, recall, thresholds = precision_recall_curve(true_labels, fraud_scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1_values))])


def metrics_at_threshold(true_labels, fraud_scores, threshold: float) -> dict:
    predicted_labels = (fraud_scores >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    return {
        "fraud_prevalence": float(np.mean(true_labels)),
        "pr_auc": float(average_precision_score(true_labels, fraud_scores)),
        "pr_auc_random_baseline": float(np.mean(true_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels)),
        "f1": float(f1_score(true_labels, predicted_labels)),
        "threshold": float(threshold),
        "confusion_matrix": {"true_negative": int(true_negative), "false_positive": int(false_positive),
                             "false_negative": int(false_negative), "true_positive": int(true_positive)},
    }


def plot_pr_curves(curves: dict, prevalence: float, title: str, output_path: Path):
    """curves: {model_name: (true_labels, fraud_scores)}"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    for model_name, (true_labels, fraud_scores) in curves.items():
        precision, recall, _ = precision_recall_curve(true_labels, fraud_scores)
        axis.plot(recall, precision, label=f"{model_name} (PR-AUC {average_precision_score(true_labels, fraud_scores):.3f})")
    axis.axhline(prevalence, linestyle="--", color="gray", label=f"random guess ({prevalence:.2%})")
    axis.set_xlabel("Recall"); axis.set_ylabel("Precision"); axis.set_title(title); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output_path, dpi=150); plt.close(figure)


def plot_confusion_matrix(metrics: dict, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = metrics["confusion_matrix"]
    values = np.array([[matrix["true_negative"], matrix["false_positive"]],
                       [matrix["false_negative"], matrix["true_positive"]]])
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], ["predicted legit", "predicted fraud"]); axis.set_yticks([0, 1], ["actual legit", "actual fraud"])
    axis.set_title(title, fontsize=9); figure.tight_layout(); figure.savefig(output_path, dpi=150); plt.close(figure)


def save_json(data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


MINIMUM_ALARM_PRECISION = 0.50


def threshold_for_minimum_precision(true_labels, fraud_scores, minimum_precision: float = MINIMUM_ALARM_PRECISION) -> float:
    """Lowest cutoff (most fraud caught) where at least `minimum_precision` of alarms are real fraud.
    Call this on VALIDATION only."""
    precision, recall, thresholds = precision_recall_curve(true_labels, fraud_scores)
    allowed = np.where(precision[:-1] >= minimum_precision)[0]
    if len(allowed) == 0:
        return float("inf")  # rule not reachable on validation: the model raises no alarms under this rule
    best_index = allowed[np.argmax(recall[:-1][allowed])]
    return float(thresholds[best_index])
