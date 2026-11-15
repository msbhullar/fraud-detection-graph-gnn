"""Paired bootstrap: is the difference between two experiments on the SAME test rows real or noise?
Usage: python src/bootstrap_compare.py e1 e2"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RESULTS_DIR = Path("results")
NUMBER_OF_RESAMPLES = 1000
RANDOM_SEED = 42


def label_of(scores: pd.DataFrame):
    """PaySim score files use 'isFraud', IBM score files use 'label'."""
    return (scores["label"] if "label" in scores else scores["isFraud"]).values


def recall_at(labels, scores, threshold):
    return ((scores >= threshold) & (labels == 1)).sum() / max((labels == 1).sum(), 1)


def compare_pairs(baseline: str, candidate: str, pairs: list, normal_days_only: bool = False):
    global NUMBER_OF_RESAMPLES
    """pairs: [(baseline_model, candidate_model), ...] scored on the same test rows."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    baseline_metrics = json.loads((RESULTS_DIR / f"{baseline}_metrics.json").read_text())
    candidate_metrics = json.loads((RESULTS_DIR / f"{candidate}_metrics.json").read_text())
    summary = {"baseline": baseline, "candidate": candidate, "resamples": NUMBER_OF_RESAMPLES, "pairs": {}}
    for baseline_model, candidate_model in pairs:
        b = pd.read_parquet(RESULTS_DIR / f"{baseline}_test_scores_{baseline_model}.parquet")
        c = pd.read_parquet(RESULTS_DIR / f"{candidate}_test_scores_{candidate_model}.parquet")
        assert (label_of(b) == label_of(c)).all(), "test rows must match"
        labels = label_of(b)
        if normal_days_only:
            keep = ~b["is_tail"].values
            b, c, labels = b[keep].reset_index(drop=True), c[keep].reset_index(drop=True), labels[keep]
        bt = baseline_metrics["models"][baseline_model]["test_precision_floor"]["threshold"]
        ct = candidate_metrics["models"][candidate_model]["test_precision_floor"]["threshold"]
        pr, rc = [], []
        for _ in range(NUMBER_OF_RESAMPLES):
            rows = random_generator.integers(0, len(labels), len(labels)); y = labels[rows]
            pr.append(average_precision_score(y, c.fraud_score.values[rows]) - average_precision_score(y, b.fraud_score.values[rows]))
            rc.append(recall_at(y, c.fraud_score.values[rows], ct) - recall_at(y, b.fraud_score.values[rows], bt))
        pr, rc = np.array(pr), np.array(rc)
        key = f"{candidate}:{candidate_model} vs {baseline}:{baseline_model}"
        summary["pairs"][key] = {"pr_auc_difference_mean": float(pr.mean()), "pr_auc_difference_95ci": [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))],
                                 "share_candidate_better_pr_auc": float((pr > 0).mean()),
                                 "recall_difference_mean": float(rc.mean()), "recall_difference_95ci": [float(np.percentile(rc, 2.5)), float(np.percentile(rc, 97.5))],
                                 "share_candidate_better_recall": float((rc > 0).mean())}
        r = summary["pairs"][key]
        print(f"{key:45s} PR-AUC {r['pr_auc_difference_mean']:+.3f} [{r['pr_auc_difference_95ci'][0]:+.3f}, {r['pr_auc_difference_95ci'][1]:+.3f}] better in {r['share_candidate_better_pr_auc']:.0%} | "
              f"recall {r['recall_difference_mean']:+.3f} [{r['recall_difference_95ci'][0]:+.3f}, {r['recall_difference_95ci'][1]:+.3f}] better in {r['share_candidate_better_recall']:.0%}")
    out = RESULTS_DIR / f"{baseline}_vs_{candidate}_{'_'.join(c for _, c in pairs)}{'_normal_days' if normal_days_only else ''}_bootstrap.json"; out.write_text(json.dumps(summary, indent=2)); print("saved", out)


def main(baseline: str, candidate: str):
    baseline_metrics = json.loads((RESULTS_DIR / f"{baseline}_metrics.json").read_text())
    candidate_metrics = json.loads((RESULTS_DIR / f"{candidate}_metrics.json").read_text())
    random_generator = np.random.default_rng(RANDOM_SEED)
    summary = {"baseline": baseline, "candidate": candidate, "resamples": NUMBER_OF_RESAMPLES, "models": {}}
    for model_name in baseline_metrics["models"]:
        baseline_scores = pd.read_parquet(RESULTS_DIR / f"{baseline}_test_scores_{model_name}.parquet")
        candidate_scores = pd.read_parquet(RESULTS_DIR / f"{candidate}_test_scores_{model_name}.parquet")
        assert (baseline_scores.isFraud.values == candidate_scores.isFraud.values).all(), "test rows must match"
        labels = baseline_scores.isFraud.values
        baseline_threshold = baseline_metrics["models"][model_name]["test_precision_floor"]["threshold"]
        candidate_threshold = candidate_metrics["models"][model_name]["test_precision_floor"]["threshold"]
        pr_auc_differences, recall_differences = [], []
        for _ in range(NUMBER_OF_RESAMPLES):
            rows = random_generator.integers(0, len(labels), len(labels))  # same resampled rows for both models
            resampled_labels = labels[rows]
            pr_auc_differences.append(average_precision_score(resampled_labels, candidate_scores.fraud_score.values[rows])
                                      - average_precision_score(resampled_labels, baseline_scores.fraud_score.values[rows]))
            recall_differences.append(recall_at(resampled_labels, candidate_scores.fraud_score.values[rows], candidate_threshold)
                                      - recall_at(resampled_labels, baseline_scores.fraud_score.values[rows], baseline_threshold))
        pr_auc_differences, recall_differences = np.array(pr_auc_differences), np.array(recall_differences)
        summary["models"][model_name] = {
            "pr_auc_difference_mean": float(pr_auc_differences.mean()),
            "pr_auc_difference_95ci": [float(np.percentile(pr_auc_differences, 2.5)), float(np.percentile(pr_auc_differences, 97.5))],
            "share_of_resamples_candidate_worse_pr_auc": float((pr_auc_differences < 0).mean()),
            "recall_difference_mean": float(recall_differences.mean()),
            "recall_difference_95ci": [float(np.percentile(recall_differences, 2.5)), float(np.percentile(recall_differences, 97.5))],
            "share_of_resamples_candidate_worse_recall": float((recall_differences < 0).mean()),
        }
    output_path = RESULTS_DIR / f"{baseline}_vs_{candidate}_bootstrap.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"{candidate.upper()} minus {baseline.upper()} on the same test rows ({NUMBER_OF_RESAMPLES} resamples)")
    for model_name, r in summary["models"].items():
        print(f"{model_name:20s} PR-AUC diff {r['pr_auc_difference_mean']:+.3f}  95% range [{r['pr_auc_difference_95ci'][0]:+.3f}, {r['pr_auc_difference_95ci'][1]:+.3f}]  "
              f"{candidate.upper()} worse in {r['share_of_resamples_candidate_worse_pr_auc']:.0%}")
        print(f"{'':20s} recall diff {r['recall_difference_mean']:+.3f}  95% range [{r['recall_difference_95ci'][0]:+.3f}, {r['recall_difference_95ci'][1]:+.3f}]  "
              f"{candidate.upper()} worse in {r['share_of_resamples_candidate_worse_recall']:.0%}")
    print("saved", output_path)


if __name__ == "__main__":
    import builtins
    # python src/bootstrap_compare.py e1 e2                     (same model names in both)
    # python src/bootstrap_compare.py e1 e3 xgboost:graphsage    (explicit pairs)
    if len(sys.argv) > 3:
        normal_only = "--normal-days" in sys.argv
        for argument in sys.argv:
            if argument.startswith("--resamples="):
                NUMBER_OF_RESAMPLES = int(argument.split("=")[1])
        compare_pairs(sys.argv[1], sys.argv[2], [tuple(p.split(":")) for p in sys.argv[3:] if ":" in p], normal_only)
    else:
        main(sys.argv[1], sys.argv[2])
