"""Collect the honest and naive PaySim runs into one table (results/naive_leakage_summary.csv).
Reads only saved metrics files; trains nothing. Run: python src/naive_leakage_summary.py"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
RUNS = [  # (label, metrics file, model key or None, split, inputs)
    ("Honest E1 XGBoost", "e1_metrics.json", "xgboost", "temporal", "8 past-only tabular features"),
    ("Honest E2 XGBoost (past-only graph snapshots)", "e2_metrics.json", "xgboost", "temporal", "E1 + daily past-only Neo4j features"),
    ("Naive 1: + balance columns", "naive_balance_metrics.json", None, "temporal", "E1 + 4 balance columns + isFlaggedFraud"),
    ("Naive 2a: random split only", "naive_random_honest_metrics.json", None, "random", "E1 features only"),
    ("Naive 2b: balance columns + random split", "naive_random_balance_metrics.json", None, "random", "E1 + balance columns + isFlaggedFraud"),
    ("Naive 3a: full-graph features only", "naive_fullgraph_metrics.json", None, "temporal", "E1 + graph features from the FULL graph"),
    ("Naive 3b: everything (brief's naive run)", "naive_all_metrics.json", None, "random", "E1 + balance + isFlaggedFraud + full-graph features"),
]

rows = []
for label, file_name, model_key, split, inputs in RUNS:
    metrics = json.loads((RESULTS_DIR / file_name).read_text())
    result = metrics["models"][model_key] if model_key else metrics
    test, floor = result["test"], result["test_precision_floor"]
    rows.append({"run": label, "split": split, "inputs": inputs,
                 "test_fraud_prevalence": test["fraud_prevalence"], "pr_auc": test["pr_auc"],
                 "pr_auc_lift_over_random": test["pr_auc"] / test["pr_auc_random_baseline"],
                 "precision": test["precision"], "recall": test["recall"], "f1": test["f1"],
                 "floor_precision": floor["precision"], "floor_recall": floor["recall"]})
summary = pd.DataFrame(rows)
summary.to_csv(RESULTS_DIR / "naive_leakage_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 48)
print(summary.drop(columns=["inputs"]).round(4).to_string(index=False))
