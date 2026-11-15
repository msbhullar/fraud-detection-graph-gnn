"""Professor comment 2: evaluate test transactions separately for recurring, new and cold-start accounts.
Uses the saved test scores and the frozen (validation-chosen) thresholds; no retraining."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import PROCESSED_DIR, TRAIN_LAST_STEP, VALIDATION_LAST_STEP

RESULTS_DIR = Path("results")
MODELS = [("E1 XGBoost", "e1", "xgboost"), ("E2 XGBoost", "e2", "xgboost"),
          ("E3 network, no graph", "e3_nograph", "graphsage"), ("E3b GraphSAGE", "e3", "graphsage")]


def assign_groups(transactions: pd.DataFrame) -> pd.Series:
    seen_in_training = set(transactions.loc[transactions.step <= TRAIN_LAST_STEP, "nameOrig"]) | \
                       set(transactions.loc[transactions.step <= TRAIN_LAST_STEP, "nameDest"])
    first_step = pd.concat([transactions[["nameOrig", "step"]].rename(columns={"nameOrig": "account"}),
                            transactions[["nameDest", "step"]].rename(columns={"nameDest": "account"})]).groupby("account").step.min()
    test = transactions[transactions.step > VALIDATION_LAST_STEP]
    recurring = test.nameOrig.isin(seen_in_training) | test.nameDest.isin(seen_in_training)
    has_earlier_activity = (test.nameOrig.map(first_step) < test.step) | (test.nameDest.map(first_step) < test.step)
    return pd.Series(np.select([recurring, has_earlier_activity], ["recurring", "new"], "cold-start"), index=test.index)


def main():
    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet", columns=["step", "nameOrig", "nameDest", "isFraud"])
    groups = assign_groups(transactions).values
    test_labels = transactions.loc[transactions.step > VALIDATION_LAST_STEP, "isFraud"].values
    rows = []
    for display_name, experiment, model in MODELS:
        scores = pd.read_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model}.parquet")
        assert (scores.isFraud.values == test_labels).all()
        threshold = json.loads((RESULTS_DIR / f"{experiment}_metrics.json").read_text())["models"][model]["test_precision_floor"]["threshold"]
        for group in ["recurring", "new", "cold-start", "all"]:
            mask = np.ones(len(groups), bool) if group == "all" else groups == group
            y, s = test_labels[mask], scores.fraud_score.values[mask]
            flagged = s >= threshold
            rows.append({"model": display_name, "group": group, "transactions": int(mask.sum()), "frauds": int(y.sum()),
                         "fraud_rate": y.mean(), "pr_auc": average_precision_score(y, s) if y.sum() else np.nan,
                         "recall": (flagged & (y == 1)).sum() / max(y.sum(), 1),
                         "precision": (flagged & (y == 1)).sum() / max(flagged.sum(), 1)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "test_by_account_group.csv", index=False)
    return table


if __name__ == "__main__":
    table = main()
    pd.set_option("display.width", 200)
    sizes = table[table.model == "E1 XGBoost"][["group", "transactions", "frauds", "fraud_rate"]]
    print("Test set by group:\n", sizes.to_string(index=False, formatters={"fraud_rate": "{:.2%}".format}))
    for metric in ["pr_auc", "recall", "precision"]:
        print(f"\n{metric} (precision>=50% cutoff for recall/precision):")
        print(table.pivot(index="model", columns="group", values=metric)[["recurring", "new", "cold-start", "all"]]
              .reindex([m[0] for m in MODELS]).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsaved results/test_by_account_group.csv")
