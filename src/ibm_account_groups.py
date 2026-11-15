"""Professor comment 2 (IBM AML): test results for recurring, new and cold-start accounts.
Uses saved test scores and frozen validation thresholds; no retraining."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR, TRAIN_END, VALIDATION_END, TAIL_START

RESULTS_DIR = Path("results")
MODELS = [("E1 XGBoost", "ibm_e1", "xgboost"), ("E2 XGBoost", "ibm_e2", "xgboost"),
          ("E3 network, no graph", "ibm_e3_nograph", "graphsage"), ("E3b GraphSAGE", "ibm_e3", "graphsage")]


def assign_groups(transactions: pd.DataFrame) -> pd.DataFrame:
    train = transactions.timestamp < TRAIN_END
    seen_in_training = set(transactions.loc[train, "sender_account_id"]) | set(transactions.loc[train, "receiver_account_id"])
    first_seen = pd.concat([transactions[["sender_account_id", "timestamp"]].set_axis(["account", "timestamp"], axis=1),
                            transactions[["receiver_account_id", "timestamp"]].set_axis(["account", "timestamp"], axis=1)]
                           ).groupby("account").timestamp.min()
    test = transactions[transactions.timestamp >= VALIDATION_END].copy()
    recurring = test.sender_account_id.isin(seen_in_training) | test.receiver_account_id.isin(seen_in_training)
    earlier = (test.sender_account_id.map(first_seen) < test.timestamp) | (test.receiver_account_id.map(first_seen) < test.timestamp)
    test["group"] = np.select([recurring, earlier], ["recurring", "new"], "cold-start")
    test["is_tail"] = test.timestamp >= TAIL_START
    return test


def main():
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                   columns=["timestamp", "sender_account_id", "receiver_account_id", "Is Laundering"])
    test = assign_groups(transactions)
    rows = []
    for name, experiment, model in MODELS:
        scores = pd.read_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model}.parquet")
        assert (scores.label.values == test["Is Laundering"].values).all() and (scores.is_tail.values == test.is_tail.values).all()
        threshold = json.loads((RESULTS_DIR / f"{experiment}_metrics.json").read_text())["models"][model]["test_precision_floor"]["threshold"]
        for period, period_mask in [("Sept 9-10", ~test.is_tail.values), ("Sept 11-18", test.is_tail.values)]:
            for group in ["recurring", "new", "cold-start"]:
                mask = period_mask & (test.group.values == group)
                y = scores.label.values[mask]; s = scores.fraud_score.values[mask]; flagged = s >= threshold
                caught = int((flagged & (y == 1)).sum()); false_alarms = int((flagged & (y == 0)).sum())
                rows.append({"model": name, "period": period, "group": group, "transactions": int(mask.sum()), "laundering": int(y.sum()),
                             "laundering_rate": y.mean() if mask.any() else np.nan,
                             "pr_auc": average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                             "caught": caught, "false_alarms": false_alarms,
                             "recall": caught / y.sum() if y.sum() else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "ibm_test_by_account_group.csv", index=False)
    return table


if __name__ == "__main__":
    table = main()
    pd.set_option("display.width", 220)
    sizes = table[table.model == "E1 XGBoost"][["period", "group", "transactions", "laundering", "laundering_rate"]]
    print(sizes.to_string(index=False, formatters={"laundering_rate": "{:.3%}".format}))
    normal = table[table.period == "Sept 9-10"]
    for metric in ["pr_auc", "recall", "caught", "false_alarms"]:
        print(f"\nSept 9-10, {metric}:")
        print(normal.pivot(index="model", columns="group", values=metric)[["recurring", "new", "cold-start"]]
              .reindex([m[0] for m in MODELS]).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsaved results/ibm_test_by_account_group.csv")


def side_by_side_groups():
    """Finer split: was the sender / the receiver seen in training (Sept 1-6)? Normal test days only."""
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                   columns=["timestamp", "sender_account_id", "receiver_account_id", "Is Laundering"])
    train = transactions.timestamp < TRAIN_END
    seen = set(transactions.loc[train, "sender_account_id"]) | set(transactions.loc[train, "receiver_account_id"])
    test = transactions[transactions.timestamp >= VALIDATION_END].reset_index(drop=True)
    sender_known = test.sender_account_id.isin(seen).values; receiver_known = test.receiver_account_id.isin(seen).values
    groups = np.select([sender_known & receiver_known, sender_known & ~receiver_known, ~sender_known & receiver_known],
                       ["both known", "receiver new", "sender new"], "both new")
    normal = (test.timestamp < TAIL_START).values
    rows = []
    for name, experiment, model in MODELS:
        scores = pd.read_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model}.parquet")
        threshold = json.loads((RESULTS_DIR / f"{experiment}_metrics.json").read_text())["models"][model]["test_precision_floor"]["threshold"]
        for group in ["both known", "receiver new", "sender new", "both new"]:
            mask = normal & (groups == group)
            y = scores.label.values[mask]; s = scores.fraud_score.values[mask]; flagged = s >= threshold
            caught = int((flagged & (y == 1)).sum())
            rows.append({"model": name, "group": group, "transactions": int(mask.sum()), "laundering": int(y.sum()),
                         "pr_auc": average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                         "caught": caught, "false_alarms": int((flagged & (y == 0)).sum()),
                         "recall": caught / y.sum() if y.sum() else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "ibm_test_by_sender_receiver_known.csv", index=False)
    return table
