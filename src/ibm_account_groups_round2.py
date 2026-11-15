"""Professor comment 2, updated 2026-10-29: recurring / new / cold-start test groups (and the finer sender/receiver-known split)
for the CURRENT models (tuned on Sol). Uses saved Sol test scores in results_sol/ and the thresholds frozen on validation;
no retraining, no new choices. Same group definitions as src/ibm_account_groups.py (unchanged).
Run on the Mac:  python src/ibm_account_groups_round2.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibm_account_groups import assign_groups
from src.ibm_data_prep import PROCESSED_DIR, TRAIN_END, TAIL_START

SOL_RESULTS_DIR = Path("results_sol")
OUTPUT_DIR = Path("results")
MODELS = [("E1 XGBoost (tuned)", "ibm_e1_tuned", "xgboost"), ("E2 XGBoost Neo4j (tuned)", "ibm_e2_tuned", "xgboost"),
          ("E2 + patterns (tuned)", "ibm_e2p_tuned", "xgboost"), ("E2r XGBoost (best)", "ibm_e2r_tuned", "xgboost"),
          ("E3 v2 graph removed", "ibm_e3v2_nograph", "graphsage"), ("E3 v2 GraphSAGE", "ibm_e3v2", "graphsage")]


def summarize(y, s, threshold):
    flagged = s >= threshold; caught = int((flagged & (y == 1)).sum())
    return {"transactions": int(len(y)), "laundering": int(y.sum()), "laundering_rate": float(y.mean()) if len(y) else np.nan,
            "pr_auc": average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan,
            "caught": caught, "false_alarms": int((flagged & (y == 0)).sum()), "recall": caught / y.sum() if y.sum() else np.nan}


def main():
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                   columns=["timestamp", "sender_account_id", "receiver_account_id", "Is Laundering"])
    test = assign_groups(transactions).reset_index(drop=True)
    seen = set(transactions.loc[transactions.timestamp < TRAIN_END, "sender_account_id"]) | set(transactions.loc[transactions.timestamp < TRAIN_END, "receiver_account_id"])
    del transactions
    sender_known = test.sender_account_id.isin(seen).values; receiver_known = test.receiver_account_id.isin(seen).values
    finer = np.select([sender_known & receiver_known, sender_known & ~receiver_known, ~sender_known & receiver_known],
                      ["both known", "receiver new", "sender new"], "both new")
    group_rows, finer_rows = [], []
    for name, experiment, model in MODELS:
        scores = pd.read_parquet(SOL_RESULTS_DIR / f"{experiment}_test_scores_{model}.parquet")
        assert (scores.label.values == test["Is Laundering"].values).all() and (scores.is_tail.values == test.is_tail.values).all()
        threshold = json.loads((SOL_RESULTS_DIR / f"{experiment}_metrics.json").read_text())["models"][model]["test_precision_floor"]["threshold"]
        y, s = scores.label.values, scores.fraud_score.values
        for period, period_mask in [("Sept 9-10", ~test.is_tail.values), ("Sept 11-18", test.is_tail.values)]:
            for group in ["recurring", "new", "cold-start"]:
                mask = period_mask & (test.group.values == group)
                group_rows.append({"model": name, "period": period, "group": group, **summarize(y[mask], s[mask], threshold)})
        for group in ["both known", "receiver new", "sender new", "both new"]:
            mask = ~test.is_tail.values & (finer == group)
            finer_rows.append({"model": name, "group": group, **summarize(y[mask], s[mask], threshold)})
    groups = pd.DataFrame(group_rows); finer_table = pd.DataFrame(finer_rows)
    groups.to_csv(OUTPUT_DIR / "ibm_test_by_account_group_round2.csv", index=False)
    finer_table.to_csv(OUTPUT_DIR / "ibm_test_by_sender_receiver_known_round2.csv", index=False)
    pd.set_option("display.width", 220)
    print(groups[groups.model == MODELS[0][0]][["period", "group", "transactions", "laundering"]].to_string(index=False))
    normal = groups[groups.period == "Sept 9-10"]
    for metric in ["pr_auc", "caught", "false_alarms"]:
        print(f"\nSept 9-10, {metric} by group:")
        print(normal.pivot(index="model", columns="group", values=metric)[["recurring", "new", "cold-start"]].reindex([m[0] for m in MODELS]).to_string(float_format=lambda v: f"{v:.3f}"))
    for metric in ["pr_auc", "caught"]:
        print(f"\nSept 9-10, sender/receiver seen in training, {metric}:")
        print(finer_table.pivot(index="model", columns="group", values=metric)[["both known", "receiver new", "sender new", "both new"]].reindex([m[0] for m in MODELS]).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nlaundering per finer group: {finer_table[finer_table.model == MODELS[0][0]].set_index('group').laundering.to_dict()}")
    print("saved results/ibm_test_by_account_group_round2.csv and results/ibm_test_by_sender_receiver_known_round2.csv")


if __name__ == "__main__":
    main()
