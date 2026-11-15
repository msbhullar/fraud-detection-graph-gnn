"""E4 typology case set (rule: docs/e4_typology_selection_rule.md). Read-only on the data; writes results/e4/cases_typology.csv.
150 flagged, pattern-matched true laundering cases + 50 false alarms from Sept 9-10 at the frozen precision-floor cut of ibm_e2r XGBoost,
excluding the original 50 E4 cases. pattern_name / pattern_attempt_number are grading-only columns.
Run from the project root:  python src/e4_select_cases_typology.py"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.e4_pattern_check import load_pattern_rows_with_attempt_ids
from src.ibm_data_prep import PROCESSED_DIR, TAIL_START, VALIDATION_END

RANDOM_SEED = 42
NUMBER_OF_TRUE_PATTERN_CASES = 150
NUMBER_OF_FALSE_ALARMS = 50
OUTPUT_PATH = Path("results/e4/cases_typology.csv")


def main():
    test_transactions = pd.read_parquet(
        PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
        columns=["transaction_id", "timestamp", "sender_account_id", "receiver_account_id",
                 "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"],
        filters=[("timestamp", ">=", VALIDATION_END)]).reset_index(drop=True)
    scores = pd.read_parquet("results_sol/ibm_e2r_tuned_test_scores_xgboost.parquet")
    assert len(test_transactions) == len(scores)
    assert (test_transactions["Is Laundering"].values == scores["label"].values).all(), "scores do not line up with test rows"
    threshold = json.loads(Path("results_sol/ibm_e2r_tuned_metrics.json").read_text())["models"]["xgboost"]["validation_precision_floor"]["threshold"]
    test_transactions["fraud_score"] = scores["fraud_score"].values
    test_transactions["amount_paid"] = test_transactions["Amount Paid"].round(2)

    pool = test_transactions[(test_transactions["timestamp"] < TAIL_START) & (test_transactions["fraud_score"] >= threshold)]
    original_ids = set(pd.read_csv("results/e4/cases.csv")["transaction_id"])
    pool = pool[~pool["transaction_id"].isin(original_ids)]

    patterns = load_pattern_rows_with_attempt_ids().drop_duplicates(["timestamp", "sender_account_id", "receiver_account_id", "amount_paid"])
    keys = ["timestamp", "sender_account_id", "receiver_account_id", "amount_paid"]
    pool = pool.merge(patterns, on=keys, how="left")
    pool = pool.drop_duplicates("transaction_id")
    true_pattern = pool[(pool["Is Laundering"] == 1) & pool["pattern_name"].notna()]
    true_other = pool[(pool["Is Laundering"] == 1) & pool["pattern_name"].isna()]
    false_alarms = pool[pool["Is Laundering"] == 0]
    print(f"threshold {threshold:.4f}; pool after excluding the original 50: {len(pool):,}")
    print(f"  true laundering in a pattern attempt: {len(true_pattern):,} | true laundering not in the pattern file: {len(true_other):,} | false alarms: {len(false_alarms):,}")
    print("  pattern mix of the available true cases:", true_pattern["pattern_name"].value_counts().to_dict())

    take_true = min(NUMBER_OF_TRUE_PATTERN_CASES, len(true_pattern))
    chosen = pd.concat([true_pattern.sample(take_true, random_state=RANDOM_SEED),
                        false_alarms.sample(NUMBER_OF_FALSE_ALARMS, random_state=RANDOM_SEED)]).sort_values("timestamp", kind="stable")
    cases = chosen.rename(columns={"Payment Currency": "payment_currency", "Payment Format": "payment_format",
                                   "Is Laundering": "true_label_for_grading_only"})
    cases.insert(0, "case_id", [f"TY-{number:03d}" for number in range(1, len(cases) + 1)])
    cases["payment_currency"] = cases["payment_currency"].astype(str)
    cases["payment_format"] = cases["payment_format"].astype(str)
    cases = cases.drop(columns=["Amount Paid"])
    cases.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved {OUTPUT_PATH}: shape {cases.shape}")
    print(cases["true_label_for_grading_only"].value_counts().rename({1: "true laundering (pattern member)", 0: "false alarm"}).to_string())
    print("pattern mix in the chosen set:", cases["pattern_name"].value_counts().to_dict())
    print("by day:", cases["timestamp"].dt.day.value_counts().sort_index().to_dict())
    print(cases[["case_id", "timestamp", "amount_paid", "payment_format", "fraud_score", "true_label_for_grading_only", "pattern_name"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
