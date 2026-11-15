"""E4 step 1: choose the 50 flagged test cases to explain (35 true laundering, 15 false alarms).
Cases come from Sept 9-10 (normal days) at the FROZEN precision-floor cut of the best model (ibm_e2r XGBoost),
whose threshold was fixed on validation. Random draw with a fixed seed. The true label is stored in a column
used only for grading; it must never be shown to the LLM.
Run from the project root: python src/e4_select_cases.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR, TAIL_START, VALIDATION_END

RANDOM_SEED = 42
NUMBER_OF_TRUE_POSITIVES = 35
NUMBER_OF_FALSE_ALARMS = 15
OUTPUT_PATH = Path("results/e4/cases.csv")


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
    flagged_normal_days = test_transactions[(test_transactions["timestamp"] < TAIL_START) & (test_transactions["fraud_score"] >= threshold)]
    true_positives = flagged_normal_days[flagged_normal_days["Is Laundering"] == 1]
    false_alarms = flagged_normal_days[flagged_normal_days["Is Laundering"] == 0]
    print(f"threshold {threshold:.4f}: flagged on Sept 9-10 = {len(flagged_normal_days):,} "
          f"(true laundering {len(true_positives):,}, false alarms {len(false_alarms):,})")

    chosen = pd.concat([true_positives.sample(NUMBER_OF_TRUE_POSITIVES, random_state=RANDOM_SEED),
                        false_alarms.sample(NUMBER_OF_FALSE_ALARMS, random_state=RANDOM_SEED)]).sort_values("timestamp", kind="stable")
    cases = chosen.rename(columns={"Amount Paid": "amount_paid", "Payment Currency": "payment_currency",
                                   "Payment Format": "payment_format", "Is Laundering": "true_label_for_grading_only"})
    cases.insert(0, "case_id", [f"E4-{number:03d}" for number in range(1, len(cases) + 1)])
    cases["payment_currency"] = cases["payment_currency"].astype(str)
    cases["payment_format"] = cases["payment_format"].astype(str)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUT_PATH, index=False)
    print(f"saved {OUTPUT_PATH}: shape {cases.shape}")
    print(cases["true_label_for_grading_only"].value_counts().rename({1: "true laundering", 0: "false alarm"}).to_string())
    print(cases.head(8).drop(columns=["true_label_for_grading_only"]).to_string(index=False))
    print("\nformats:", cases["payment_format"].value_counts().to_dict())
    print("score range:", round(cases["fraud_score"].min(), 3), "-", round(cases["fraud_score"].max(), 3))


if __name__ == "__main__":
    main()
