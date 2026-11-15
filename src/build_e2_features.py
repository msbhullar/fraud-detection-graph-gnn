"""Combine daily graph features, check them, and join to E1 features -> E2 feature files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import PROCESSED_DIR, TRAIN_LAST_STEP, VALIDATION_LAST_STEP
from src.features_tabular import add_tabular_features
DAILY_FEATURES_DIR = PROCESSED_DIR / "e2_daily_graph_features"  # written by src/graph_features.py

GRAPH_FEATURE_COLUMNS = [f"{role}_{name}" for role in ["sender", "receiver"] for name in
                         ["is_new_account", "graph_in_degree", "graph_out_degree", "graph_weighted_in_amount",
                          "graph_weighted_out_amount", "graph_pagerank", "graph_community_size"]]


def main():
    daily = pd.concat([pd.read_parquet(p) for p in sorted(DAILY_FEATURES_DIR.glob("day_*.parquet"))], ignore_index=True)
    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet")
    transactions["transaction_id"] = transactions.index
    print("daily files combined:", daily.shape, "| every transaction covered exactly once:",
          daily.transaction_id.is_unique and len(daily) == len(transactions))

    # Leakage test: for 5,000 random rows, GDS receiver in-degree must equal a pandas count of
    # payments to that receiver with step < first step of the row's day (never the same day or later).
    sample = daily.sample(5000, random_state=42).merge(transactions[["transaction_id", "step", "nameDest"]], on="transaction_id")
    sample["day_start_step"] = ((sample.step - 1) // 24) * 24 + 1
    receiver_steps = transactions.groupby("nameDest")["step"].apply(np.sort)
    sample["pandas_past_in_degree"] = [int(np.searchsorted(receiver_steps[name], cutoff, side="left"))
                                       for name, cutoff in zip(sample.nameDest, sample.day_start_step)]
    print("leakage test (5,000 rows): GDS in-degree == count of strictly earlier-day payments:",
          (sample.pandas_past_in_degree == sample.receiver_graph_in_degree).all())

    with_features = add_tabular_features(transactions)  # same E1 features, same order
    with_features["transaction_id"] = with_features.index
    with_features = with_features.merge(daily, on="transaction_id", how="left")
    print("missing graph values after join:", int(with_features[GRAPH_FEATURE_COLUMNS].isna().sum().sum()))

    masks = {"train": with_features.step <= TRAIN_LAST_STEP,
             "validation": (with_features.step > TRAIN_LAST_STEP) & (with_features.step <= VALIDATION_LAST_STEP),
             "test": with_features.step > VALIDATION_LAST_STEP}
    for name, mask in masks.items():
        part = with_features[mask].reset_index(drop=True)
        part.to_parquet(PROCESSED_DIR / f"e2_features_{name}.parquet", index=False)
        print(f"saved e2_features_{name}.parquet shape={part.shape} fraud prevalence={part.isFraud.mean():.2%}")
    return with_features, masks


if __name__ == "__main__":
    with_features, masks = main()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", "{:,.3f}".format)
    train = with_features[masks["train"]]
    print("\nTrain: graph feature averages, legit (0) vs fraud (1):")
    print(train.groupby("isFraud")[GRAPH_FEATURE_COLUMNS].mean().T)
    print("\nCorrelation of receiver graph features with E1 receiver_prior_transaction_count (train):")
    print(train[["receiver_graph_in_degree", "receiver_graph_pagerank", "receiver_graph_community_size"]]
          .corrwith(train["receiver_prior_transaction_count"]).round(3))
