"""Combine IBM daily graph features, test them for leakage, and join to IBM E1 features -> ibm_e2_features_*."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR

DAILY_DIR = PROCESSED_DIR / "ibm_e2_daily_graph_features"
IBM_GRAPH_FEATURE_COLUMNS = [f"{role}_{name}" for role in ["sender", "receiver"] for name in
                             ["is_new_account", "graph_in_degree", "graph_out_degree", "graph_weighted_in_amount",
                              "graph_weighted_out_amount", "graph_pagerank", "graph_community_size"]]


def main():
    daily = pd.concat([pd.read_parquet(p) for p in sorted(DAILY_DIR.glob("day_*.parquet"))], ignore_index=True)
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                   columns=["transaction_id", "timestamp", "receiver_account_id"])
    print("daily files:", daily.shape, "| every transaction exactly once:",
          daily.transaction_id.is_unique and len(daily) == len(transactions))

    # Leakage test: GDS receiver in-degree == number of payments the receiver got before the start of that day
    sample = daily.sample(5000, random_state=42).merge(transactions, on="transaction_id")
    day_start = sample.timestamp.dt.normalize()
    received_times = transactions.groupby("receiver_account_id").timestamp.apply(lambda s: np.sort(s.values))
    sample["pandas_in_degree"] = [int(np.searchsorted(received_times[a], np.datetime64(d), side="left"))
                                  for a, d in zip(sample.receiver_account_id, day_start)]
    print("leakage test (5,000 rows): GDS in-degree == count of strictly earlier-day payments:",
          (sample.pandas_in_degree == sample.receiver_graph_in_degree).all())
    del transactions

    for name in ["train", "validation", "test"]:
        e1 = pd.read_parquet(PROCESSED_DIR / f"ibm_e1_features_{name}.parquet")
        e2 = e1.merge(daily, on="transaction_id", how="left")
        assert e2[IBM_GRAPH_FEATURE_COLUMNS].notna().all().all() and len(e2) == len(e1)
        e2.to_parquet(PROCESSED_DIR / f"ibm_e2_features_{name}.parquet", index=False)
        print(f"saved ibm_e2_features_{name}.parquet shape={e2.shape} laundering rate={e2['Is Laundering'].mean():.3%}")
        if name == "train":
            pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:,.3f}".format)
            print("\nTrain graph feature averages, legit (0) vs laundering (1):")
            print(e2.groupby("Is Laundering")[IBM_GRAPH_FEATURE_COLUMNS].mean().T)
            print("\nCorrelation with E1 history counts (train):")
            print(f"  receiver in-degree vs receiver_prior_received_count: {e2.receiver_graph_in_degree.corr(e2.receiver_prior_received_count):.3f}")
            print(f"  sender out-degree vs sender_prior_sent_count:        {e2.sender_graph_out_degree.corr(e2.sender_prior_sent_count):.3f}")
            print(f"  receiver pagerank vs receiver in-degree:             {e2.receiver_graph_pagerank.corr(e2.receiver_graph_in_degree):.3f}")
            print(f"  receiver community size vs receiver in-degree:       {e2.receiver_graph_community_size.corr(e2.receiver_graph_in_degree):.3f}")
        del e1, e2


if __name__ == "__main__":
    main()
