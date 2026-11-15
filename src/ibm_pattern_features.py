"""IBM AML E2b: Graph-Feature-Preprocessor-style pattern features (Blanuša et al., ICAIF'24), written in Python.
Streaming in time order: features for a transaction at minute t use only transactions from earlier minutes
inside a time window (1 day; 6 hours for scatter-gather paths). No labels are used.
Run on the Mac:  python src/ibm_pattern_features.py          (all 5M transactions, then builds E2b files)
                 python src/ibm_pattern_features.py smoke    (first 300,000 transactions + checks)"""
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR, split_name
from src.ibm_features_tabular import usd_rates

DAY_WINDOW_MINUTES = 1440          # GFP uses one day for fan-in/out, cycles and account statistics
SCATTER_GATHER_WINDOW_MINUTES = 360  # GFP uses six hours for scatter-gather
INTERSECTION_CAP = 3000            # safety cap when intersecting neighbor sets of huge hub accounts

ROLE_FEATURES = ["fan_out_1d", "fan_in_1d", "sent_count_1d", "received_count_1d",
                 "sent_amount_sum_1d", "sent_amount_mean_1d", "sent_amount_std_1d",
                 "received_amount_sum_1d", "received_amount_mean_1d", "received_amount_std_1d",
                 "minutes_since_last_sent_1d", "minutes_since_last_received_1d"]
PAIR_FEATURES = ["same_pair_prior_count_1d", "reverse_pair_count_1d", "three_cycle_paths_1d", "scatter_gather_paths_6h"]
PATTERN_FEATURE_COLUMNS = [f"{role}_{name}" for role in ["sender", "receiver"] for name in ROLE_FEATURES] + PAIR_FEATURES


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_inputs(limit=None):
    columns = ["transaction_id", "timestamp", "sender_account_id", "receiver_account_id", "Amount Paid",
               "Payment Currency", "Receiving Currency", "Amount Received"]
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet", columns=columns)
    if limit:
        transactions = transactions.iloc[:limit].copy()
    rates = usd_rates(transactions, split_name(transactions.timestamp) == "train")
    amount_usd = (transactions["Amount Paid"] * transactions["Payment Currency"].astype(str).map(rates).fillna(1.0).values).values
    codes, _ = pd.factorize(pd.concat([transactions.sender_account_id, transactions.receiver_account_id], ignore_index=True))
    minutes = ((transactions.timestamp - pd.Timestamp("2022-09-01")).dt.total_seconds() // 60).astype(np.int64).values
    return transactions, codes[:len(transactions)], codes[len(transactions):], minutes, amount_usd


class Side:
    """One direction (sent or received) for every account: window queue, counterparty counts, running sums."""
    def __init__(self):
        self.queue, self.counterparties, self.sums = {}, {}, {}

    def expire(self, account, now):
        queue = self.queue.get(account)
        if not queue:
            return
        cutoff = now - DAY_WINDOW_MINUTES
        counterparties = self.counterparties[account]; sums = self.sums[account]
        while queue and queue[0][0] < cutoff:
            minute, other, amount = queue.popleft()
            entry = counterparties[other]; entry[0] -= 1
            if entry[0] == 0:
                del counterparties[other]
            sums[0] -= amount; sums[1] -= amount * amount

    def add(self, account, minute, other, amount):
        if account not in self.queue:
            self.queue[account] = deque(); self.counterparties[account] = {}; self.sums[account] = [0.0, 0.0]
        self.queue[account].append((minute, other, amount))
        entry = self.counterparties[account].get(other)
        if entry is None:
            self.counterparties[account][other] = [1, minute]
        else:
            entry[0] += 1; entry[1] = minute
        sums = self.sums[account]; sums[0] += amount; sums[1] += amount * amount

    def stats(self, account, now):
        queue = self.queue.get(account)
        if not queue:
            return 0, 0, 0.0, 0.0, 0.0, -1.0
        count = len(queue); total, total_square = self.sums[account]
        mean = total / count
        std = math.sqrt(max(total_square / count - mean * mean, 0.0))
        return len(self.counterparties[account]), count, max(total, 0.0), mean, std, float(now - queue[-1][0])


def shared_counterparties(first: dict, second: dict, since_minute=None):
    """Accounts present in both maps (optionally both edges at or after since_minute)."""
    small, large = (first, second) if len(first) <= len(second) else (second, first)
    shared = 0
    for checked, key in enumerate(small):
        if checked >= INTERSECTION_CAP:
            break
        other = large.get(key)
        if other is not None and (since_minute is None or (small[key][1] >= since_minute and other[1] >= since_minute)):
            shared += 1
    return shared


def compute(sender, receiver, minutes, amounts):
    sender = sender.tolist(); receiver = receiver.tolist(); minutes = minutes.tolist(); amounts = amounts.tolist()
    total = len(sender); features = np.zeros((total, len(PATTERN_FEATURE_COLUMNS)), dtype=np.float32)
    sent, received = Side(), Side()
    start = time.time(); index = 0
    while index < total:
        now = minutes[index]; end = index
        while end < total and minutes[end] == now:
            end += 1
        for row in range(index, end):          # features from strictly earlier minutes
            u, v = sender[row], receiver[row]
            for account in (u, v):
                sent.expire(account, now); received.expire(account, now)
            values = []
            for account in (u, v):
                fan_out, sent_count, sent_sum, sent_mean, sent_std, since_sent = sent.stats(account, now)
                fan_in, received_count, received_sum, received_mean, received_std, since_received = received.stats(account, now)
                values += [fan_out, fan_in, sent_count, received_count, sent_sum, sent_mean, sent_std,
                           received_sum, received_mean, received_std, since_sent, since_received]
            u_sent = sent.counterparties.get(u, {}); v_sent = sent.counterparties.get(v, {})
            u_received = received.counterparties.get(u, {}); v_received = received.counterparties.get(v, {})
            same_pair = u_sent.get(v, (0,))[0]
            reverse_pair = v_sent.get(u, (0,))[0] if u != v else 0
            three_cycle = shared_counterparties(v_sent, u_received) if u != v else 0          # v -> w -> u, then u -> v closes it
            scatter_gather = shared_counterparties(u_sent, v_received, now - SCATTER_GATHER_WINDOW_MINUTES) if u != v else 0  # u -> w -> v
            features[row] = values + [same_pair, reverse_pair, three_cycle, scatter_gather]
        for row in range(index, end):          # then add this minute's transactions to the window
            sent.add(sender[row], now, receiver[row], amounts[row]); received.add(receiver[row], now, sender[row], amounts[row])
        if end // 500_000 != index // 500_000:
            log(f"  {end:,} / {total:,} transactions ({time.time() - start:.0f}s)")
        index = end
    log(f"computed {total:,} transactions in {time.time() - start:.0f}s")
    return features


def leakage_and_correctness_test(transactions, sender, receiver, minutes, features, sample_size=500):
    """Recompute a few features by brute force for random rows: only earlier minutes inside the 1-day window."""
    rng = np.random.default_rng(42)
    rows = rng.choice(len(transactions), size=min(sample_size, len(transactions)), replace=False)
    column = {name: i for i, name in enumerate(PATTERN_FEATURE_COLUMNS)}
    order = np.argsort(sender, kind="stable"); sorted_sender = sender[order]
    order_r = np.argsort(receiver, kind="stable"); sorted_receiver = receiver[order_r]
    mismatches = 0
    for row in rows:
        u, v, t = sender[row], receiver[row], minutes[row]
        lo, hi = np.searchsorted(sorted_sender, u, "left"), np.searchsorted(sorted_sender, u, "right")
        u_edges = order[lo:hi]; u_window = u_edges[(minutes[u_edges] >= t - DAY_WINDOW_MINUTES) & (minutes[u_edges] < t)]
        lo, hi = np.searchsorted(sorted_receiver, v, "left"), np.searchsorted(sorted_receiver, v, "right")
        v_edges = order_r[lo:hi]; v_window = v_edges[(minutes[v_edges] >= t - DAY_WINDOW_MINUTES) & (minutes[v_edges] < t)]
        expected = {"sender_sent_count_1d": len(u_window), "sender_fan_out_1d": len(np.unique(receiver[u_window])),
                    "receiver_received_count_1d": len(v_window), "receiver_fan_in_1d": len(np.unique(sender[v_window])),
                    "same_pair_prior_count_1d": int((receiver[u_window] == v).sum())}
        for name, value in expected.items():
            if features[row, column[name]] != value:
                mismatches += 1
    log(f"correctness + past-only test on {len(rows)} random transactions: mismatches = {mismatches}")
    assert mismatches == 0


def build_experiment_files():
    """E2b = E1 + pattern features; ibm_e2p = E1 + Neo4j graph features + pattern features."""
    patterns = pd.read_parquet(PROCESSED_DIR / "ibm_pattern_features.parquet")
    for name in ["train", "validation", "test"]:
        for source, target in [("ibm_e1_features", "ibm_e2b_features"), ("ibm_e2_features", "ibm_e2p_features")]:
            base = pd.read_parquet(PROCESSED_DIR / f"{source}_{name}.parquet")
            merged = base.merge(patterns, on="transaction_id", how="left")
            assert len(merged) == len(base) and merged[PATTERN_FEATURE_COLUMNS].notna().all().all()
            merged.to_parquet(PROCESSED_DIR / f"{target}_{name}.parquet", index=False)
            log(f"saved {target}_{name}.parquet shape={merged.shape}")
            del base, merged


def main(smoke=False):
    transactions, sender, receiver, minutes, amounts = load_inputs(limit=300_000 if smoke else None)
    log(f"loaded {len(transactions):,} transactions")
    features = compute(sender, receiver, minutes, amounts)
    leakage_and_correctness_test(transactions, sender, receiver, minutes, features)
    table = pd.DataFrame(features, columns=PATTERN_FEATURE_COLUMNS)
    table.insert(0, "transaction_id", transactions.transaction_id.values)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(table.describe().T[["mean", "max"]].to_string(float_format=lambda x: f"{x:,.2f}"))
    if smoke:
        log("SMOKE TEST OK"); return
    table.to_parquet(PROCESSED_DIR / "ibm_pattern_features.parquet", index=False)
    log("saved ibm_pattern_features.parquet")
    build_experiment_files()
    log("ALL DONE")


if __name__ == "__main__":
    main(smoke=len(sys.argv) > 1 and sys.argv[1] == "smoke")
