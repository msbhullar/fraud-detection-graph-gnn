"""IBM AML round 2 causal features (decided 2026-10-28, see docs/leakage_notes.md).
Group 1: multi-window account activity for 1 hour, 6 hours and 7 days
         (the 1-day window already exists in src/ibm_pattern_features.py).
Group 2: pass-through (money comes in, then goes out again soon), using all earlier history.
Group 3: long pair history (same sender -> receiver pair, and the reverse pair), using all earlier history
         (the 1-day pair counts already exist in src/ibm_pattern_features.py).
Group 4: the sender's payment format and currency mix over the last 7 days (is this payment unusual for them?).
Group 5: time-ordered 3-cycles in the last day: u -> v closes a loop only if v -> w happened BEFORE w -> u
         (src/ibm_pattern_features.py counts v -> w and w -> u in any order).
Streaming in time order: a transaction at minute t uses only transactions from EARLIER minutes. No labels.
Run:  python src/ibm_window_features.py smoke   (first 1,000,000 transactions + checks, nothing saved)
      python src/ibm_window_features.py         (all 5M transactions, saves data/processed/ibm_aml/ibm_window_features.parquet)"""
import gc
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR
from src.ibm_pattern_features import load_inputs

WINDOWS_MINUTES = {"1h": 60, "6h": 360, "7d": 10080}
DIRECTION_STATS = {"sent": ["sent_count", "fan_out", "sent_amount_sum"],
                   "received": ["received_count", "fan_in", "received_amount_sum"]}
WINDOW_FEATURE_COLUMNS = [f"{role}_{stat}_{window}"
                          for role in ["sender", "receiver"]
                          for window in WINDOWS_MINUTES
                          for direction in ["sent", "received"]
                          for stat in DIRECTION_STATS[direction]]
NO_HISTORY_VALUE = -1.0
PASS_THROUGH_COLUMNS = ["sender_minutes_since_last_received_any", "sender_amount_vs_last_received_amount",
                        "sender_inflow_vs_current_amount_1h", "sender_inflow_vs_current_amount_6h",
                        "sender_net_flow_after_this_payment_6h", "receiver_outflow_vs_inflow_6h"]
PAIR_HISTORY_COLUMNS = ["pair_prior_count_any", "pair_minutes_since_last_payment_any", "pair_amount_vs_previous_amount_any",
                        "reverse_pair_prior_count_any", "reverse_pair_minutes_since_last_payment_any"]
MIX_WINDOW_MINUTES = 10080                     # 7 days
MIX_COLUMNS = ["sender_distinct_formats_7d", "sender_distinct_currencies_7d", "sender_ach_share_7d",
               "sender_format_is_new_7d", "sender_currency_is_new_7d"]
CYCLE_WINDOW_MINUTES = 1440                    # 1 day, same as the existing pattern features
INTERSECTION_CAP = 3000                        # same safety cap as src/ibm_pattern_features.py for huge hub accounts
CYCLE_COLUMNS = ["three_cycle_ordered_1d"]
ALL_FEATURE_COLUMNS = WINDOW_FEATURE_COLUMNS + PASS_THROUGH_COLUMNS + PAIR_HISTORY_COLUMNS + MIX_COLUMNS + CYCLE_COLUMNS
SMOKE_ROWS = 1_000_000   # about the first 21 hours of Sept 1 (the first 300,000 rows cover only 27 minutes)


def load_format_and_currency(limit=None):
    """Payment Format and Payment Currency as integer codes, in the same row order as load_inputs()."""
    mix = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet", columns=["Payment Format", "Payment Currency"])
    if limit:
        mix = mix.iloc[:limit]
    ach_code = list(mix["Payment Format"].cat.categories).index("ACH")
    return mix["Payment Format"].cat.codes.values.astype(np.int16), mix["Payment Currency"].cat.codes.values.astype(np.int16), ach_code


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class WindowSide:
    """One direction (sent or received) inside one time window, for every account.
    One shared time-ordered queue of (minute, account, counterparty, amount) per window; per account only
    [payment count, amount sum, {counterparty: count}]. Accounts with nothing left in the window are removed."""
    def __init__(self, window_minutes):
        self.window_minutes = window_minutes
        self.events, self.state = deque(), {}

    def expire_before(self, now):
        cutoff = now - self.window_minutes
        events, state = self.events, self.state
        while events and events[0][0] < cutoff:
            _, account, other, amount = events.popleft()
            entry = state[account]
            entry[0] -= 1; entry[1] -= amount
            counterparties = entry[2]
            counterparties[other] -= 1
            if counterparties[other] == 0:
                del counterparties[other]
            if entry[0] == 0:
                del state[account]

    def add(self, account, minute, other, amount):
        self.events.append((minute, account, other, amount))
        entry = self.state.get(account)
        if entry is None:
            self.state[account] = [1, amount, {other: 1}]
        else:
            entry[0] += 1; entry[1] += amount
            entry[2][other] = entry[2].get(other, 0) + 1

    def stats(self, account):
        entry = self.state.get(account)
        if entry is None:
            return 0, 0, 0.0
        return entry[0], len(entry[2]), max(entry[1], 0.0)


class TimedSide:
    """One direction inside the 1-day window, keeping the minutes of every payment per counterparty
    (a short list, oldest first), so we know the earliest and the latest payment to each counterparty."""
    def __init__(self, window_minutes):
        self.window_minutes = window_minutes
        self.events, self.counterparty_minutes = deque(), {}

    def expire_before(self, now):
        cutoff = now - self.window_minutes
        events, state = self.events, self.counterparty_minutes
        while events and events[0][0] < cutoff:
            _, account, other = events.popleft()
            by_counterparty = state[account]
            minutes_list = by_counterparty[other]
            del minutes_list[0]                    # the shared queue is in time order, so the oldest goes first
            if not minutes_list:
                del by_counterparty[other]
                if not by_counterparty:
                    del state[account]

    def add(self, account, minute, other):
        self.events.append((minute, account, other))
        self.counterparty_minutes.setdefault(account, {}).setdefault(other, []).append(minute)

    def get(self, account):
        return self.counterparty_minutes.get(account, {})


class MixSide:
    """The sender's payment formats and currencies inside the 7-day window: [payment count, {format: n}, {currency: n}]."""
    def __init__(self, window_minutes):
        self.window_minutes = window_minutes
        self.events, self.state = deque(), {}

    def expire_before(self, now):
        cutoff = now - self.window_minutes
        events, state = self.events, self.state
        while events and events[0][0] < cutoff:
            _, account, payment_format, payment_currency = events.popleft()
            entry = state[account]
            entry[0] -= 1
            for counts, key in ((entry[1], payment_format), (entry[2], payment_currency)):
                counts[key] -= 1
                if counts[key] == 0:
                    del counts[key]
            if entry[0] == 0:
                del state[account]

    def add(self, account, minute, payment_format, payment_currency):
        self.events.append((minute, account, payment_format, payment_currency))
        entry = self.state.get(account)
        if entry is None:
            self.state[account] = [1, {payment_format: 1}, {payment_currency: 1}]
        else:
            entry[0] += 1
            entry[1][payment_format] = entry[1].get(payment_format, 0) + 1
            entry[2][payment_currency] = entry[2].get(payment_currency, 0) + 1


def ordered_three_cycles(v_sent: dict, u_received: dict):
    """Count accounts w with a payment v -> w that happened before some payment w -> u."""
    small, large = (v_sent, u_received) if len(v_sent) <= len(u_received) else (u_received, v_sent)
    small_is_v_sent = small is v_sent
    count = 0
    for checked, w in enumerate(small):
        if checked >= INTERSECTION_CAP:
            break
        other = large.get(w)
        if other is None:
            continue
        v_to_w, w_to_u = (small[w], other) if small_is_v_sent else (other, small[w])
        if v_to_w[0] < w_to_u[-1]:                 # earliest v -> w is before the latest w -> u
            count += 1
    return count


def compute(*arguments):
    """Python's garbage collector keeps rescanning the millions of small window objects; pausing it during the
    stream makes the run several times faster and does not change any result."""
    gc.disable()
    try:
        return compute_stream(*arguments)
    finally:
        gc.enable()


def compute_stream(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code):
    payment_formats = payment_formats.tolist(); payment_currencies = payment_currencies.tolist()
    sender = sender.tolist(); receiver = receiver.tolist(); minutes = minutes.tolist(); amounts = amounts.tolist()
    total = len(sender)
    features = np.zeros((total, len(WINDOW_FEATURE_COLUMNS)), dtype=np.float32)   # group 1; group 2 is added at the end
    window_sides = [(WindowSide(length), WindowSide(length)) for length in WINDOWS_MINUTES.values()]   # (sent, received) per window
    last_received = {}                           # account -> (minute, amount) of its latest incoming payment
    last_received_minute = np.full(total, np.nan); last_received_amount = np.full(total, np.nan)
    pair_history = {}                            # (sender, receiver) -> [payment count, last minute, last amount]
    pair_features = np.full((total, len(PAIR_HISTORY_COLUMNS)), NO_HISTORY_VALUE, dtype=np.float32)
    mix = MixSide(MIX_WINDOW_MINUTES)
    mix_features = np.full((total, len(MIX_COLUMNS)), NO_HISTORY_VALUE, dtype=np.float32)
    cycle_sent, cycle_received = TimedSide(CYCLE_WINDOW_MINUTES), TimedSide(CYCLE_WINDOW_MINUTES)
    cycle_features = np.zeros((total, len(CYCLE_COLUMNS)), dtype=np.float32)
    all_windows = [side for pair in window_sides for side in pair] + [mix, cycle_sent, cycle_received]
    start = time.time(); index = 0
    while index < total:
        now = minutes[index]; end = index
        while end < total and minutes[end] == now:
            end += 1
        for side in all_windows:                 # drop everything older than each window, once per minute
            side.expire_before(now)
        for row in range(index, end):            # features from strictly earlier minutes
            u, v = sender[row], receiver[row]
            values = []
            for account in (u, v):
                for sent_side, received_side in window_sides:
                    values += sent_side.stats(account)
                    values += received_side.stats(account)
            features[row] = values
            previous = last_received.get(u)
            if previous is not None:
                last_received_minute[row], last_received_amount[row] = previous
            sender_mix = mix.state.get(u)
            if sender_mix is not None:           # sender has sent something in the last 7 days
                count, formats, currencies = sender_mix
                mix_features[row] = [len(formats), len(currencies), formats.get(ach_code, 0) / count,
                                     int(payment_formats[row] not in formats), int(payment_currencies[row] not in currencies)]
            if u != v:
                cycle_features[row, 0] = ordered_three_cycles(cycle_sent.get(v), cycle_received.get(u))
            same = pair_history.get((u, v))
            pair_features[row, 0] = 0
            if same is not None:
                pair_features[row, 0:3] = [same[0], now - same[1], amounts[row] / (same[2] + 1.0)]
            if u != v:                           # self-transfers have no separate reverse pair (stays -1)
                pair_features[row, 3] = 0
                reverse = pair_history.get((v, u))
                if reverse is not None:
                    pair_features[row, 3:5] = [reverse[0], now - reverse[1]]
        for row in range(index, end):            # then add this minute's transactions
            u, v, amount = sender[row], receiver[row], amounts[row]
            for sent_side, received_side in window_sides:
                sent_side.add(u, now, v, amount)
                received_side.add(v, now, u, amount)
            last_received[v] = (now, amount)
            cycle_sent.add(u, now, v); cycle_received.add(v, now, u)
            mix.add(u, now, payment_formats[row], payment_currencies[row])
            entry = pair_history.get((u, v))
            if entry is None:
                pair_history[(u, v)] = [1, now, amount]
            else:
                entry[0] += 1; entry[1] = now; entry[2] = amount
        if end // 500_000 != index // 500_000:
            log(f"  {end:,} / {total:,} transactions ({time.time() - start:.0f}s)")
        index = end
    log(f"computed {total:,} transactions in {time.time() - start:.0f}s")
    return np.hstack([features, pass_through_features(features, np.asarray(minutes), np.asarray(amounts),
                                                      last_received_minute, last_received_amount), pair_features, mix_features, cycle_features])


def pass_through_features(window_features, minutes, amounts, last_received_minute, last_received_amount):
    """Group 2. A mule account receives money and sends most of it on quickly."""
    column = {name: window_features[:, i].astype(np.float64) for i, name in enumerate(WINDOW_FEATURE_COLUMNS)}
    has_history = ~np.isnan(last_received_minute)
    output = np.full((len(minutes), len(PASS_THROUGH_COLUMNS)), NO_HISTORY_VALUE)
    output[has_history, 0] = minutes[has_history] - last_received_minute[has_history]
    output[has_history, 1] = amounts[has_history] / (last_received_amount[has_history] + 1.0)
    output[:, 2] = column["sender_received_amount_sum_1h"] / (amounts + 1.0)
    output[:, 3] = column["sender_received_amount_sum_6h"] / (amounts + 1.0)
    output[:, 4] = column["sender_received_amount_sum_6h"] - column["sender_sent_amount_sum_6h"] - amounts
    output[:, 5] = column["receiver_sent_amount_sum_6h"] / (column["receiver_received_amount_sum_6h"] + 1.0)
    return output.astype(np.float32)


def brute_force_test(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code, features, sample_size=500):
    """Recompute every column by brute force for random rows: only earlier minutes inside each window."""
    rng = np.random.default_rng(42)
    rows = rng.choice(len(sender), size=min(sample_size, len(sender)), replace=False)
    column = {name: i for i, name in enumerate(ALL_FEATURE_COLUMNS)}
    order_by_sender = np.argsort(sender, kind="stable"); sorted_sender = sender[order_by_sender]
    order_by_receiver = np.argsort(receiver, kind="stable"); sorted_receiver = receiver[order_by_receiver]

    def edges_of(account, sorted_ids, order):
        lo, hi = np.searchsorted(sorted_ids, account, "left"), np.searchsorted(sorted_ids, account, "right")
        return order[lo:hi]

    mismatches = 0
    for row in rows:
        t = minutes[row]
        for role, account in [("sender", sender[row]), ("receiver", receiver[row])]:
            sent_edges = edges_of(account, sorted_sender, order_by_sender)
            received_edges = edges_of(account, sorted_receiver, order_by_receiver)
            for window, length in WINDOWS_MINUTES.items():
                sent_in = sent_edges[(minutes[sent_edges] >= t - length) & (minutes[sent_edges] < t)]
                received_in = received_edges[(minutes[received_edges] >= t - length) & (minutes[received_edges] < t)]
                expected = {f"{role}_sent_count_{window}": len(sent_in),
                            f"{role}_fan_out_{window}": len(np.unique(receiver[sent_in])),
                            f"{role}_sent_amount_sum_{window}": amounts[sent_in].sum(),
                            f"{role}_received_count_{window}": len(received_in),
                            f"{role}_fan_in_{window}": len(np.unique(sender[received_in])),
                            f"{role}_received_amount_sum_{window}": amounts[received_in].sum()}
                for name, value in expected.items():
                    if not np.isclose(features[row, column[name]], value, rtol=1e-4, atol=1e-2):
                        mismatches += 1
        sender_received = edges_of(sender[row], sorted_receiver, order_by_receiver)
        earlier = sender_received[minutes[sender_received] < t]
        if len(earlier):
            latest = earlier[np.argmax(minutes[earlier])]       # same-minute ties: the stream keeps the last row of that minute
            tied = earlier[minutes[earlier] == minutes[latest]]
            expected_minutes = t - minutes[latest]
            expected_ratio_options = amounts[row] / (amounts[tied] + 1.0)
            if features[row, column["sender_minutes_since_last_received_any"]] != expected_minutes:
                mismatches += 1
            if not np.isclose(expected_ratio_options, features[row, column["sender_amount_vs_last_received_amount"]], rtol=1e-4).any():
                mismatches += 1
        elif features[row, column["sender_minutes_since_last_received_any"]] != NO_HISTORY_VALUE:
            mismatches += 1
        u, v = sender[row], receiver[row]
        sender_sent = edges_of(u, sorted_sender, order_by_sender)
        same_pair = sender_sent[(receiver[sender_sent] == v) & (minutes[sender_sent] < t)]
        if features[row, column["pair_prior_count_any"]] != len(same_pair):
            mismatches += 1
        if len(same_pair):
            last_minute = minutes[same_pair].max(); tied = same_pair[minutes[same_pair] == last_minute]
            if features[row, column["pair_minutes_since_last_payment_any"]] != t - last_minute:
                mismatches += 1
            if not np.isclose(amounts[row] / (amounts[tied] + 1.0), features[row, column["pair_amount_vs_previous_amount_any"]], rtol=1e-4).any():
                mismatches += 1
        elif features[row, column["pair_minutes_since_last_payment_any"]] != NO_HISTORY_VALUE:
            mismatches += 1
        if u != v:
            receiver_sent = edges_of(v, sorted_sender, order_by_sender)
            reverse_pair = receiver_sent[(receiver[receiver_sent] == u) & (minutes[receiver_sent] < t)]
            if features[row, column["reverse_pair_prior_count_any"]] != len(reverse_pair):
                mismatches += 1
            expected_since = t - minutes[reverse_pair].max() if len(reverse_pair) else NO_HISTORY_VALUE
            if features[row, column["reverse_pair_minutes_since_last_payment_any"]] != expected_since:
                mismatches += 1
        recent = sender_sent[(minutes[sender_sent] >= t - MIX_WINDOW_MINUTES) & (minutes[sender_sent] < t)]
        if len(recent):
            expected_mix = [len(np.unique(payment_formats[recent])), len(np.unique(payment_currencies[recent])),
                            (payment_formats[recent] == ach_code).mean(),
                            int(payment_formats[row] not in set(payment_formats[recent])),
                            int(payment_currencies[row] not in set(payment_currencies[recent]))]
        else:
            expected_mix = [NO_HISTORY_VALUE] * len(MIX_COLUMNS)
        for name, value in zip(MIX_COLUMNS, expected_mix):
            if not np.isclose(features[row, column[name]], value, rtol=1e-5, atol=1e-6):
                mismatches += 1
        if u != v:                                 # ordered 3-cycles: v -> w at t1, w -> u at t2, t1 < t2 < t, both inside 1 day
            v_sent = edges_of(v, sorted_sender, order_by_sender)
            v_sent = v_sent[(minutes[v_sent] >= t - CYCLE_WINDOW_MINUTES) & (minutes[v_sent] < t)]
            u_received = edges_of(u, sorted_receiver, order_by_receiver)
            u_received = u_received[(minutes[u_received] >= t - CYCLE_WINDOW_MINUTES) & (minutes[u_received] < t)]
            if min(len(np.unique(receiver[v_sent])), len(np.unique(sender[u_received]))) <= INTERSECTION_CAP:
                earliest_v_to_w = pd.Series(minutes[v_sent]).groupby(receiver[v_sent]).min()
                latest_w_to_u = pd.Series(minutes[u_received]).groupby(sender[u_received]).max()
                shared = earliest_v_to_w.index.intersection(latest_w_to_u.index)
                expected_cycles = int((earliest_v_to_w[shared].values < latest_w_to_u[shared].values).sum())
                if features[row, column["three_cycle_ordered_1d"]] != expected_cycles:
                    mismatches += 1
    log(f"brute-force + past-only test on {len(rows)} random transactions x {len(ALL_FEATURE_COLUMNS)} columns: mismatches = {mismatches}")
    assert mismatches == 0


def future_append_test(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code,
                       prefix_rows=100_000, extra_rows=50_000):
    """Adding later transactions must not change any earlier row's features."""
    def first(rows):
        return (sender[:rows], receiver[:rows], minutes[:rows], amounts[:rows], payment_formats[:rows], payment_currencies[:rows], ach_code)
    short = compute(*first(prefix_rows))
    longer = compute(*first(prefix_rows + extra_rows))
    changed_rows = int((short != longer[:prefix_rows]).any(axis=1).sum())
    log(f"future-append test: first {prefix_rows:,} rows, then {extra_rows:,} later rows appended -> changed rows = {changed_rows}")
    assert changed_rows == 0


def main(smoke=False):
    transactions, sender, receiver, minutes, amounts = load_inputs(limit=SMOKE_ROWS if smoke else None)
    log(f"loaded {len(transactions):,} transactions, {transactions.timestamp.min()} to {transactions.timestamp.max()}")
    payment_formats, payment_currencies, ach_code = load_format_and_currency(limit=SMOKE_ROWS if smoke else None)
    features = compute(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code)
    brute_force_test(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code, features)
    future_append_test(sender, receiver, minutes, amounts, payment_formats, payment_currencies, ach_code)
    table = pd.DataFrame(features, columns=ALL_FEATURE_COLUMNS)
    table.insert(0, "transaction_id", transactions.transaction_id.values)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 12)
    log(f"window feature table shape = {table.shape}")
    print(table[["transaction_id"] + CYCLE_COLUMNS].sample(5, random_state=1).to_string())
    print(table[CYCLE_COLUMNS].describe().T[["mean", "50%", "max"]].to_string(float_format=lambda x: f"{x:,.2f}"))
    if smoke:
        log("SMOKE TEST OK (nothing saved)"); return
    table.to_parquet(PROCESSED_DIR / "ibm_window_features.parquet", index=False)
    log("saved ibm_window_features.parquet")


if __name__ == "__main__":
    main(smoke=len(sys.argv) > 1 and sys.argv[1] == "smoke")
