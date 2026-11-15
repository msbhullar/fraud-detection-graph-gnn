"""E4 v2 (step 2 of the 2026-11-12 plan): same past-only 2-hop subgraph text as src/e4_subgraph.py, plus rule-made GRAPH facts
(distinct counterparties, common counterparties, A -> other account -> B paths, and the earlier-days graph snapshot numbers)
written as numbered fact lines so the LLM only translates them and a checker can verify them. The old G1/G2 lines are removed.
This file is a patched copy of src/e4_subgraph.py (the original is unchanged).
Leakage rules: only transactions from EARLIER minutes than the flagged payment are used, and labels are never read.
Run from the project root:
    python src/e4_subgraph_v2.py build   # writes results/e4/subgraph_cases.jsonl and prints 3 example texts
    python src/e4_subgraph_v2.py test    # automated checks (past-only edges, facts recomputed by brute force)
The LLM later receives only the 'text' field. Facts come from rules on ALL earlier transactions of the sender and
receiver (not only the trimmed edges shown), so the test recomputes them from the raw transaction arrays."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibm_data_prep import PROCESSED_DIR, TRAIN_END
from src.ibm_features_tabular import usd_rates

# Optional: build another case set, e.g.  E4_CASES_PATH=results/e4/cases_typology.csv E4_SUBGRAPH_OUTPUT=results/e4/subgraph_cases_typology_v2.jsonl python src/e4_subgraph_v2.py build
CASES_PATH = Path(os.environ.get("E4_CASES_PATH", "results/e4/cases.csv"))
OUTPUT_PATH = Path(os.environ.get("E4_SUBGRAPH_OUTPUT", "results/e4/subgraph_cases_v2.jsonl"))
ALERT_THRESHOLD = json.loads(Path("results_sol/ibm_e2r_tuned_metrics.json").read_text())["models"]["xgboost"]["validation_precision_floor"]["threshold"]
TOKEN_BUDGET = 1500
CHARACTERS_PER_TOKEN = 2.2       # conservative (llama ~2.8, qwen ~2.2 characters per token); Ollama logs the real prompt token count
BIG_AMOUNT_RATIO = 5.0           # signal: payment >= 5x the sender's median earlier payment (fixed in advance, never tuned on labels)
FAN_THRESHOLD = 5                # signal: >= 5 different counterparties in 24 h
START_NEIGHBORS_PER_HOP = 10
MINIMUM_NEIGHBORS_PER_HOP = 3
GRAPH_FEATURE_COLUMNS = ["sender_graph_in_degree", "sender_graph_out_degree", "sender_graph_pagerank", "sender_graph_community_size",
                         "receiver_graph_in_degree", "receiver_graph_out_degree", "receiver_graph_pagerank", "receiver_graph_community_size"]


def as_codes(series: pd.Series):
    categorical = series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")
    return categorical.cat.codes.values.astype(np.int16), [str(name) for name in categorical.cat.categories]


class TransactionIndex:
    """All IBM transactions as compact arrays, sorted by time, with per-account row lists."""

    def __init__(self):
        frame = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                columns=["timestamp", "sender_account_id", "receiver_account_id", "Amount Paid", "Amount Received",
                                         "Payment Currency", "Receiving Currency", "Payment Format"])
        rate_by_currency = usd_rates(frame, frame["timestamp"] < TRAIN_END)  # training-period rates, same as the models
        self.minutes = frame["timestamp"].values.astype("datetime64[m]").astype(np.int64)
        assert (np.diff(self.minutes) >= 0).all(), "transactions must be sorted by time"
        number_of_rows = len(frame)
        codes, names = pd.factorize(pd.concat([frame["sender_account_id"], frame["receiver_account_id"]], ignore_index=True))
        self.sender, self.receiver = codes[:number_of_rows].astype(np.int32), codes[number_of_rows:].astype(np.int32)
        self.account_index = names
        self.amount_paid = frame["Amount Paid"].values.astype(np.float64)
        self.currency_codes, self.currency_names = as_codes(frame["Payment Currency"])
        self.format_codes, self.format_names = as_codes(frame["Payment Format"])
        rates = np.array([rate_by_currency.get(name, 1.0) for name in self.currency_names])
        self.amount_usd = self.amount_paid * rates[self.currency_codes]
        del frame, codes
        number_of_accounts = len(names)
        self.rows_by_sender = np.argsort(self.sender, kind="stable")      # stable: rows stay in time order per account
        self.sender_bounds = np.searchsorted(self.sender[self.rows_by_sender], np.arange(number_of_accounts + 1))
        self.rows_by_receiver = np.argsort(self.receiver, kind="stable")
        self.receiver_bounds = np.searchsorted(self.receiver[self.rows_by_receiver], np.arange(number_of_accounts + 1))

    def past_rows_of(self, account_code: int, boundary_row: int):
        """Rows where the account sent / received, strictly before boundary_row (= first row of the flagged minute)."""
        sent = self.rows_by_sender[self.sender_bounds[account_code]:self.sender_bounds[account_code + 1]]
        received = self.rows_by_receiver[self.receiver_bounds[account_code]:self.receiver_bounds[account_code + 1]]
        return sent[:np.searchsorted(sent, boundary_row)], received[:np.searchsorted(received, boundary_row)]

    def rows_touching(self, account_codes, boundary_row: int) -> np.ndarray:
        parts = []
        for code in account_codes:
            sent, received = self.past_rows_of(int(code), boundary_row)
            parts += [sent, received]
        return np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)

    def rank_and_cut(self, rows: np.ndarray, keep: int):
        """Top `keep` rows by USD amount, then recency. Returns (kept rows in time order, dropped rows)."""
        order = np.lexsort((-rows, -self.amount_usd[rows]))
        return np.sort(rows[order[:keep]]), rows[order[keep:]]


def fact_numbers(index: TransactionIndex, sender_code: int, receiver_code: int, boundary_row: int, case_minute: int, row: int) -> dict:
    """Plain numbers behind the facts (also used by the brute-force test)."""
    a_sent, a_received = index.past_rows_of(sender_code, boundary_row)
    b_sent, b_received = index.past_rows_of(receiver_code, boundary_row)
    pair_rows = a_sent[index.receiver[a_sent] == receiver_code]
    reverse_rows = a_received[index.sender[a_received] == receiver_code]
    inflow_6h = a_received[index.minutes[a_received] >= case_minute - 360]
    outflow_6h = a_sent[index.minutes[a_sent] >= case_minute - 360]
    sender_partners = np.unique(np.concatenate([index.receiver[a_sent], index.sender[a_received]]))
    receiver_partners = np.unique(np.concatenate([index.receiver[b_sent], index.sender[b_received]]))
    sender_partners = sender_partners[sender_partners != sender_code]
    receiver_partners = receiver_partners[receiver_partners != receiver_code]
    other_accounts = np.array([sender_code, receiver_code])
    common_partners = np.setdiff1d(np.intersect1d(sender_partners, receiver_partners), other_accounts)
    relay_accounts = np.setdiff1d(np.intersect1d(np.unique(index.receiver[a_sent]), np.unique(index.sender[b_received])), other_accounts)
    return {
        "sender_distinct_partners": int(len(sender_partners)), "receiver_distinct_partners": int(len(receiver_partners)),
        "common_partner_count": int(len(common_partners)), "relay_path_count": int(len(relay_accounts)),
        "sender_past_sent": int(len(a_sent)), "sender_past_received": int(len(a_received)),
        "receiver_past_sent": int(len(b_sent)), "receiver_past_received": int(len(b_received)),
        "pair_count": int(len(pair_rows)), "reverse_count": int(len(reverse_rows)),
        "last_pair_minutes_ago": int(case_minute - index.minutes[pair_rows[-1]]) if len(pair_rows) else None,
        "last_reverse_minutes_ago": int(case_minute - index.minutes[reverse_rows[-1]]) if len(reverse_rows) else None,
        "sender_inflow_6h_count": int(len(inflow_6h)), "sender_inflow_6h_usd": float(index.amount_usd[inflow_6h].sum()),
        "sender_outflow_6h_count": int(len(outflow_6h)),
        "sender_distinct_receivers_24h": int(np.unique(index.receiver[a_sent[index.minutes[a_sent] >= case_minute - 1440]]).size),
        "receiver_distinct_senders_24h": int(np.unique(index.sender[b_received[index.minutes[b_received] >= case_minute - 1440]]).size),
        "sender_median_past_usd": float(np.median(index.amount_usd[a_sent])) if len(a_sent) else None,
        "sender_distinct_currencies_before": int(np.unique(index.currency_codes[a_sent]).size),
        "currency_is_new_for_sender": bool(len(a_sent) > 0 and not (index.currency_codes[a_sent] == index.currency_codes[row]).any()),
        "format_is_new_for_sender": bool(len(a_sent) > 0 and not (index.format_codes[a_sent] == index.format_codes[row]).any()),
    }


def describe_hours(minutes: int) -> str:
    return f"{minutes / 60:.1f} h ago"


def build_facts(index, case, sender_code, receiver_code, row, numbers):
    """Rule layer: numbered statements backed by numbers from earlier transactions only, plus the list of
    RISK SIGNALS (facts whose rule fired). Thresholds are fixed in advance and never tuned on labels."""
    facts, signals = [], []

    def add(text, is_signal=False):
        facts.append(text)
        if is_signal:
            signals.append(len(facts))

    self_transfer = sender_code == receiver_code
    add(f"The model gave this payment a risk score of {case['fraud_score']:.2f}; the alert threshold is {ALERT_THRESHOLD:.2f}.")
    if self_transfer:
        add("Sender and receiver are the same account (a self-transfer).", True)
    elif numbers["pair_count"] == 0:
        add("This is the first recorded payment from A to B.", True)
    else:
        add(f"A paid B {numbers['pair_count']} time(s) before; the last was {describe_hours(numbers['last_pair_minutes_ago'])}.")
    if not self_transfer:
        if numbers["reverse_count"] > 0:
            add(f"B paid A {numbers['reverse_count']} time(s) before (last {describe_hours(numbers['last_reverse_minutes_ago'])}), so money flows both ways.", True)
        else:
            add("B has never paid A before.")
    if numbers["sender_past_sent"] + numbers["sender_past_received"] == 0:
        add("A has no earlier activity in the data (a new account).", True)
    else:
        add(f"Before this payment A sent {numbers['sender_past_sent']} and received {numbers['sender_past_received']} payment(s).")
    if not self_transfer:
        if numbers["receiver_past_sent"] + numbers["receiver_past_received"] == 0:
            add("B has no earlier activity in the data (a new account).", True)
        else:
            add(f"Before this payment B sent {numbers['receiver_past_sent']} and received {numbers['receiver_past_received']} payment(s).")
    if numbers["sender_past_sent"] > 0:
        currency_name = index.currency_names[index.currency_codes[row]]
        format_name = index.format_names[index.format_codes[row]]
        if numbers["currency_is_new_for_sender"]:
            add(f"{currency_name} is a NEW payment currency for A (A paid in {numbers['sender_distinct_currencies_before']} other currency/currencies before).", True)
        else:
            add(f"{currency_name} is a currency A has used before.")
        if numbers["format_is_new_for_sender"]:
            add(f"{format_name} is a NEW payment format for A.", True)
        else:
            add(f"{format_name} is a payment format A has used before.")
        median_usd = numbers["sender_median_past_usd"]
        ratio = index.amount_usd[row] / max(median_usd, 1.0)
        add(f"This payment is about {index.amount_usd[row]:,.0f} USD, {ratio:.1f}x A's median earlier payment ({median_usd:,.0f} USD).", bool(ratio >= BIG_AMOUNT_RATIO))
    if numbers["sender_inflow_6h_count"] > 0:
        add(f"In the 6 h before, A received {numbers['sender_inflow_6h_count']} payment(s) worth about {numbers['sender_inflow_6h_usd']:,.0f} USD "
            f"and sent {numbers['sender_outflow_6h_count']} other payment(s).", True)
    else:
        add(f"In the 6 h before, A received no payments and sent {numbers['sender_outflow_6h_count']} other payment(s).")
    add(f"In the last 24 h A paid {numbers['sender_distinct_receivers_24h']} different account(s); B received from {numbers['receiver_distinct_senders_24h']} different account(s).",
        bool(numbers["sender_distinct_receivers_24h"] >= FAN_THRESHOLD or numbers["receiver_distinct_senders_24h"] >= FAN_THRESHOLD))
    numbers["first_graph_fact_number"] = len(facts) + 1      # F-lines from here to the last one are the graph facts
    if self_transfer:      # A and B are the same account: say it once
        add(f"Before this payment A had traded with {numbers['sender_distinct_partners']} different account(s).")
    else:
        add(f"Before this payment A had traded with {numbers['sender_distinct_partners']} different account(s) and B with {numbers['receiver_distinct_partners']}.")
    if not self_transfer:
        add(f"{numbers['common_partner_count']} account(s) traded with both A and B before.")
        if numbers["relay_path_count"] > 0:
            add(f"{numbers['relay_path_count']} other account(s) received money from A and also paid B earlier, so money could already have moved from A to B through another account.", True)
        else:
            add("No other account both received money from A and paid B earlier.")
    if numbers.get("graph_available"):
        for role, label in ([("sender", "A")] if self_transfer else [("sender", "A"), ("receiver", "B")]):
            add(f"In the graph built from earlier days, {label} had {numbers[f'{role}_graph_in_degree']:.0f} incoming and {numbers[f'{role}_graph_out_degree']:.0f} outgoing links, "
                f"PageRank {numbers[f'{role}_graph_pagerank']:.2f}, and belonged to a community of {numbers[f'{role}_graph_community_size']:,.0f} accounts.")
    return facts, signals


def graph_numbers(graph_row) -> dict:
    if graph_row is None:
        return {"graph_available": False}
    values = {column: float(graph_row[column]) for column in GRAPH_FEATURE_COLUMNS}
    values["graph_available"] = True
    return values


def build_case(index: TransactionIndex, case: pd.Series, graph_row, neighbors_per_hop: int) -> dict:
    row = int(case["transaction_id"])
    sender_code, receiver_code = int(index.sender[row]), int(index.receiver[row])
    assert index.account_index[sender_code] == case["sender_account_id"] and index.account_index[receiver_code] == case["receiver_account_id"]
    case_minute = int(index.minutes[row])
    boundary_row = int(np.searchsorted(index.minutes, case_minute, side="left"))   # first row of the flagged minute
    seeds = [sender_code] if sender_code == receiver_code else [sender_code, receiver_code]

    hop1_all = index.rows_touching(seeds, boundary_row)
    hop1_kept, hop1_dropped = index.rank_and_cut(hop1_all, neighbors_per_hop)
    neighbor_codes = [c for c in pd.unique(np.concatenate([index.sender[hop1_kept], index.receiver[hop1_kept]])) if c not in seeds]
    hop2_all = np.setdiff1d(index.rows_touching(neighbor_codes, boundary_row), hop1_all)
    hop2_kept, hop2_dropped = index.rank_and_cut(hop2_all, neighbors_per_hop)

    aliases = {sender_code: "A", receiver_code: "B"}
    counters = {1: 0, 2: 0}

    def alias(code: int, hop: int) -> str:
        code = int(code)
        if code not in aliases:
            counters[hop] += 1
            aliases[code] = f"H{hop}-{counters[hop]:02d}"
        return aliases[code]

    def edge_line(edge_number: int, edge_row: int, hop: int) -> str:
        hours_before = (case_minute - index.minutes[edge_row]) / 60.0
        return (f"E{edge_number:02d} t-{hours_before:.1f}h {alias(index.sender[edge_row], hop)} -> {alias(index.receiver[edge_row], hop)} "
                f"{index.format_names[index.format_codes[edge_row]]} {index.amount_paid[edge_row]:,.2f} {index.currency_names[index.currency_codes[edge_row]]}")

    for edge_row in hop1_kept:                       # fix aliases of hop-1 neighbours first
        alias(index.sender[edge_row], 1); alias(index.receiver[edge_row], 1)
    edge_number = 0
    hop1_lines, hop2_lines = [], []
    for edge_row in hop1_kept:
        edge_number += 1; hop1_lines.append(edge_line(edge_number, int(edge_row), 1))
    for edge_row in hop2_kept:
        edge_number += 1; hop2_lines.append(edge_line(edge_number, int(edge_row), 2))

    numbers = fact_numbers(index, sender_code, receiver_code, boundary_row, case_minute, row)
    numbers.update(graph_numbers(graph_row))
    facts, signals = build_facts(index, case, sender_code, receiver_code, row, numbers)
    timestamp = pd.Timestamp(case["timestamp"]).strftime("%Y-%m-%d %H:%M")
    currency_name = index.currency_names[index.currency_codes[row]]
    lines = [f"CASE {case['case_id']}",
             f"FLAGGED PAYMENT at {timestamp}: A -> B, {index.format_names[index.format_codes[row]]}, {index.amount_paid[row]:,.2f} {currency_name} (about {index.amount_usd[row]:,.0f} USD)",
             'Legend: A = sender; B = receiver; H1-xx = traded directly with A or B; H2-xx = traded with an H1 account. In edge lines, "t-Nh" means N hours before the flagged payment. USD values are approximate.',
             "FACTS (rules applied to earlier data only; the graph numbers come from a snapshot built from earlier days):"]
    lines += [f"F{number} {text}" for number, text in enumerate(facts, start=1)]
    lines.append("RISK SIGNALS (rules that fired): " + (", ".join(f"F{number}" for number in signals) if signals else "none"))
    lines.append(f"HOP-1 TRANSACTIONS (largest {len(hop1_kept)} of {len(hop1_all)} shown, oldest first):")
    lines += hop1_lines
    if len(hop1_dropped):
        lines.append(f"+{len(hop1_dropped)} other hop-1 transactions not shown (about {index.amount_usd[hop1_dropped].sum():,.0f} USD in total)")
    lines.append(f"HOP-2 TRANSACTIONS (largest {len(hop2_kept)} of {len(hop2_all)} shown, oldest first):")
    lines += hop2_lines
    if len(hop2_dropped):
        lines.append(f"+{len(hop2_dropped)} other hop-2 transactions not shown (about {index.amount_usd[hop2_dropped].sum():,.0f} USD in total)")
    text = "\n".join(lines)
    legend = {alias_name: str(index.account_index[code]) for code, alias_name in aliases.items()}
    return {"case_id": case["case_id"], "transaction_id": row, "text": text, "estimated_tokens": int(len(text) / CHARACTERS_PER_TOKEN),
            "neighbors_per_hop": neighbors_per_hop, "hop1_total": int(len(hop1_all)), "hop1_shown": int(len(hop1_kept)),
            "hop2_total": int(len(hop2_all)), "hop2_shown": int(len(hop2_kept)),
            "trimmed": bool(len(hop1_dropped) or len(hop2_dropped)), "shrunk_to_fit_budget": neighbors_per_hop < START_NEIGHBORS_PER_HOP,
            "shown_rows": [int(r) for r in np.concatenate([hop1_kept, hop2_kept])], "case_minute": case_minute,
            "numbers": numbers, "facts": facts, "risk_signals": [f"F{number}" for number in signals],
            "graph_fact_labels": [f"F{number}" for number in range(numbers["first_graph_fact_number"], len(facts) + 1)], "alias_legend_for_grading_only": legend}


def load_cases_and_graph_features():
    cases = pd.read_csv(CASES_PATH)
    graph = pd.read_parquet(PROCESSED_DIR / "ibm_e2_features_test.parquet", columns=["transaction_id"] + GRAPH_FEATURE_COLUMNS,
                            filters=[("transaction_id", "in", cases["transaction_id"].tolist())]).set_index("transaction_id")
    return cases, graph


def build_all(index: TransactionIndex) -> list:
    cases, graph = load_cases_and_graph_features()
    results = []
    for _, case in cases.iterrows():
        graph_row = graph.loc[case["transaction_id"]] if case["transaction_id"] in graph.index else None
        for neighbors_per_hop in range(START_NEIGHBORS_PER_HOP, MINIMUM_NEIGHBORS_PER_HOP - 1, -1):
            built = build_case(index, case, graph_row, neighbors_per_hop)
            if built["estimated_tokens"] <= TOKEN_BUDGET:
                break
        results.append(built)
    return results


def run_build():
    index = TransactionIndex()
    results = build_all(index)
    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    summary = pd.DataFrame([{k: r[k] for k in ["case_id", "estimated_tokens", "neighbors_per_hop", "hop1_total", "hop1_shown", "hop2_total", "hop2_shown", "trimmed", "shrunk_to_fit_budget"]} for r in results])
    print(f"saved {OUTPUT_PATH}: {len(results)} cases")
    print(summary.describe().loc[["min", "50%", "max"], ["estimated_tokens", "hop1_total", "hop2_total"]].round(0).to_string())
    print(f"trimmed (something dropped): {summary['trimmed'].mean():.0%} | shrunk below {START_NEIGHBORS_PER_HOP} neighbours to fit {TOKEN_BUDGET} tokens: {summary['shrunk_to_fit_budget'].mean():.0%}")
    print(summary.head(6).to_string(index=False))
    for example in [results[0], results[3], results[6]]:
        print("\n" + "=" * 100 + f"\n(estimated {example['estimated_tokens']} tokens)\n" + example["text"])


def brute_force_numbers(index: TransactionIndex, row: int) -> dict:
    """Recompute the fact numbers with plain boolean masks over ALL rows, independent of the per-account row lists."""
    a, b, minute = int(index.sender[row]), int(index.receiver[row]), int(index.minutes[row])
    earlier = index.minutes < minute
    a_sent, a_received = earlier & (index.sender == a), earlier & (index.receiver == a)
    b_sent, b_received = earlier & (index.sender == b), earlier & (index.receiver == b)
    pair, reverse = a_sent & (index.receiver == b), a_received & (index.sender == b)
    inflow, outflow = a_received & (index.minutes >= minute - 360), a_sent & (index.minutes >= minute - 360)
    sender_partners = (set(index.receiver[a_sent].tolist()) | set(index.sender[a_received].tolist())) - {a}
    receiver_partners = (set(index.receiver[b_sent].tolist()) | set(index.sender[b_received].tolist())) - {b}
    relay_accounts = (set(index.receiver[a_sent].tolist()) & set(index.sender[b_received].tolist())) - {a, b}
    return {"sender_distinct_partners": len(sender_partners), "receiver_distinct_partners": len(receiver_partners),
            "common_partner_count": len((sender_partners & receiver_partners) - {a, b}), "relay_path_count": len(relay_accounts),
            "sender_past_sent": int(a_sent.sum()), "sender_past_received": int(a_received.sum()),
            "receiver_past_sent": int(b_sent.sum()), "receiver_past_received": int(b_received.sum()),
            "pair_count": int(pair.sum()), "reverse_count": int(reverse.sum()),
            "sender_inflow_6h_count": int(inflow.sum()), "sender_inflow_6h_usd": float(index.amount_usd[inflow].sum()),
            "sender_outflow_6h_count": int(outflow.sum()),
            "sender_distinct_receivers_24h": int(np.unique(index.receiver[a_sent & (index.minutes >= minute - 1440)]).size),
            "receiver_distinct_senders_24h": int(np.unique(index.sender[b_received & (index.minutes >= minute - 1440)]).size),
            "sender_distinct_currencies_before": int(np.unique(index.currency_codes[a_sent]).size),
            "currency_is_new_for_sender": bool(a_sent.any() and not (index.currency_codes[a_sent] == index.currency_codes[row]).any()),
            "format_is_new_for_sender": bool(a_sent.any() and not (index.format_codes[a_sent] == index.format_codes[row]).any())}


def run_test():
    index = TransactionIndex()
    results = [json.loads(line) for line in OUTPUT_PATH.read_text().splitlines()]
    _, graph_table = load_cases_and_graph_features()
    for result in results:
        row, case_minute = result["transaction_id"], result["case_minute"]
        shown = np.array(result["shown_rows"])
        assert len(shown) == 0 or index.minutes[shown].max() < case_minute, f"{result['case_id']}: an edge is not strictly earlier"
        assert row not in set(shown.tolist()), "flagged payment itself must not be shown"
        assert result["estimated_tokens"] <= TOKEN_BUDGET, f"{result['case_id']}: over token budget"
        expected = brute_force_numbers(index, row)
        for key, value in expected.items():
            got = result["numbers"][key]
            assert (abs(got - value) < 1e-6 * max(1.0, abs(value))) if isinstance(value, float) else got == value, f"{result['case_id']} {key}: {got} != {value}"
        assert "laundering" not in result["text"].lower(), "label words must not appear in the text"
        assert "GRAPH FEATURES" not in result["text"], "old G lines must be gone"
        for key in GRAPH_FEATURE_COLUMNS:       # graph numbers in the facts must equal the saved snapshot file
            expected_value = float(graph_table.loc[row, key])
            assert abs(result["numbers"][key] - expected_value) < 1e-6 * max(1.0, abs(expected_value)), f"{result['case_id']} {key}"
    print(f"TEST PASSED for {len(results)} cases: all shown edges are strictly earlier than the flagged payment, "
          f"18 fact numbers match a brute-force recount, graph numbers equal the saved snapshot file, token budget respected, no label words in the text")


if __name__ == "__main__":
    commands = {"build": run_build, "test": run_test}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        sys.exit("usage: python src/e4_subgraph_v2.py <build|test>")
