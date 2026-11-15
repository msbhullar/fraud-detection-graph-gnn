"""E4 Option 2 feasibility (read-only): for each E4 case that belongs to an IBM laundering pattern attempt, how many OTHER members of the
same attempt are visible in the case text the LLM sees? (Past-only, top-10 edges per hop.) If the rest of the pattern is trimmed away or
happens later, no model could name the pattern from that text.
Uses the attempt number from results/e4/pattern_match_cases.csv and the shown transaction rows stored in results/e4/subgraph_cases_v2.jsonl.
Writes results/e4/pattern_visibility.csv. Run from the project root:  python src/e4_pattern_visibility.py"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.e4_pattern_check import load_pattern_rows_with_attempt_ids
from src.ibm_data_prep import PROCESSED_DIR

# Default = original 50 cases. For the typology set:
#   python src/e4_pattern_visibility.py results/e4/cases_typology.csv results/e4/subgraph_cases_typology_v2.jsonl results/e4/pattern_visibility_typology.csv
CASES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/e4/pattern_match_cases.csv")
SUBGRAPH_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/e4/subgraph_cases_v2.jsonl")
OUTPUT_FILE = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/e4/pattern_visibility.csv")


def main() -> None:
    pattern_rows = load_pattern_rows_with_attempt_ids()
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet",
                                   columns=["timestamp", "sender_account_id", "receiver_account_id", "Amount Paid"])
    transactions["transaction_id"] = transactions.index                  # row position = transaction_id (same convention as the E4 scripts)
    transactions["amount_paid"] = transactions["Amount Paid"].round(2)
    keys = ["timestamp", "sender_account_id", "receiver_account_id", "amount_paid"]
    members = pattern_rows.merge(transactions[keys + ["transaction_id"]], on=keys, how="left")
    print("pattern rows:", len(pattern_rows), "| matched to a transaction_id:", int(members["transaction_id"].notna().sum()))
    del transactions

    cases = pd.read_csv(CASES_FILE, parse_dates=["timestamp"])
    cases = cases[cases["pattern_name"].notna()]
    subgraphs = {record["case_id"]: record for record in map(json.loads, SUBGRAPH_FILE.read_text().splitlines())}
    rows = []
    for _, case in cases.iterrows():
        attempt = members[members["pattern_attempt_number"] == case["pattern_attempt_number"]]
        flagged_id = int(case["transaction_id"])
        earlier = attempt[attempt["timestamp"] < case["timestamp"]]
        later_or_same = attempt[attempt["timestamp"] >= case["timestamp"]]
        touching_flagged_accounts = earlier[(earlier["sender_account_id"].isin([case["sender_account_id"], case["receiver_account_id"]]))
                                            | (earlier["receiver_account_id"].isin([case["sender_account_id"], case["receiver_account_id"]]))]
        shown = set(subgraphs[case["case_id"]]["shown_rows"])
        rows.append({"case_id": case["case_id"], "pattern_name": case["pattern_name"], "attempt": int(case["pattern_attempt_number"]),
                     "attempt_size": len(attempt), "members_earlier": len(earlier), "members_same_time_or_later": len(later_or_same) - 1,
                     "earlier_touching_A_or_B": len(touching_flagged_accounts),
                     "earlier_visible_in_text": int(earlier["transaction_id"].isin(shown).sum()),
                     "hop1_total": subgraphs[case["case_id"]]["hop1_total"], "hop1_shown": subgraphs[case["case_id"]]["hop1_shown"]})
    table = pd.DataFrame(rows)
    table["any_pattern_edge_visible"] = table["earlier_visible_in_text"] > 0
    table.to_csv(OUTPUT_FILE, index=False)
    print("\nshape:", table.shape)
    print(table.head(10).to_string(index=False))
    print("\nBy pattern (cases, median attempt size, cases with >=1 earlier member, with >=1 visible in the text, with >=2 visible):")
    summary = table.groupby("pattern_name").agg(cases=("case_id", "count"), median_attempt_size=("attempt_size", "median"),
                                                 has_earlier=("members_earlier", lambda v: int((v > 0).sum())),
                                                 touching_A_or_B=("earlier_touching_A_or_B", lambda v: int((v > 0).sum())),
                                                 visible_ge1=("earlier_visible_in_text", lambda v: int((v >= 1).sum())),
                                                 visible_ge2=("earlier_visible_in_text", lambda v: int((v >= 2).sum())))
    print(summary.to_string())
    print(f"\nALL {len(table)} matched cases: earlier member exists in {int((table['members_earlier'] > 0).sum())}; "
          f">=1 visible in text in {int((table['earlier_visible_in_text'] >= 1).sum())}; >=2 visible in {int((table['earlier_visible_in_text'] >= 2).sum())}")


if __name__ == "__main__":
    main()
