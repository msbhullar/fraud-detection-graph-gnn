"""E4 step 1 feasibility check (read-only): how many of the 50 E4 cases can be matched to a laundering
pattern listed in IBM's HI-Small_Patterns.txt?

Match key: timestamp + sender account id + receiver account id + amount paid (rounded to 2 decimals).
Writes results/e4/pattern_match_cases.csv. Does not change any existing file or any model input.
Run from the project root:  python src/e4_pattern_check.py
"""
import re
from pathlib import Path

import pandas as pd

PATTERNS_FILE = Path("data/raw/ibm_aml/HI-Small_Patterns.txt")
CASES_FILE = Path("results/e4/cases.csv")
OUTPUT_FILE = Path("results/e4/pattern_match_cases.csv")
NORMAL_DAYS_END = pd.Timestamp("2022-09-11")  # Sept 9-10 = normal days; Sept 11-18 = simulator tail


def load_pattern_rows_with_attempt_ids() -> pd.DataFrame:
    """One row per transaction listed inside a laundering attempt, with the attempt number and pattern name."""
    rows, pattern_name, attempt_number = [], None, 0
    for line in PATTERNS_FILE.read_text().splitlines():
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt_number += 1
            pattern_name = re.sub(r"BEGIN LAUNDERING ATTEMPT - ", "", line).split(":")[0].strip()
        elif line.startswith("END LAUNDERING ATTEMPT") or not line.strip():
            continue
        else:
            parts = line.split(",")
            rows.append({
                "pattern_attempt_number": attempt_number,
                "pattern_name": pattern_name,
                "timestamp": pd.to_datetime(parts[0], format="%Y/%m/%d %H:%M"),
                "sender_account_id": parts[1] + "_" + parts[2],
                "receiver_account_id": parts[3] + "_" + parts[4],
                "amount_paid": round(float(parts[7]), 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    pattern_rows = load_pattern_rows_with_attempt_ids()
    print("pattern rows shape:", pattern_rows.shape)
    print("attempts:", pattern_rows["pattern_attempt_number"].nunique())
    print(pattern_rows.head(3).to_string())

    # How much of the Sept 9-10 laundering does the pattern file cover? (956 laundering rows on those days.)
    normal_day_pattern_rows = pattern_rows[pattern_rows["timestamp"].between("2022-09-09", NORMAL_DAYS_END, inclusive="left")]
    print("\npattern rows on Sept 9-10:", len(normal_day_pattern_rows), "(laundering rows on those days: 956)")

    cases = pd.read_csv(CASES_FILE, parse_dates=["timestamp"])
    cases["amount_paid"] = cases["amount_paid"].round(2)
    match_key = ["timestamp", "sender_account_id", "receiver_account_id", "amount_paid"]
    matched = cases.merge(pattern_rows, on=match_key, how="left")

    # A transaction could appear in two attempts; keep every match but count cases once.
    matches_per_case = matched.groupby("case_id")["pattern_name"].apply(lambda names: names.dropna().nunique())
    print("\ncases with more than one different pattern name:", int((matches_per_case > 1).sum()))
    cases_with_pattern = matched.drop_duplicates("case_id").assign(
        in_pattern_file=lambda frame: frame["pattern_name"].notna())

    true_laundering = cases_with_pattern[cases_with_pattern["true_label_for_grading_only"] == 1]
    false_alarms = cases_with_pattern[cases_with_pattern["true_label_for_grading_only"] == 0]
    print("\nTRUE laundering cases:", len(true_laundering), "-> found in pattern file:", int(true_laundering["in_pattern_file"].sum()))
    print("FALSE alarm cases:", len(false_alarms), "-> found in pattern file (should be 0):", int(false_alarms["in_pattern_file"].sum()))
    print("\npattern name counts among true laundering cases:")
    print(true_laundering["pattern_name"].fillna("NOT IN PATTERN FILE").value_counts().to_string())

    cases_with_pattern.to_csv(OUTPUT_FILE, index=False)
    print("\nsaved:", OUTPUT_FILE)
    print(cases_with_pattern[["case_id", "true_label_for_grading_only", "pattern_name", "pattern_attempt_number"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
