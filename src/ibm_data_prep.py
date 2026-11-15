"""IBM AML HI-Small: load, add clear account ids and time columns, save parquet.
Original columns are kept as pandas reads them (the second 'Account' column becomes 'Account.1')."""
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/ibm_aml")
PROCESSED_DIR = Path("data/processed/ibm_aml")
LABEL_COLUMN = "Is Laundering"


def load_ibm_transactions() -> pd.DataFrame:
    transactions = pd.read_csv(RAW_DIR / "HI-Small_Trans.csv", engine="pyarrow",
                               dtype={"From Bank": str, "Account": str, "To Bank": str, "Account.1": str})
    for column in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        transactions[column] = transactions[column].astype("category")
    transactions["timestamp"] = pd.to_datetime(transactions["Timestamp"], format="%Y/%m/%d %H:%M")
    transactions["sender_account_id"] = transactions["From Bank"] + "_" + transactions["Account"]
    transactions["receiver_account_id"] = transactions["To Bank"] + "_" + transactions["Account.1"]
    transactions = transactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    transactions["transaction_id"] = transactions.index
    return transactions


def load_patterns() -> pd.DataFrame:
    """Each laundering attempt block -> pattern name for its transactions (sender, receiver, timestamp, amount)."""
    rows, pattern = [], None
    for line in (RAW_DIR / "HI-Small_Patterns.txt").read_text().splitlines():
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            pattern = re.sub(r"BEGIN LAUNDERING ATTEMPT - ", "", line).split(":")[0].strip()
        elif line.startswith("END LAUNDERING ATTEMPT") or not line.strip():
            continue
        else:
            parts = line.split(",")
            rows.append({"pattern": pattern, "timestamp": pd.to_datetime(parts[0], format="%Y/%m/%d %H:%M"),
                         "sender_account_id": parts[1] + "_" + parts[2], "receiver_account_id": parts[3] + "_" + parts[4],
                         "Amount Paid": float(parts[7])})
    return pd.DataFrame(rows)


# Temporal split by whole days, full data kept (decided 2026-10-25, before any IBM model was trained).
TRAIN_END = pd.Timestamp("2022-09-07")       # train: Sept 1-6
VALIDATION_END = pd.Timestamp("2022-09-09")  # validation: Sept 7-8; test: Sept 9-18 (includes the simulator tail Sept 11-18)
TAIL_START = pd.Timestamp("2022-09-11")      # reported separately inside test, never used for decisions


def split_name(timestamps: pd.Series) -> pd.Series:
    return pd.Series(pd.Categorical.from_codes(
        (timestamps >= TRAIN_END).astype(int) + (timestamps >= VALIDATION_END).astype(int),
        categories=["train", "validation", "test"]), index=timestamps.index)
