"""Phase 1 data preparation: filter, subsample, temporal split."""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
RAW_CSV_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
PROCESSED_DIR = Path("data/processed")
FRAUD_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT"]


def load_raw_transactions(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the original PaySim CSV with all original columns unchanged."""
    return pd.read_csv(csv_path)


def filter_to_fraud_types(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep only TRANSFER and CASH_OUT, the only types that contain fraud.
    All columns are kept (balance columns are needed later for the naive run)."""
    is_fraud_type = raw_transactions["type"].isin(FRAUD_TRANSACTION_TYPES)
    return raw_transactions.loc[is_fraud_type].reset_index(drop=True)


# Temporal split (split A): 60/20/20 of the 743 steps. Never shuffled.
TRAIN_LAST_STEP = 446
VALIDATION_LAST_STEP = 595


def temporal_split(transactions: pd.DataFrame):
    """Split by step: earliest steps for training, next block for validation, last block for test."""
    train_rows = transactions[transactions["step"] <= TRAIN_LAST_STEP]
    validation_rows = transactions[(transactions["step"] > TRAIN_LAST_STEP) & (transactions["step"] <= VALIDATION_LAST_STEP)]
    test_rows = transactions[transactions["step"] > VALIDATION_LAST_STEP]
    return (train_rows.reset_index(drop=True),
            validation_rows.reset_index(drop=True),
            test_rows.reset_index(drop=True))
