"""E1 tabular features. Past-only: a row only sees transactions from strictly earlier steps.
No fraud labels are used. Original PaySim columns are never changed."""
import numpy as np
import pandas as pd

MODEL_FEATURE_COLUMNS = [
    "is_transfer",
    "amount",
    "hour_of_day",
    "receiver_prior_transaction_count",
    "receiver_prior_amount_sum",
    "receiver_hours_since_last_payment",
    "amount_vs_receiver_prior_average",
    "sender_prior_transaction_count",
]
NO_HISTORY_VALUE = -1.0


def _prior_history_per_account(transactions: pd.DataFrame, account_column: str) -> pd.DataFrame:
    """For each (account, step): count and amount sum from strictly earlier steps, and the last earlier step."""
    per_account_step = (transactions.groupby([account_column, "step"], sort=True)["amount"]
                        .agg(step_count="size", step_amount_sum="sum").reset_index())
    grouped = per_account_step.groupby(account_column, sort=False)
    per_account_step["prior_count"] = grouped["step_count"].cumsum() - per_account_step["step_count"]
    per_account_step["prior_amount_sum"] = grouped["step_amount_sum"].cumsum() - per_account_step["step_amount_sum"]
    per_account_step["previous_step"] = grouped["step"].shift(1)
    return per_account_step[[account_column, "step", "prior_count", "prior_amount_sum", "previous_step"]]


def add_tabular_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """transactions must contain ALL rows (train+validation+test) so history runs across split boundaries,
    always looking backward in time only."""
    with_features = transactions.copy()
    with_features["is_transfer"] = (with_features["type"] == "TRANSFER").astype(int)
    with_features["hour_of_day"] = with_features["step"] % 24

    receiver_history = _prior_history_per_account(with_features, "nameDest").rename(columns={
        "prior_count": "receiver_prior_transaction_count",
        "prior_amount_sum": "receiver_prior_amount_sum",
        "previous_step": "receiver_previous_step"})
    with_features = with_features.merge(receiver_history, on=["nameDest", "step"], how="left")

    sender_history = _prior_history_per_account(with_features, "nameOrig")[["nameOrig", "step", "prior_count"]]
    sender_history = sender_history.rename(columns={"prior_count": "sender_prior_transaction_count"})
    with_features = with_features.merge(sender_history, on=["nameOrig", "step"], how="left")

    has_receiver_history = with_features["receiver_prior_transaction_count"] > 0
    with_features["receiver_hours_since_last_payment"] = np.where(
        has_receiver_history, with_features["step"] - with_features["receiver_previous_step"], NO_HISTORY_VALUE)
    receiver_prior_average = with_features["receiver_prior_amount_sum"] / with_features["receiver_prior_transaction_count"].where(has_receiver_history)
    with_features["amount_vs_receiver_prior_average"] = np.where(
        has_receiver_history, with_features["amount"] / (receiver_prior_average + 1.0), NO_HISTORY_VALUE)
    return with_features.drop(columns=["receiver_previous_step"])
