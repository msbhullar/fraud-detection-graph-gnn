"""E1 tabular features for IBM AML. Past-only: account history uses transactions from strictly earlier minutes.
No laundering labels are used. Original columns are never changed."""
import numpy as np
import pandas as pd

PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
NO_HISTORY_VALUE = -1.0


def usd_rates(transactions: pd.DataFrame, train_mask: pd.Series) -> dict:
    """Rate per currency from training rows paid in that currency and received in US Dollars."""
    rows = transactions[train_mask & (transactions["Receiving Currency"].astype(str) == "US Dollar")]
    rates = (rows["Amount Received"] / rows["Amount Paid"]).groupby(rows["Payment Currency"].astype(str), observed=True).median()
    rates["US Dollar"] = 1.0
    return rates.to_dict()


class AccountHistory:
    """Sorted (account, minute) index: count and amount sum of an account's sent or received payments before minute t."""

    def __init__(self, account_codes, minutes, amounts, number_of_accounts):
        order = np.lexsort((minutes, account_codes))
        self.key = account_codes[order].astype(np.int64) * 10_000_000 + minutes[order]
        self.minute = minutes[order]
        self.pointer = np.searchsorted(account_codes[order], np.arange(number_of_accounts + 1))
        self.amount_prefix = np.concatenate([[0.0], np.cumsum(amounts[order])])

    def before(self, account_codes, minutes):
        start = self.pointer[account_codes]
        end = np.searchsorted(self.key, account_codes.astype(np.int64) * 10_000_000 + minutes, side="left")
        count = end - start
        amount = self.amount_prefix[end] - self.amount_prefix[start]
        last_minute = np.where(count > 0, self.minute[np.maximum(end - 1, 0)], -1)
        return count, amount, last_minute


def add_ibm_tabular_features(transactions: pd.DataFrame, train_mask: pd.Series) -> tuple:
    features = pd.DataFrame(index=transactions.index)
    for payment_format in PAYMENT_FORMATS:
        features[f"payment_format_{payment_format.lower().replace(' ', '_')}"] = (transactions["Payment Format"] == payment_format).astype(np.int8)
    rates = usd_rates(transactions, train_mask)
    amount_usd = transactions["Amount Paid"] * transactions["Payment Currency"].astype(str).map(rates).fillna(1.0).values
    features["log_amount_usd"] = np.log1p(amount_usd)
    features["currency_changed"] = (transactions["Payment Currency"].astype(str) != transactions["Receiving Currency"].astype(str)).astype(np.int8)
    features["is_self_transfer"] = (transactions["sender_account_id"] == transactions["receiver_account_id"]).astype(np.int8)
    features["is_same_bank"] = (transactions["From Bank"] == transactions["To Bank"]).astype(np.int8)
    features["hour_of_day"] = transactions["timestamp"].dt.hour.astype(np.int8)
    features["is_weekend"] = (transactions["timestamp"].dt.dayofweek >= 5).astype(np.int8)

    codes, _ = pd.factorize(pd.concat([transactions["sender_account_id"], transactions["receiver_account_id"]], ignore_index=True))
    sender_code, receiver_code = codes[:len(transactions)], codes[len(transactions):]
    minutes = ((transactions["timestamp"] - transactions["timestamp"].min()).dt.total_seconds() // 60).astype(np.int64).values
    number_of_accounts = int(codes.max()) + 1
    sent = AccountHistory(sender_code, minutes, amount_usd.values, number_of_accounts)
    received = AccountHistory(receiver_code, minutes, amount_usd.values, number_of_accounts)

    for role, code in [("sender", sender_code), ("receiver", receiver_code)]:
        sent_count, sent_amount, last_sent = sent.before(code, minutes)
        received_count, received_amount, last_received = received.before(code, minutes)
        features[f"{role}_prior_sent_count"] = sent_count
        features[f"{role}_prior_sent_amount_usd"] = sent_amount
        features[f"{role}_prior_received_count"] = received_count
        features[f"{role}_prior_received_amount_usd"] = received_amount
        last_activity = np.maximum(last_sent, last_received)
        features[f"{role}_minutes_since_last_activity"] = np.where(last_activity >= 0, minutes - last_activity, NO_HISTORY_VALUE)
    sender_usual = features["sender_prior_sent_amount_usd"] / features["sender_prior_sent_count"].replace(0, np.nan)
    receiver_usual = features["receiver_prior_received_amount_usd"] / features["receiver_prior_received_count"].replace(0, np.nan)
    features["amount_vs_sender_usual_sent"] = (amount_usd / (sender_usual + 1.0)).fillna(NO_HISTORY_VALUE)
    features["amount_vs_receiver_usual_received"] = (amount_usd / (receiver_usual + 1.0)).fillna(NO_HISTORY_VALUE)
    return features, rates


IBM_E1_FEATURE_COLUMNS = (
    [f"payment_format_{f.lower().replace(' ', '_')}" for f in PAYMENT_FORMATS]
    + ["log_amount_usd", "currency_changed", "is_self_transfer", "is_same_bank", "hour_of_day", "is_weekend"]
    + [f"{role}_{name}" for role in ["sender", "receiver"] for name in
       ["prior_sent_count", "prior_sent_amount_usd", "prior_received_count", "prior_received_amount_usd", "minutes_since_last_activity"]]
    + ["amount_vs_sender_usual_sent", "amount_vs_receiver_usual_received"])
