"""IBM AML E3b: GraphSAGE (2-layer mean aggregator) + classifier head on transaction (edge) labels, plain PyTorch.
Time-aware sampling: for a transaction at minute t, every hop only uses transactions with minute < t.
Neighbor messages carry amount (USD), payment format, how long ago, and direction. No labels are used as features.
Run on the Mac:  python src/ibm_graphsage.py  |  python src/ibm_graphsage.py no_graph  |  python src/ibm_graphsage.py smoke"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, best_f1_threshold, metrics_at_threshold, plot_confusion_matrix,
                          plot_pr_curves, save_json, threshold_for_minimum_precision, MINIMUM_ALARM_PRECISION)
from src.graphsage import SageLayer, GraphSageEdgeClassifier
from src.ibm_data_prep import PROCESSED_DIR, TRAIN_END, VALIDATION_END, TAIL_START, split_name
from src.ibm_features_tabular import IBM_E1_FEATURE_COLUMNS, PAYMENT_FORMATS, usd_rates
from src.ibm_train_tabular import linear_column_values

RANDOM_SEED = 42
TIME_KEY = 100_000                    # minutes < 100,000, so key = account * 100,000 + minute
FIRST_HOP, SECOND_HOP = 10, 5
BATCH_SIZE = 4096
LEGIT_SHARE_PER_EPOCH = 0.10
MAX_EPOCHS, PATIENCE = 40, 4
CONFIGS = [{"hidden_size": 64, "learning_rate": 1e-3, "dropout": 0.2},
           {"hidden_size": 32, "learning_rate": 3e-3, "dropout": 0.2}]
LABEL = "Is Laundering"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class TemporalGraph:
    def __init__(self, sender, receiver, minutes, edge_attributes, amounts, number_of_accounts):
        def sorted_table(owner, other, direction):
            order = np.lexsort((minutes, owner))
            return {"key": owner[order].astype(np.int64) * TIME_KEY + minutes[order],
                    "pointer": np.searchsorted(owner[order], np.arange(number_of_accounts + 1)),
                    "neighbor": other[order], "minute": minutes[order], "attributes": edge_attributes[order],
                    "direction": np.full(len(order), direction, np.float32),
                    "amount_prefix": np.concatenate([[0.0], np.cumsum(amounts[order])])}
        self.outgoing = sorted_table(sender, receiver, 0.0)
        self.incoming = sorted_table(receiver, sender, 1.0)

    @staticmethod
    def _range(table, accounts, times):
        return table["pointer"][accounts], np.searchsorted(table["key"], accounts.astype(np.int64) * TIME_KEY + times, side="left")

    def node_features(self, accounts, times):
        columns = []
        for table in [self.incoming, self.outgoing]:
            start, end = self._range(table, accounts, times)
            columns += [np.log1p(end - start), np.log1p(table["amount_prefix"][end] - table["amount_prefix"][start])]
        return np.stack(columns, axis=-1).astype(np.float32)

    def sample_neighbors(self, accounts, times, count):
        """Most recent `count` transactions (sent or received) with minute < query minute; half from each direction when possible."""
        parts = []
        for table in [self.incoming, self.outgoing]:
            start, end = self._range(table, accounts, times)
            positions = end[:, None] - count + np.arange(count)[None, :]
            mask = positions >= start[:, None]
            positions = np.where(mask, positions, 0)
            hours_ago = np.log1p(np.maximum(times[:, None] - table["minute"][positions], 0) / 60.0)
            attributes = np.concatenate([table["attributes"][positions], hours_ago[..., None].astype(np.float32),
                                         table["direction"][positions][..., None]], axis=-1)
            parts.append((table["neighbor"][positions], table["minute"][positions], attributes * mask[..., None], mask))
        neighbors = np.concatenate([p[0] for p in parts], axis=1); minutes = np.concatenate([p[1] for p in parts], axis=1)
        attributes = np.concatenate([p[2] for p in parts], axis=1); mask = np.concatenate([p[3] for p in parts], axis=1)
        # keep the `count` most recent of the 2*count candidates
        recency = np.where(mask, minutes, -1)
        keep = np.argsort(-recency, axis=1, kind="stable")[:, :count]
        take = lambda a: np.take_along_axis(a, keep if a.ndim == 2 else keep[..., None], axis=1)
        return take(neighbors), take(minutes), take(attributes), take(mask)


class BatchBuilder:
    def __init__(self, graph, sender, receiver, minutes, transaction_features, node_stats, edge_stats):
        self.graph, self.sender, self.receiver, self.minutes = graph, sender, receiver, minutes
        self.transaction_features = transaction_features
        self.node_mean, self.node_std = node_stats
        self.edge_mean, self.edge_std = edge_stats
        self.drop_neighbors = False

    def build(self, rows, return_minutes=False):
        times = self.minutes[rows]
        accounts = np.concatenate([self.sender[rows], self.receiver[rows]]); query_times = np.concatenate([times, times])
        first_nodes, first_minutes, first_edges, first_mask = self.graph.sample_neighbors(accounts, query_times, FIRST_HOP)
        first_times = np.repeat(query_times, FIRST_HOP)
        second_nodes, second_minutes, second_edges, second_mask = self.graph.sample_neighbors(first_nodes.ravel(), first_times, SECOND_HOP)
        node = lambda a, t: (self.graph.node_features(a, t) - self.node_mean) / self.node_std
        edge = lambda e, m: ((e - self.edge_mean) / self.edge_std) * m[..., None]
        q = len(accounts)
        batch = {"query_nodes": node(accounts, query_times),
                 "first_nodes": node(first_nodes.ravel(), first_times).reshape(q, FIRST_HOP, -1),
                 "first_edges": edge(first_edges, first_mask), "first_mask": first_mask.astype(np.float32),
                 "second_nodes": node(second_nodes.ravel(), np.repeat(first_times, SECOND_HOP)).reshape(q, FIRST_HOP, SECOND_HOP, -1),
                 "second_edges": edge(second_edges, second_mask).reshape(q, FIRST_HOP, SECOND_HOP, -1),
                 "second_mask": second_mask.reshape(q, FIRST_HOP, SECOND_HOP).astype(np.float32),
                 "transaction_features": self.transaction_features[rows]}
        if self.drop_neighbors:
            batch["first_mask"] = np.zeros_like(batch["first_mask"]); batch["second_mask"] = np.zeros_like(batch["second_mask"])
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in batch.items()}
        if return_minutes:
            return tensors, (query_times, first_minutes, first_mask, second_minutes.reshape(q, -1), second_mask.reshape(q, -1))
        return tensors


def prepare(smoke):
    columns = ["transaction_id", "timestamp", "sender_account_id", "receiver_account_id", "Amount Paid", "Payment Currency",
               "Receiving Currency", "Amount Received", "Payment Format", LABEL]
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet", columns=columns)
    if smoke:
        transactions = transactions[transactions.timestamp >= pd.Timestamp("2022-09-05")].reset_index(drop=True)
    split = split_name(transactions.timestamp).astype(str).values
    rates = usd_rates(transactions, pd.Series(split == "train"))
    amounts = (transactions["Amount Paid"] * transactions["Payment Currency"].astype(str).map(rates).fillna(1.0).values).values
    formats = np.stack([(transactions["Payment Format"].astype(str) == f).values for f in PAYMENT_FORMATS], axis=1).astype(np.float32)
    edge_attributes = np.concatenate([np.log1p(amounts)[:, None].astype(np.float32), formats], axis=1)
    codes, _ = pd.factorize(pd.concat([transactions.sender_account_id, transactions.receiver_account_id], ignore_index=True))
    sender, receiver = codes[:len(transactions)].astype(np.int64), codes[len(transactions):].astype(np.int64)
    minutes = ((transactions.timestamp - pd.Timestamp("2022-09-01")).dt.total_seconds() // 60).astype(np.int64).values
    graph = TemporalGraph(sender, receiver, minutes, edge_attributes, amounts, int(codes.max()) + 1)
    labels = transactions[LABEL].values
    is_tail = (transactions.timestamp >= TAIL_START).values
    transaction_ids = transactions.transaction_id.values
    del transactions

    # transaction features = the IBM E1 features, reshaped like the logistic regression inputs, standardized on train
    e1 = pd.concat([pd.read_parquet(PROCESSED_DIR / f"ibm_e1_features_{n}.parquet", columns=["transaction_id"] + IBM_E1_FEATURE_COLUMNS)
                    for n in ["train", "validation", "test"]]).set_index("transaction_id").loc[transaction_ids]
    pieces = [values for column in IBM_E1_FEATURE_COLUMNS for _, values in linear_column_values(e1, column)]
    transaction_features = np.stack(pieces, axis=1).astype(np.float32); del pieces, e1
    train_rows = np.where(split == "train")[0]
    mean = transaction_features[train_rows].mean(0); std = transaction_features[train_rows].std(0) + 1e-6
    transaction_features = (transaction_features - mean) / std

    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(train_rows, size=min(50_000, len(train_rows)), replace=False)
    sample_nodes = graph.node_features(np.concatenate([sender[sample], receiver[sample]]), np.concatenate([minutes[sample], minutes[sample]]))
    _, _, sample_edges, sample_mask = graph.sample_neighbors(receiver[sample], minutes[sample], FIRST_HOP)
    real_edges = sample_edges[sample_mask]
    builder = BatchBuilder(graph, sender, receiver, minutes, transaction_features,
                           (sample_nodes.mean(0), sample_nodes.std(0) + 1e-6), (real_edges.mean(0), real_edges.std(0) + 1e-6))
    rows = {n: np.where(split == n)[0] for n in ["train", "validation", "test"]}
    log(f"prepared: {len(labels):,} transactions, train {len(rows['train']):,} / validation {len(rows['validation']):,} / test {len(rows['test']):,}, "
        f"edge features {real_edges.shape[1]}, transaction features {transaction_features.shape[1]}")
    return builder, rows, labels, is_tail, transaction_features.shape[1], real_edges.shape[1]


def leakage_test(builder, test_rows):
    rows = np.random.default_rng(RANDOM_SEED).choice(test_rows, size=min(2000, len(test_rows)), replace=False)
    _, (query, first, first_mask, second, second_mask) = builder.build(rows, return_minutes=True)
    ok1 = np.all((first < query[:, None]) | (first_mask == 0)); ok2 = np.all((second < query[:, None]) | (second_mask == 0))
    log(f"leakage test on {len(rows)} test transactions: hop-1 all earlier = {ok1}, hop-2 all earlier = {ok2}")
    assert ok1 and ok2


def score(model, builder, rows):
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            out.append(torch.sigmoid(model(builder.build(rows[start:start + BATCH_SIZE]))).numpy())
    return np.concatenate(out)


def train_config(index, config, builder, rows, labels, transaction_size, edge_size, runs_dir, smoke):
    torch.manual_seed(RANDOM_SEED); rng = np.random.default_rng(RANDOM_SEED + index)
    model = GraphSageEdgeClassifier(4, edge_size, transaction_size, config["hidden_size"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    positives = rows["train"][labels[rows["train"]] == 1]; negatives = rows["train"][labels[rows["train"]] == 0]
    best = {"validation_pr_auc": -1, "epoch": 0}; stale = 0; history = []
    # early-stopping monitor: all validation laundering + a fixed 25% of validation legit rows (speed only;
    # setting choice, thresholds and all reported numbers use the FULL validation and test sets)
    monitor_rng = np.random.default_rng(RANDOM_SEED)
    validation_positive = rows["validation"][labels[rows["validation"]] == 1]
    validation_negative = rows["validation"][labels[rows["validation"]] == 0]
    monitor_rows = np.sort(np.concatenate([validation_positive, monitor_rng.choice(validation_negative, size=len(validation_negative) // 4, replace=False)]))
    for epoch in range(1, (2 if smoke else MAX_EPOCHS) + 1):
        start = time.time(); model.train()
        sampled = rng.choice(negatives, size=int(len(negatives) * LEGIT_SHARE_PER_EPOCH), replace=False)
        epoch_rows = rng.permutation(np.concatenate([positives, sampled]))
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(sampled) / len(positives)))
        total = 0.0
        for b in range(0, len(epoch_rows), BATCH_SIZE):
            batch_rows = epoch_rows[b:b + BATCH_SIZE]
            loss = loss_function(model(builder.build(batch_rows)), torch.from_numpy(labels[batch_rows].astype(np.float32)))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item() * len(batch_rows)
        validation_pr_auc = average_precision_score(labels[monitor_rows], score(model, builder, monitor_rows))
        history.append({"epoch": epoch, "train_loss": total / len(epoch_rows), "validation_pr_auc": float(validation_pr_auc), "seconds": round(time.time() - start, 1)})
        log(f"config {index} {config} epoch {epoch}: loss {history[-1]['train_loss']:.4f}, validation PR-AUC {validation_pr_auc:.4f} ({history[-1]['seconds']}s)")
        if validation_pr_auc > best["validation_pr_auc"]:
            best = {"validation_pr_auc": float(validation_pr_auc), "epoch": epoch}; stale = 0
            torch.save(model.state_dict(), runs_dir / f"graphsage_{index}.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                log("early stopping"); break
    model.load_state_dict(torch.load(runs_dir / f"graphsage_{index}.pt"))   # best epoch, scored on the FULL validation set
    full_validation_pr_auc = average_precision_score(labels[rows["validation"]], score(model, builder, rows["validation"]))
    log(f"config {index}: best epoch {best['epoch']}, full validation PR-AUC {full_validation_pr_auc:.4f}")
    entry = {"config_index": index, "parameters": config, "validation_pr_auc": float(full_validation_pr_auc),
             "monitor_validation_pr_auc": best["validation_pr_auc"], "best_epoch": best["epoch"], "history": history}
    save_json(entry, runs_dir / f"graphsage_{index}.json"); return entry


def main(mode):
    experiment = {"full": "ibm_e3", "no_graph": "ibm_e3_nograph", "smoke": "ibm_e3_smoke"}[mode]
    runs_dir = RESULTS_DIR / f"{experiment}_runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    configs = CONFIGS[:1] if mode == "smoke" else CONFIGS
    if mode == "no_graph":
        configs = [json.loads((RESULTS_DIR / "ibm_e3_metrics.json").read_text())["models"]["graphsage"]["best_parameters"]]
    builder, rows, labels, is_tail, transaction_size, edge_size = prepare(mode == "smoke")
    leakage_test(builder, rows["test"])
    builder.drop_neighbors = mode == "no_graph"
    if mode == "no_graph":
        log("ABLATION: neighbors removed (same network, same settings, no graph information)")
    tuning = []
    for index, config in enumerate(configs):
        done = runs_dir / f"graphsage_{index}.json"
        if done.exists() and mode != "smoke":
            log(f"config {index} already trained, loading"); tuning.append(json.loads(done.read_text())); continue
        tuning.append(train_config(index, config, builder, rows, labels, transaction_size, edge_size, runs_dir, mode == "smoke"))
    if mode == "smoke":
        log("SMOKE TEST OK"); return
    best = max(tuning, key=lambda e: e["validation_pr_auc"])
    model = GraphSageEdgeClassifier(4, edge_size, transaction_size, best["parameters"]["hidden_size"], best["parameters"]["dropout"])
    model.load_state_dict(torch.load(runs_dir / f"graphsage_{best['config_index']}.pt"))
    validation_scores = score(model, builder, rows["validation"]); test_scores = score(model, builder, rows["test"])
    yv, yt, tail = labels[rows["validation"]], labels[rows["test"]], is_tail[rows["test"]]
    f1_threshold = best_f1_threshold(yv, validation_scores); floor = threshold_for_minimum_precision(yv, validation_scores)
    result = {"best_parameters": best["parameters"], "best_epoch": best["best_epoch"], "tuning": tuning,
              "validation": metrics_at_threshold(yv, validation_scores, f1_threshold), "test": metrics_at_threshold(yt, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(yv, validation_scores, floor),
              "test_precision_floor": metrics_at_threshold(yt, test_scores, floor),
              "test_normal_days_precision_floor": metrics_at_threshold(yt[~tail], test_scores[~tail], floor),
              "test_tail_precision_floor": metrics_at_threshold(yt[tail], test_scores[tail], floor)}
    title = "IBM E3b GraphSAGE + MLP head" if mode == "full" else "IBM E3b ablation: same network, graph neighbors removed"
    save_json({"experiment": title, "models": {"graphsage": result}}, RESULTS_DIR / f"{experiment}_metrics.json")
    pd.DataFrame({"label": yt, "fraud_score": test_scores, "is_tail": tail}).to_parquet(RESULTS_DIR / f"{experiment}_test_scores_graphsage.parquet")
    plot_confusion_matrix(result["test_precision_floor"], f"{experiment} (test)", FIGURES_DIR / f"{experiment}_confusion_graphsage.png")
    plot_pr_curves({"graphsage": (yt, test_scores)}, float(yt.mean()), f"{experiment} test precision-recall", FIGURES_DIR / f"{experiment}_pr_curves_test.png")
    n = ~tail
    log(f"ALL DONE -> results/{experiment}_metrics.json | test PR-AUC all {result['test']['pr_auc']:.4f}, "
        f"Sept 9-10 {average_precision_score(yt[n], test_scores[n]):.4f} | half-real cutoff caught {result['test_precision_floor']['recall']:.3f} "
        f"(Sept 9-10 {result['test_normal_days_precision_floor']['recall']:.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
