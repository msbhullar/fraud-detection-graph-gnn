"""E3b: GraphSAGE (mean aggregator, 2 layers) + classifier head, trained end to end on transaction (edge) labels.
Plain PyTorch with our own time-aware neighbor sampler: for a transaction at step t, every hop only uses
transactions with step < t (strictly earlier hours). No fraud labels are used as features.
Run on the Mac:  python src/graphsage.py            (full run, resumable per config)
                 python src/graphsage.py smoke      (tiny quick test)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import PROCESSED_DIR, TRAIN_LAST_STEP, VALIDATION_LAST_STEP
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, best_f1_threshold, metrics_at_threshold, plot_confusion_matrix,
                          plot_pr_curves, save_json, threshold_for_minimum_precision, MINIMUM_ALARM_PRECISION)
from src.features_tabular import MODEL_FEATURE_COLUMNS, add_tabular_features
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
STEP_KEY = 1000                       # step < 1000, so key = node * 1000 + step sorts by node, then time
FIRST_HOP_NEIGHBORS = 10              # most recent earlier transactions per account
SECOND_HOP_NEIGHBORS = 5
BATCH_SIZE = 4096
LEGIT_SHARE_PER_EPOCH = 0.10          # each epoch: all training fraud + a fresh random 10% of training legit rows
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 4
CONFIGS = [{"hidden_size": 64, "learning_rate": 1e-3, "dropout": 0.2},
           {"hidden_size": 32, "learning_rate": 3e-3, "dropout": 0.2}]
RUNS_DIR = RESULTS_DIR / "e3_runs"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class TemporalGraph:
    """Account-level adjacency sorted by (account, step), so 'all edges of account a before step t'
    is one binary search. Built from ALL transactions, but every lookup is cut at the query time."""

    def __init__(self, sender_index, receiver_index, steps, log_amounts, is_transfer, number_of_accounts):
        def build(owner, other, direction):
            order = np.lexsort((steps, owner))
            return {"key": owner[order].astype(np.int64) * STEP_KEY + steps[order],
                    "pointer": np.searchsorted(owner[order], np.arange(number_of_accounts + 1)),
                    "neighbor": other[order], "step": steps[order], "log_amount": log_amounts[order],
                    "is_transfer": is_transfer[order], "direction": np.full(len(order), direction, dtype=np.float32),
                    "amount_prefix": np.concatenate([[0.0], np.cumsum(np.expm1(log_amounts[order]))])}
        self.outgoing = build(sender_index, receiver_index, 0.0)
        self.incoming = build(receiver_index, sender_index, 1.0)
        both_owner = np.concatenate([sender_index, receiver_index])
        both_other = np.concatenate([receiver_index, sender_index])
        both = {"steps": np.concatenate([steps, steps]), "log_amount": np.concatenate([log_amounts, log_amounts]),
                "is_transfer": np.concatenate([is_transfer, is_transfer]),
                "direction": np.concatenate([np.zeros(len(steps), np.float32), np.ones(len(steps), np.float32)])}
        order = np.lexsort((both["steps"], both_owner))
        self.both = {"key": both_owner[order].astype(np.int64) * STEP_KEY + both["steps"][order],
                     "pointer": np.searchsorted(both_owner[order], np.arange(number_of_accounts + 1)),
                     "neighbor": both_other[order], "step": both["steps"][order],
                     "log_amount": both["log_amount"][order], "is_transfer": both["is_transfer"][order],
                     "direction": both["direction"][order]}

    @staticmethod
    def _past_range(table, accounts, times):
        start = table["pointer"][accounts]
        end = np.searchsorted(table["key"], accounts.astype(np.int64) * STEP_KEY + times, side="left")  # first edge with step >= t
        return start, end

    def node_features(self, accounts, times):
        """Past-only activity of each account: log counts and log sums of money in and out."""
        features = []
        for table in [self.incoming, self.outgoing]:
            start, end = self._past_range(table, accounts, times)
            features.append(np.log1p(end - start))
            features.append(np.log1p(table["amount_prefix"][end] - table["amount_prefix"][start]))
        return np.stack(features, axis=-1).astype(np.float32)

    def sample_neighbors(self, accounts, times, count):
        """The `count` most recent transactions of each account with step < its query time."""
        start, end = self._past_range(self.both, accounts, times)
        positions = end[:, None] - count + np.arange(count)[None, :]
        mask = positions >= start[:, None]
        positions = np.where(mask, positions, 0)
        table = self.both
        hours_ago = np.log1p(np.maximum(times[:, None] - table["step"][positions], 0))
        edge_features = np.stack([table["log_amount"][positions], table["is_transfer"][positions],
                                  hours_ago, table["direction"][positions]], axis=-1).astype(np.float32)
        return table["neighbor"][positions], table["step"][positions], edge_features * mask[..., None], mask


class SageLayer(nn.Module):
    def __init__(self, input_size, edge_size, output_size, dropout):
        super().__init__()
        self.self_linear = nn.Linear(input_size, output_size)
        self.message_linear = nn.Linear(input_size + edge_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, own, neighbors, edges, mask):
        messages = self.message_linear(torch.cat([neighbors, edges], dim=-1)) * mask.unsqueeze(-1)
        mean_message = messages.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.dropout(torch.relu(self.self_linear(own) + mean_message))


class GraphSageEdgeClassifier(nn.Module):
    def __init__(self, node_size, edge_size, transaction_size, hidden_size, dropout):
        super().__init__()
        self.layer_one = SageLayer(node_size, edge_size, hidden_size, dropout)
        self.layer_two = SageLayer(hidden_size, edge_size, hidden_size, dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden_size + transaction_size, hidden_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, batch):
        query_count, first_hop = batch["first_mask"].shape
        second_hop = batch["second_mask"].shape[-1]
        hidden_query = self.layer_one(batch["query_nodes"], batch["first_nodes"], batch["first_edges"], batch["first_mask"])
        hidden_first = self.layer_one(batch["first_nodes"].reshape(query_count * first_hop, -1),
                                      batch["second_nodes"].reshape(query_count * first_hop, second_hop, -1),
                                      batch["second_edges"].reshape(query_count * first_hop, second_hop, -1),
                                      batch["second_mask"].reshape(query_count * first_hop, second_hop)
                                      ).reshape(query_count, first_hop, -1)
        embedding = self.layer_two(hidden_query, hidden_first, batch["first_edges"], batch["first_mask"])
        sender_embedding, receiver_embedding = embedding.chunk(2, dim=0)
        return self.head(torch.cat([sender_embedding, receiver_embedding, batch["transaction_features"]], dim=1)).squeeze(1)


class BatchBuilder:
    def __init__(self, graph, sender_index, receiver_index, steps, transaction_features, node_mean, node_std, edge_mean, edge_std):
        self.graph, self.sender_index, self.receiver_index, self.steps = graph, sender_index, receiver_index, steps
        self.transaction_features = transaction_features
        self.node_mean, self.node_std, self.edge_mean, self.edge_std = node_mean, node_std, edge_mean, edge_std
        self.drop_neighbors = False  # ablation: same network, no graph neighbors

    def build(self, rows, return_steps=False):
        times = self.steps[rows]
        query_accounts = np.concatenate([self.sender_index[rows], self.receiver_index[rows]])
        query_times = np.concatenate([times, times])
        first_accounts, first_steps, first_edges, first_mask = self.graph.sample_neighbors(query_accounts, query_times, FIRST_HOP_NEIGHBORS)
        first_times = np.repeat(query_times, FIRST_HOP_NEIGHBORS)
        second_accounts, second_steps, second_edges, second_mask = self.graph.sample_neighbors(first_accounts.ravel(), first_times, SECOND_HOP_NEIGHBORS)
        normalize_node = lambda f: (f - self.node_mean) / self.node_std
        normalize_edge = lambda e, m: ((e - self.edge_mean) / self.edge_std) * m[..., None]
        batch = {
            "query_nodes": normalize_node(self.graph.node_features(query_accounts, query_times)),
            "first_nodes": normalize_node(self.graph.node_features(first_accounts.ravel(), first_times)).reshape(len(query_accounts), FIRST_HOP_NEIGHBORS, -1),
            "first_edges": normalize_edge(first_edges, first_mask), "first_mask": first_mask.astype(np.float32),
            "second_nodes": normalize_node(self.graph.node_features(second_accounts.ravel(), np.repeat(first_times, SECOND_HOP_NEIGHBORS))
                                           ).reshape(len(query_accounts), FIRST_HOP_NEIGHBORS, SECOND_HOP_NEIGHBORS, -1),
            "second_edges": normalize_edge(second_edges, second_mask).reshape(len(query_accounts), FIRST_HOP_NEIGHBORS, SECOND_HOP_NEIGHBORS, -1),
            "second_mask": second_mask.reshape(len(query_accounts), FIRST_HOP_NEIGHBORS, SECOND_HOP_NEIGHBORS).astype(np.float32),
            "transaction_features": self.transaction_features[rows],
        }
        if self.drop_neighbors:
            batch["first_mask"] = np.zeros_like(batch["first_mask"]); batch["second_mask"] = np.zeros_like(batch["second_mask"])
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in batch.items()}
        if return_steps:
            return tensors, (query_times, first_steps, first_mask, second_steps.reshape(len(query_accounts), -1), second_mask.reshape(len(query_accounts), -1))
        return tensors


def leakage_test(builder, test_rows, sample_size=2000):
    """Brief requirement: every sampled neighbor edge must have step < the target transaction's step."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = rng.choice(test_rows, size=min(sample_size, len(test_rows)), replace=False)
    _, (query_times, first_steps, first_mask, second_steps, second_mask) = builder.build(rows, return_steps=True)
    first_ok = np.all((first_steps < query_times[:, None]) | (first_mask == 0))
    second_ok = np.all((second_steps < query_times[:, None]) | (second_mask == 0))
    log(f"leakage test on {len(rows)} test transactions: hop-1 neighbors all earlier = {first_ok}, hop-2 neighbors all earlier = {second_ok}")
    assert first_ok and second_ok, "a sampled neighbor is not strictly in the past"


def score(model, builder, rows):
    model.eval(); outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            outputs.append(torch.sigmoid(model(builder.build(rows[start:start + BATCH_SIZE]))).numpy())
    return np.concatenate(outputs)


def prepare(smoke: bool):
    from src.train_tabular import linear_model_inputs  # same reshaped E1 inputs as the logistic regression
    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet")
    if smoke:
        keep = np.concatenate([np.arange(0, 600_000), np.where(transactions.step > TRAIN_LAST_STEP)[0]])
        transactions = transactions.iloc[keep].reset_index(drop=True)
    with_features = add_tabular_features(transactions)
    account_codes, _ = pd.factorize(pd.concat([with_features.nameOrig, with_features.nameDest]))
    sender_index = account_codes[:len(with_features)].astype(np.int64)
    receiver_index = account_codes[len(with_features):].astype(np.int64)
    steps = with_features.step.values.astype(np.int64)
    log_amounts = np.log1p(with_features.amount.values).astype(np.float32)
    is_transfer = with_features.is_transfer.values.astype(np.float32)
    graph = TemporalGraph(sender_index, receiver_index, steps, log_amounts, is_transfer, int(account_codes.max()) + 1)

    split = np.where(steps <= TRAIN_LAST_STEP, "train", np.where(steps <= VALIDATION_LAST_STEP, "validation", "test"))
    rows = {name: np.where(split == name)[0] for name in ["train", "validation", "test"]}
    linear_inputs = linear_model_inputs(with_features)
    scaler = StandardScaler().fit(linear_inputs.iloc[rows["train"]])          # fit on train only
    transaction_features = scaler.transform(linear_inputs).astype(np.float32)

    rng = np.random.default_rng(RANDOM_SEED)
    sample_rows = rng.choice(rows["train"], size=min(50_000, len(rows["train"])), replace=False)
    sample_nodes = graph.node_features(np.concatenate([sender_index[sample_rows], receiver_index[sample_rows]]),
                                       np.concatenate([steps[sample_rows], steps[sample_rows]]))
    _, _, sample_edges, sample_mask = graph.sample_neighbors(receiver_index[sample_rows], steps[sample_rows], FIRST_HOP_NEIGHBORS)
    real_edges = sample_edges[sample_mask]
    builder = BatchBuilder(graph, sender_index, receiver_index, steps, transaction_features,
                           sample_nodes.mean(0), sample_nodes.std(0) + 1e-6, real_edges.mean(0), real_edges.std(0) + 1e-6)
    labels = with_features.isFraud.values
    log(f"prepared: {len(with_features):,} transactions, {graph.both['pointer'].shape[0] - 1:,} accounts, "
        f"train {len(rows['train']):,} / validation {len(rows['validation']):,} / test {len(rows['test']):,}")
    return builder, rows, labels, transaction_features.shape[1]


def train_one_config(config_index, config, builder, rows, labels, transaction_size, smoke):
    torch.manual_seed(RANDOM_SEED); rng = np.random.default_rng(RANDOM_SEED + config_index)
    model = GraphSageEdgeClassifier(4, 4, transaction_size, config["hidden_size"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_fraud = rows["train"][labels[rows["train"]] == 1]
    train_legit = rows["train"][labels[rows["train"]] == 0]
    best = {"validation_pr_auc": -1}; epochs_without_improvement = 0; history = []
    max_epochs = 2 if smoke else MAX_EPOCHS
    for epoch in range(1, max_epochs + 1):
        start_time = time.time(); model.train()
        legit_sample = rng.choice(train_legit, size=int(len(train_legit) * LEGIT_SHARE_PER_EPOCH), replace=False)
        epoch_rows = rng.permutation(np.concatenate([train_fraud, legit_sample]))
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(legit_sample) / len(train_fraud)))
        total_loss = 0.0
        for start in range(0, len(epoch_rows), BATCH_SIZE):
            batch_rows = epoch_rows[start:start + BATCH_SIZE]
            loss = loss_function(model(builder.build(batch_rows)), torch.from_numpy(labels[batch_rows].astype(np.float32)))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item() * len(batch_rows)
        validation_scores = score(model, builder, rows["validation"])
        validation_pr_auc = average_precision_score(labels[rows["validation"]], validation_scores)
        history.append({"epoch": epoch, "train_loss": total_loss / len(epoch_rows), "validation_pr_auc": float(validation_pr_auc),
                        "seconds": round(time.time() - start_time, 1)})
        log(f"config {config_index} {config} epoch {epoch}: train loss {history[-1]['train_loss']:.4f}, "
            f"validation PR-AUC {validation_pr_auc:.4f} ({history[-1]['seconds']}s)")
        if validation_pr_auc > best["validation_pr_auc"]:
            best = {"validation_pr_auc": float(validation_pr_auc), "epoch": epoch}
            torch.save(model.state_dict(), RUNS_DIR / f"graphsage_{config_index}.pt"); epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                log("early stopping"); break
    entry = {"config_index": config_index, "parameters": config, "validation_pr_auc": best["validation_pr_auc"],
             "best_epoch": best["epoch"], "history": history}
    save_json(entry, RUNS_DIR / f"graphsage_{config_index}.json")
    return entry


def main(smoke=False, no_graph=False):
    global RUNS_DIR, CONFIGS
    experiment = "e3_nograph" if no_graph else "e3"
    if smoke:
        RUNS_DIR = RESULTS_DIR / "e3_smoke_runs"  # never mixed with real runs
    if no_graph:
        RUNS_DIR = RESULTS_DIR / "e3_nograph_runs"
        CONFIGS = [json.loads((RESULTS_DIR / "e3_metrics.json").read_text())["models"]["graphsage"]["best_parameters"]]
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    builder, rows, labels, transaction_size = prepare(smoke)
    leakage_test(builder, rows["test"])
    builder.drop_neighbors = no_graph
    if no_graph:
        log("ABLATION: neighbors removed (same network, same settings, no graph information)")
    tuning_log = []
    for config_index, config in enumerate(CONFIGS[:1] if smoke else CONFIGS):
        done_path = RUNS_DIR / f"graphsage_{config_index}.json"
        if done_path.exists() and not smoke:
            log(f"config {config_index} already trained, loading result"); tuning_log.append(json.loads(done_path.read_text())); continue
        tuning_log.append(train_one_config(config_index, config, builder, rows, labels, transaction_size, smoke))
    if smoke:
        log("SMOKE TEST OK"); return

    best_entry = max(tuning_log, key=lambda e: e["validation_pr_auc"])
    model = GraphSageEdgeClassifier(4, 4, transaction_size, best_entry["parameters"]["hidden_size"], best_entry["parameters"]["dropout"])
    model.load_state_dict(torch.load(RUNS_DIR / f"graphsage_{best_entry['config_index']}.pt"))
    validation_labels, test_labels = labels[rows["validation"]], labels[rows["test"]]
    validation_scores = score(model, builder, rows["validation"])
    test_scores = score(model, builder, rows["test"])                                    # test touched once
    f1_threshold = best_f1_threshold(validation_labels, validation_scores)               # both frozen on validation
    floor_threshold = threshold_for_minimum_precision(validation_labels, validation_scores)
    result = {"best_parameters": best_entry["parameters"], "best_epoch": best_entry["best_epoch"], "tuning": tuning_log,
              "validation": metrics_at_threshold(validation_labels, validation_scores, f1_threshold),
              "test": metrics_at_threshold(test_labels, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(validation_labels, validation_scores, floor_threshold),
              "test_precision_floor": metrics_at_threshold(test_labels, test_scores, floor_threshold)}
    save_json({"experiment": ("E3b ablation: same network with all graph neighbors removed" if no_graph else
                              "E3b GraphSAGE (2-layer mean aggregator, temporal sampling) + MLP head, edge classification"),
               "split": "train steps 1-446, validation 447-595, test 596-743",
               "sampling": f"{FIRST_HOP_NEIGHBORS} then {SECOND_HOP_NEIGHBORS} most recent neighbors with step < target step",
               "models": {"graphsage": result}}, RESULTS_DIR / f"{experiment}_metrics.json")
    pd.DataFrame({"isFraud": test_labels, "fraud_score": test_scores}).to_parquet(RESULTS_DIR / f"{experiment}_test_scores_graphsage.parquet")
    plot_confusion_matrix(result["test"], f"{experiment} (test, threshold {f1_threshold:.3f})", FIGURES_DIR / f"{experiment}_confusion_graphsage.png")
    plot_pr_curves({"graphsage": (test_labels, test_scores)}, float(test_labels.mean()), f"{experiment} test precision-recall",
                   FIGURES_DIR / f"{experiment}_pr_curves_test.png")
    t = result["test"]; p = result["test_precision_floor"]
    log(f"ALL DONE -> results/{experiment}_metrics.json | test PR-AUC {t['pr_auc']:.4f} | best-F1 cutoff recall {t['recall']:.3f} "
        f"precision {t['precision']:.3f} | precision>=50% cutoff recall {p['recall']:.3f} precision {p['precision']:.3f}")


if __name__ == "__main__":
    # python src/graphsage.py | python src/graphsage.py smoke | python src/graphsage.py no_graph
    main(smoke=len(sys.argv) > 1 and sys.argv[1] == "smoke", no_graph=len(sys.argv) > 1 and sys.argv[1] == "no_graph")
