"""IBM AML E3 v2 (2026-10-29): upgraded GraphSAGE on Sol GPU. New file; src/ibm_graphsage.py is unchanged and reused.
Upgrades over E3b: (1) transaction input = the 120 ibm_e2r features (E1 + Neo4j + patterns + round-2), signed-log + standardized
on train; (2) larger strictly past-only neighbourhoods, 20 then 10 most recent earlier transactions (was 10 then 5);
(3) GPU training with parallel CPU neighbour sampling; (4) 3 settings, each also trained with the graph removed (same network,
neighbours masked), so the graph's own contribution is measured. Validation decides everything; test is read only in 'finalize'.
Run on Sol:  python src/ibm_graphsage_v2.py train <task 0-5>   (0-2: with graph, 3-5: graph removed; one GPU job each)
             python src/ibm_graphsage_v2.py finalize            (best setting by validation -> test once; no-graph twin; blend with XGBoost)
             python src/ibm_graphsage_v2.py smoke               (2 short epochs on 5% of rows: code check only)"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ibm_graphsage as base_graphsage
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, MINIMUM_ALARM_PRECISION, best_f1_threshold, metrics_at_threshold,
                          plot_confusion_matrix, save_json, threshold_for_minimum_precision)
from src.graphsage import GraphSageEdgeClassifier
from src.ibm_ablation import ABLATION_DIR, BASE_EXPERIMENT, BASE_FEATURE_COLUMNS, WINDOW_FEATURES_FILE
from src.ibm_data_prep import PROCESSED_DIR
from src.ibm_train_tabular import EXPERIMENTS

base_graphsage.FIRST_HOP, base_graphsage.SECOND_HOP = 20, 10      # read by the reused sampler and batch builder
RANDOM_SEED = 42
BATCH_SIZE = 4096
LEGIT_SHARE_PER_EPOCH = 0.10
MAX_EPOCHS, PATIENCE = 20, 4
SAMPLING_WORKERS = 12
SETTINGS = [{"hidden_size": 128, "learning_rate": 1e-3, "dropout": 0.2},
            {"hidden_size": 256, "learning_rate": 1e-3, "dropout": 0.3},
            {"hidden_size": 128, "learning_rate": 3e-4, "dropout": 0.1}]
TASKS = [(index, "graph") for index in range(3)] + [(index, "no_graph") for index in range(3)]
RUNS_DIR = RESULTS_DIR / "ibm_e3v2_runs"
FINAL_XGBOOST = ABLATION_DIR / "plus_all" / "xgboost_13.joblib"     # ibm_e2r (chosen on validation 2026-10-29)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def e2r_feature_matrix(number_of_transactions, train_rows):
    """The 120 ibm_e2r columns for every transaction, in transaction_id order (= row order of the graph);
    signed log1p for heavy tails, then standardized with train mean/std only."""
    new_columns = [c for c in pd.read_parquet(WINDOW_FEATURES_FILE).columns if c != "transaction_id"]
    window = pd.read_parquet(WINDOW_FEATURES_FILE).set_index("transaction_id")
    feature_columns = BASE_FEATURE_COLUMNS + new_columns
    matrix = np.zeros((number_of_transactions, len(feature_columns)), dtype=np.float32)
    for name in ["train", "validation", "test"]:
        frame = pd.read_parquet(PROCESSED_DIR / f"{EXPERIMENTS[BASE_EXPERIMENT]['files']}_{name}.parquet",
                                columns=["transaction_id"] + BASE_FEATURE_COLUMNS)
        ids = frame.transaction_id.to_numpy()
        matrix[ids, :len(BASE_FEATURE_COLUMNS)] = frame[BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        matrix[ids, len(BASE_FEATURE_COLUMNS):] = window.loc[ids, new_columns].to_numpy(dtype=np.float32)
        del frame
    matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
    mean = matrix[train_rows].mean(0); std = matrix[train_rows].std(0) + 1e-6
    matrix = (matrix - mean) / std
    log(f"transaction features: {matrix.shape} (ibm_e2r columns), standardized on train")
    return matrix, feature_columns


def prepare():
    builder, rows, labels, is_tail, _, edge_size = base_graphsage.prepare(False)   # graph, splits, labels (E1 inputs replaced below)
    features, feature_columns = e2r_feature_matrix(len(labels), rows["train"])
    builder.transaction_features = features
    return builder, rows, labels, is_tail, features.shape[1], edge_size


class BatchesOfRows(Dataset):
    """Each item is one whole batch, built in a worker process (neighbour sampling is the slow, CPU part)."""
    def __init__(self, builder, row_batches):
        self.builder, self.row_batches = builder, row_batches

    def __len__(self):
        return len(self.row_batches)

    def __getitem__(self, index):
        return self.builder.build(self.row_batches[index])


def batches(builder, rows, shuffle_rng=None):
    if shuffle_rng is not None:
        rows = shuffle_rng.permutation(rows)
    row_batches = [rows[start:start + BATCH_SIZE] for start in range(0, len(rows), BATCH_SIZE)]
    loader = DataLoader(BatchesOfRows(builder, row_batches), batch_size=None, shuffle=False,
                        num_workers=SAMPLING_WORKERS, persistent_workers=False, prefetch_factor=2)
    for row_batch, tensors in zip(row_batches, loader):
        yield row_batch, {key: value.to(DEVICE, non_blocking=True) for key, value in tensors.items()}


def score(model, builder, rows):
    model.eval(); scores = []
    with torch.no_grad():
        for _, tensors in batches(builder, rows):
            scores.append(torch.sigmoid(model(tensors)).float().cpu().numpy())
    return np.concatenate(scores)


def new_model(setting, transaction_size, edge_size):
    return GraphSageEdgeClassifier(4, edge_size, transaction_size, setting["hidden_size"], setting["dropout"]).to(DEVICE)


def train(task, smoke=False):
    index, mode = TASKS[task]; setting = SETTINGS[index]
    name = f"setting_{index}_{mode}" + ("_smoke" if smoke else "")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"task {task}: {name} {setting} on {DEVICE}, neighbours {base_graphsage.FIRST_HOP} + {base_graphsage.SECOND_HOP}")
    builder, rows, labels, _, transaction_size, edge_size = prepare()
    base_graphsage.leakage_test(builder, rows["test"])          # sampler check only (no test labels or scores used)
    builder.drop_neighbors = mode == "no_graph"
    torch.manual_seed(RANDOM_SEED); rng = np.random.default_rng(RANDOM_SEED + index)
    model = new_model(setting, transaction_size, edge_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting["learning_rate"])
    train_rows = rows["train"]
    if smoke:
        train_rows = np.concatenate([train_rows[labels[train_rows] == 1], rng.choice(train_rows[labels[train_rows] == 0], 150_000, replace=False)])
    positives = train_rows[labels[train_rows] == 1]; negatives = train_rows[labels[train_rows] == 0]
    monitor_rng = np.random.default_rng(RANDOM_SEED)           # same early-stopping monitor rule as E3b (speed only)
    validation_positive = rows["validation"][labels[rows["validation"]] == 1]
    validation_negative = rows["validation"][labels[rows["validation"]] == 0]
    monitor_share = 50 if smoke else 4
    monitor_rows = np.sort(np.concatenate([validation_positive, monitor_rng.choice(validation_negative, len(validation_negative) // monitor_share, replace=False)]))
    best = {"monitor_pr_auc": -1.0, "epoch": 0}; stale = 0; history = []
    for epoch in range(1, (2 if smoke else MAX_EPOCHS) + 1):
        start = time.time(); model.train()
        sampled = rng.choice(negatives, size=int(len(negatives) * LEGIT_SHARE_PER_EPOCH), replace=False)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(sampled) / len(positives), device=DEVICE))
        total, seen = 0.0, 0
        for row_batch, tensors in batches(builder, np.concatenate([positives, sampled]), shuffle_rng=rng):
            target = torch.from_numpy(labels[row_batch].astype(np.float32)).to(DEVICE)
            loss = loss_function(model(tensors), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(row_batch); seen += len(row_batch)
        monitor_pr_auc = float(average_precision_score(labels[monitor_rows], score(model, builder, monitor_rows)))
        history.append({"epoch": epoch, "train_loss": total / seen, "monitor_pr_auc": monitor_pr_auc, "seconds": round(time.time() - start, 1)})
        log(f"{name} epoch {epoch}: loss {total / seen:.4f}, validation (monitor) PR-AUC {monitor_pr_auc:.4f} ({history[-1]['seconds']}s)")
        if monitor_pr_auc > best["monitor_pr_auc"]:
            best = {"monitor_pr_auc": monitor_pr_auc, "epoch": epoch}; stale = 0
            torch.save(model.state_dict(), RUNS_DIR / f"{name}.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                log("early stopping"); break
    model.load_state_dict(torch.load(RUNS_DIR / f"{name}.pt", map_location=DEVICE))
    validation_scores = score(model, builder, rows["validation"])            # FULL validation set
    validation_pr_auc = float(average_precision_score(labels[rows["validation"]], validation_scores))
    np.save(RUNS_DIR / f"{name}_validation_scores.npy", validation_scores)
    save_json({"task": task, "setting_index": index, "mode": mode, "parameters": setting, "neighbours": [base_graphsage.FIRST_HOP, base_graphsage.SECOND_HOP],
               "validation_pr_auc": validation_pr_auc, "best_epoch": best["epoch"], "history": history}, RUNS_DIR / f"{name}.json")
    log(f"DONE {name}: best epoch {best['epoch']}, FULL validation PR-AUC {validation_pr_auc:.4f}")


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


def report(model_name, yv, validation_scores, yt, test_scores, tail, extra=None):
    f1_threshold = best_f1_threshold(yv, validation_scores); floor = threshold_for_minimum_precision(yv, validation_scores)
    result = {"validation": metrics_at_threshold(yv, validation_scores, f1_threshold), "test": metrics_at_threshold(yt, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(yv, validation_scores, floor),
              "test_precision_floor": metrics_at_threshold(yt, test_scores, floor),
              "test_normal_days_precision_floor": metrics_at_threshold(yt[~tail], test_scores[~tail], floor),
              "test_tail_precision_floor": metrics_at_threshold(yt[tail], test_scores[tail], floor), **(extra or {})}
    normal = result["test_normal_days_precision_floor"]
    log(f"{model_name:28s} validation {average_precision_score(yv, validation_scores):.4f} | test PR-AUC Sept 9-10 "
        f"{average_precision_score(yt[~tail], test_scores[~tail]):.4f}, all {result['test']['pr_auc']:.4f}, F1 all {result['test']['f1']:.3f} | "
        f"Sept 9-10 caught {normal['confusion_matrix']['true_positive']} / false alarms {normal['confusion_matrix']['false_positive']}")
    return result


def finalize():
    runs = {(r["setting_index"], r["mode"]): r for r in (json.loads(p.read_text()) for p in RUNS_DIR.glob("setting_*.json") if "smoke" not in p.name)}
    assert all((i, m) in runs for i, m in TASKS), f"missing runs: {[t for t in TASKS if t not in runs]}"
    for (i, m), r in sorted(runs.items()):
        log(f"  setting {i} {m:8s}: validation PR-AUC {r['validation_pr_auc']:.4f} (best epoch {r['best_epoch']})")
    best_index = max(range(3), key=lambda i: runs[(i, "graph")]["validation_pr_auc"])           # chosen on validation only
    log(f"chosen on validation: setting {best_index} {SETTINGS[best_index]}")
    builder, rows, labels, is_tail, transaction_size, edge_size = prepare()
    yv, yt, tail = labels[rows["validation"]], labels[rows["test"]], is_tail[rows["test"]]
    scores = {}
    for mode in ["graph", "no_graph"]:
        model = new_model(SETTINGS[best_index], transaction_size, edge_size)
        model.load_state_dict(torch.load(RUNS_DIR / f"setting_{best_index}_{mode}.pt", map_location=DEVICE))
        builder.drop_neighbors = mode == "no_graph"
        scores[mode] = (np.load(RUNS_DIR / f"setting_{best_index}_{mode}_validation_scores.npy"), score(model, builder, rows["test"]))  # test once
    # blend with the final XGBoost (ibm_e2r): logit average, weight chosen on validation PR-AUC only
    xgboost_model = joblib.load(FINAL_XGBOOST)
    xgboost_columns = json.loads((ABLATION_DIR / "plus_all" / "xgboost_13.json").read_text())["feature_columns"]
    from src.ibm_ablation_finalize import load_split
    _, xgb_validation_inputs, _ = load_split("validation", xgboost_columns); _, xgb_test_inputs, _ = load_split("test", xgboost_columns)
    xgb_validation = xgboost_model.predict_proba(xgb_validation_inputs)[:, 1]; xgb_test = xgboost_model.predict_proba(xgb_test_inputs)[:, 1]
    gnn_validation, gnn_test = scores["graph"]
    weights = np.round(np.arange(0.0, 1.01, 0.1), 1)
    validation_by_weight = {float(w): float(average_precision_score(yv, w * logit(gnn_validation) + (1 - w) * logit(xgb_validation))) for w in weights}
    best_weight = max(validation_by_weight, key=validation_by_weight.get)
    log(f"blend weight for GraphSAGE chosen on validation: {best_weight} (validation PR-AUC by weight: {validation_by_weight})")
    to_probability = lambda z: 1 / (1 + np.exp(-z))
    blend_validation = to_probability(best_weight * logit(gnn_validation) + (1 - best_weight) * logit(xgb_validation))
    blend_test = to_probability(best_weight * logit(gnn_test) + (1 - best_weight) * logit(xgb_test))
    outputs = {"ibm_e3v2": ("E3 v2 GraphSAGE (e2r inputs, 20+10 neighbours)", scores["graph"], {"best_parameters": SETTINGS[best_index]}),
               "ibm_e3v2_nograph": ("E3 v2 same network, graph removed", scores["no_graph"], {"best_parameters": SETTINGS[best_index]}),
               "ibm_e3v2_blend": ("ibm_e2r XGBoost + E3 v2 GraphSAGE blend", (blend_validation, blend_test),
                                  {"graphsage_weight": best_weight, "validation_pr_auc_by_weight": validation_by_weight})}
    for experiment, (title, (validation_scores, test_scores), extra) in outputs.items():
        result = report(experiment, yv, validation_scores, yt, test_scores, tail, extra)
        save_json({"experiment": title, "models": {"graphsage": result}}, RESULTS_DIR / f"{experiment}_metrics.json")
        pd.DataFrame({"label": yt, "fraud_score": test_scores, "is_tail": tail}).to_parquet(RESULTS_DIR / f"{experiment}_test_scores_graphsage.parquet")
        plot_confusion_matrix(result["test_precision_floor"], f"{experiment} (test)", FIGURES_DIR / f"{experiment}_confusion_graphsage.png")
    log("FINALIZE DONE")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        train(int(sys.argv[2]))
    elif command == "smoke":
        SAMPLING_WORKERS = 4
        train(0, smoke=True)
    elif command == "finalize":
        finalize()
