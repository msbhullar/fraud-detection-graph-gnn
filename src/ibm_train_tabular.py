"""IBM AML tabular experiments (E1 now, E2 later): same recipe as PaySim.
Tune on validation only, freeze both thresholds on validation, evaluate test once,
and report test separately for normal days (Sept 9-10) and the simulator tail (Sept 11-18).
Usage: python src/ibm_train_tabular.py ibm_e1 <model_name> <config_index>  |  python src/ibm_train_tabular.py ibm_e1 finalize"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, best_f1_threshold, metrics_at_threshold, plot_confusion_matrix,
                          plot_pr_curves, save_json, threshold_for_minimum_precision, MINIMUM_ALARM_PRECISION)
from src.ibm_data_prep import PROCESSED_DIR, TAIL_START
from src.ibm_features_tabular import IBM_E1_FEATURE_COLUMNS, NO_HISTORY_VALUE
from src.ibm_build_e2_features import IBM_GRAPH_FEATURE_COLUMNS
from src.ibm_pattern_features import PATTERN_FEATURE_COLUMNS

RANDOM_SEED = 42
LABEL = "Is Laundering"
EXPERIMENTS = {"ibm_e1": {"files": "ibm_e1_features", "features": IBM_E1_FEATURE_COLUMNS,
                          "title": "IBM E1 tabular baselines"},
               "ibm_e2": {"files": "ibm_e2_features", "features": IBM_E1_FEATURE_COLUMNS + IBM_GRAPH_FEATURE_COLUMNS,
                          "title": "IBM E2 tabular + Neo4j graph features"},
               "ibm_e2b": {"files": "ibm_e2b_features", "features": IBM_E1_FEATURE_COLUMNS + PATTERN_FEATURE_COLUMNS,
                           "title": "IBM E2b tabular + GFP-style pattern features"},
               "ibm_e2p": {"files": "ibm_e2p_features", "features": IBM_E1_FEATURE_COLUMNS + IBM_GRAPH_FEATURE_COLUMNS + PATTERN_FEATURE_COLUMNS,
                           "title": "IBM E2 + pattern features (tabular + Neo4j + GFP-style)"}}
LOG_COLUMNS = ["sender_prior_sent_count", "sender_prior_sent_amount_usd", "sender_prior_received_count", "sender_prior_received_amount_usd",
               "receiver_prior_sent_count", "receiver_prior_sent_amount_usd", "receiver_prior_received_count", "receiver_prior_received_amount_usd",
               "sender_minutes_since_last_activity", "receiver_minutes_since_last_activity",
               "amount_vs_sender_usual_sent", "amount_vs_receiver_usual_received"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def linear_column_values(frame: pd.DataFrame, column: str):
    """Yields (new_column_name, float32 values) for the logistic-regression inputs of one original feature column."""
    values = frame[column].to_numpy(dtype=np.float32)
    if column == "hour_of_day":
        yield "hour_sin", np.sin(2 * np.pi * values / 24).astype(np.float32)
        yield "hour_cos", np.cos(2 * np.pi * values / 24).astype(np.float32)
        return
    needs_log = column in LOG_COLUMNS or column.endswith(("_degree", "_amount", "_size", "_pagerank", "_1d", "_6h"))
    if needs_log:
        missing = values == NO_HISTORY_VALUE
        yield f"{column}_missing", missing.astype(np.float32)
        values = np.log1p(np.clip(np.where(missing, 0.0, values), 0, None)).astype(np.float32)
    yield column, values


def load_inputs(experiment: str, need_linear: bool):
    """Memory-light: float32, one column at a time; tree inputs OR standardized linear inputs."""
    config = EXPERIMENTS[experiment]; feature_columns = config["features"]
    labels, inputs, is_tail = {}, {}, None
    linear_mean, linear_std = None, None
    for name in ["train", "validation", "test"]:   # train first, so scaling is fitted on train only
        frame = pd.read_parquet(PROCESSED_DIR / f"{config['files']}_{name}.parquet", columns=feature_columns + [LABEL, "timestamp"])
        labels[name] = frame[LABEL].to_numpy()
        if name == "test":
            is_tail = (frame["timestamp"] >= TAIL_START).to_numpy()
        if need_linear:
            width = sum(1 for column in feature_columns for _ in linear_column_values(frame.head(2), column))
            matrix = np.empty((len(frame), width), dtype=np.float32)
            index = 0
            for column in feature_columns:            # write each new column straight into the matrix
                for _, values in linear_column_values(frame, column):
                    matrix[:, index] = values; index += 1
                del frame[column]
            if name == "train":
                linear_mean = matrix.mean(axis=0); linear_std = matrix.std(axis=0) + 1e-6
            matrix -= linear_mean; matrix /= linear_std
            inputs[name] = matrix
        else:
            inputs[name] = frame[feature_columns].to_numpy(dtype=np.float32)
        del frame
    ratio = float((labels["train"] == 0).sum() / (labels["train"] == 1).sum())
    return labels, inputs, ratio, is_tail, feature_columns


def model_settings(ratio):
    from xgboost import XGBClassifier
    return {
        "logistic_regression": ([{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
            lambda p: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED, **p)),
        "random_forest": ([
            {"max_depth": 12, "min_samples_leaf": 1}, {"max_depth": 20, "min_samples_leaf": 5},
            {"max_depth": 20, "min_samples_leaf": 50}, {"max_depth": None, "min_samples_leaf": 20}],
            lambda p: RandomForestClassifier(n_estimators=100, max_samples=0.1, class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=RANDOM_SEED, **p)),
        "xgboost": ([{"max_depth": d, "learning_rate": lr} for d in [4, 6, 8] for lr in [0.05, 0.1]],
            lambda p: XGBClassifier(n_estimators=2000, tree_method="hist", scale_pos_weight=ratio, eval_metric="aucpr",
                                    early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.9, n_jobs=-1,
                                    random_state=RANDOM_SEED, **p)),
    }


def run_one_config(experiment, model_name, config_index):
    runs_dir = RESULTS_DIR / f"{experiment}_runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    labels, inputs, ratio, _, _ = load_inputs(experiment, need_linear=(model_name == "logistic_regression"))
    grid, build = model_settings(ratio)[model_name]
    parameters = grid[config_index]; start = time.time(); model = build(parameters)
    if model_name == "xgboost":
        model.fit(inputs["train"], labels["train"], eval_set=[(inputs["validation"], labels["validation"])], verbose=False)
    else:
        model.fit(inputs["train"], labels["train"])
    validation_scores = model.predict_proba(inputs["validation"])[:, 1]
    entry = {"model": model_name, "config_index": config_index, "parameters": parameters,
             "validation_pr_auc": float(average_precision_score(labels["validation"], validation_scores)), "seconds": round(time.time() - start, 1)}
    if model_name == "xgboost":
        entry["best_iteration"] = int(model.best_iteration)
    save_json(entry, runs_dir / f"{model_name}_{config_index}.json"); joblib.dump(model, runs_dir / f"{model_name}_{config_index}.joblib")
    log(f"{experiment} {model_name} {parameters} -> validation PR-AUC {entry['validation_pr_auc']:.4f} ({entry['seconds']}s)")


def finalize_model(experiment, model_name):
    """Best setting on validation -> freeze both thresholds on validation -> score test once. One model per process (memory)."""
    runs_dir = RESULTS_DIR / f"{experiment}_runs"
    labels, inputs, ratio, is_tail, feature_columns = load_inputs(experiment, need_linear=(model_name == "logistic_regression"))
    grid, _ = model_settings(ratio)[model_name]
    tuning = [json.loads((runs_dir / f"{model_name}_{i}.json").read_text()) for i in range(len(grid))]
    best = max(tuning, key=lambda e: e["validation_pr_auc"])
    model = joblib.load(runs_dir / f"{model_name}_{best['config_index']}.joblib")
    validation_scores = model.predict_proba(inputs["validation"])[:, 1]
    test_scores = model.predict_proba(inputs["test"])[:, 1]
    del inputs
    f1_threshold = best_f1_threshold(labels["validation"], validation_scores)
    floor_threshold = threshold_for_minimum_precision(labels["validation"], validation_scores)
    y = labels["test"]
    result = {"best_parameters": best["parameters"], "tuning": tuning,
              "validation": metrics_at_threshold(labels["validation"], validation_scores, f1_threshold),
              "test": metrics_at_threshold(y, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(labels["validation"], validation_scores, floor_threshold),
              "test_precision_floor": metrics_at_threshold(y, test_scores, floor_threshold),
              "test_normal_days_precision_floor": metrics_at_threshold(y[~is_tail], test_scores[~is_tail], floor_threshold),
              "test_tail_precision_floor": metrics_at_threshold(y[is_tail], test_scores[is_tail], floor_threshold)}
    if model_name == "xgboost":
        result["feature_importance_gain"] = {feature_columns[int(k[1:])]: float(v) for k, v in model.get_booster().get_score(importance_type="gain").items()}
    save_json(result, runs_dir / f"final_{model_name}.json")
    plot_confusion_matrix(result["test_precision_floor"], f"{experiment} {model_name} (test, threshold {floor_threshold:.3f})",
                          FIGURES_DIR / f"{experiment}_confusion_{model_name}.png")
    joblib.dump(model, RESULTS_DIR / f"{experiment}_model_{model_name}.joblib")
    pd.DataFrame({"label": y, "fraud_score": test_scores, "is_tail": is_tail}).to_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model_name}.parquet")
    log(f"{experiment} {model_name} finalized")


def combine(experiment):
    runs_dir = RESULTS_DIR / f"{experiment}_runs"
    results = {"experiment": EXPERIMENTS[experiment]["title"], "features": EXPERIMENTS[experiment]["features"],
               "split": "train Sept 1-6, validation Sept 7-8, test Sept 9-18 (tail Sept 11-18 also reported separately)", "models": {}}
    curves = {}
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        results["models"][model_name] = json.loads((runs_dir / f"final_{model_name}.json").read_text())
        scores = pd.read_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model_name}.parquet")
        curves[model_name] = (scores.label.values, scores.fraud_score.values)
    plot_pr_curves(curves, float(scores.label.mean()), f"{experiment} test precision-recall", FIGURES_DIR / f"{experiment}_pr_curves_test.png")
    save_json(results, RESULTS_DIR / f"{experiment}_metrics.json"); log(f"ALL DONE -> results/{experiment}_metrics.json")


if __name__ == "__main__":
    # python src/ibm_train_tabular.py ibm_e2 finalize <model_name>   then   python src/ibm_train_tabular.py ibm_e2 combine
    if sys.argv[2] == "finalize":
        finalize_model(sys.argv[1], sys.argv[3])
    elif sys.argv[2] == "combine":
        combine(sys.argv[1])
    else:
        run_one_config(sys.argv[1], sys.argv[2], int(sys.argv[3]))
