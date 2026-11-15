"""E1: tabular baselines (Logistic Regression, Random Forest, XGBoost).
Tune on validation only, pick threshold on validation, evaluate test once."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, best_f1_threshold, metrics_at_threshold,
                          threshold_for_minimum_precision, MINIMUM_ALARM_PRECISION,
                          plot_confusion_matrix, plot_pr_curves, save_json)
from src.features_tabular import MODEL_FEATURE_COLUMNS, NO_HISTORY_VALUE
from src.build_e2_features import GRAPH_FEATURE_COLUMNS

EXPERIMENT_FEATURES = {"e1": MODEL_FEATURE_COLUMNS, "e2": MODEL_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS}
GRAPH_LOG_COLUMNS = [c for c in GRAPH_FEATURE_COLUMNS if "is_new_account" not in c]

RANDOM_SEED = 42
PROCESSED_DIR = Path("data/processed")
SENTINEL_COLUMNS = ["receiver_hours_since_last_payment", "amount_vs_receiver_prior_average"]
LOG_COLUMNS = ["amount", "receiver_prior_transaction_count", "receiver_prior_amount_sum",
               "receiver_hours_since_last_payment", "amount_vs_receiver_prior_average", "sender_prior_transaction_count"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def linear_model_inputs(features: pd.DataFrame, feature_columns: list = MODEL_FEATURE_COLUMNS) -> pd.DataFrame:
    """Logistic Regression needs reshaped inputs: no -1 sentinels, log scale, cyclic hour."""
    linear_inputs = features[feature_columns].copy()
    linear_inputs["receiver_has_history"] = (features["receiver_prior_transaction_count"] > 0).astype(int)
    for column in SENTINEL_COLUMNS:
        linear_inputs[column] = linear_inputs[column].where(linear_inputs[column] != NO_HISTORY_VALUE, 0.0)
    for column in LOG_COLUMNS + [c for c in GRAPH_LOG_COLUMNS if c in feature_columns]:
        linear_inputs[column] = np.log1p(linear_inputs[column])
    linear_inputs["hour_sin"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
    linear_inputs["hour_cos"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)
    return linear_inputs.drop(columns=["hour_of_day"])


def load_inputs(experiment: str):
    feature_columns = EXPERIMENT_FEATURES[experiment]
    frames = {name: pd.read_parquet(PROCESSED_DIR / f"{experiment}_features_{name}.parquet", columns=feature_columns + ["isFraud"])
              for name in ["train", "validation", "test"]}
    labels = {name: frame["isFraud"].values for name, frame in frames.items()}
    tree_inputs = {name: frame[feature_columns] for name, frame in frames.items()}
    linear_scaler = StandardScaler().fit(linear_model_inputs(frames["train"], feature_columns))  # fit on train only
    linear_inputs = {name: linear_scaler.transform(linear_model_inputs(frame, feature_columns)) for name, frame in frames.items()}
    legit_to_fraud_ratio = float((labels["train"] == 0).sum() / (labels["train"] == 1).sum())
    return labels, tree_inputs, linear_inputs, legit_to_fraud_ratio


def model_settings(tree_inputs, linear_inputs, legit_to_fraud_ratio):
    from xgboost import XGBClassifier  # imported here so other scripts can reuse this file without XGBoost
    return {
        "logistic_regression": (linear_inputs, [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
            lambda p: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED, **p)),
        "random_forest": (tree_inputs, [
            {"max_depth": 12, "min_samples_leaf": 1}, {"max_depth": 20, "min_samples_leaf": 5},
            {"max_depth": 20, "min_samples_leaf": 50}, {"max_depth": None, "min_samples_leaf": 20}],
            lambda p: RandomForestClassifier(n_estimators=100, max_samples=0.2, class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=RANDOM_SEED, **p)),
        "xgboost": (tree_inputs, [{"max_depth": d, "learning_rate": lr} for d in [4, 6, 8] for lr in [0.05, 0.1]],
            lambda p: XGBClassifier(n_estimators=600, tree_method="hist", scale_pos_weight=legit_to_fraud_ratio,
                                    eval_metric="aucpr", early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.9,
                                    n_jobs=-1, random_state=RANDOM_SEED, **p)),
    }




def run_one_config(experiment: str, model_name: str, config_index: int):
    """Train ONE setting on train, score validation, save. Short enough to run as its own job."""
    runs_dir = RESULTS_DIR / f"{experiment}_runs"
    labels, tree_inputs, linear_inputs, ratio = load_inputs(experiment)
    inputs, grid, build_model = model_settings(tree_inputs, linear_inputs, ratio)[model_name]
    parameters = grid[config_index]
    start = time.time()
    model = build_model(parameters)
    if model_name == "xgboost":
        model.fit(inputs["train"], labels["train"], eval_set=[(inputs["validation"], labels["validation"])], verbose=False)
    else:
        model.fit(inputs["train"], labels["train"])
    validation_scores = model.predict_proba(inputs["validation"])[:, 1]
    entry = {"model": model_name, "config_index": config_index, "parameters": parameters,
             "validation_pr_auc": float(average_precision_score(labels["validation"], validation_scores)),
             "seconds": round(time.time() - start, 1)}
    if model_name == "xgboost":
        entry["best_iteration"] = int(model.best_iteration)
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_json(entry, runs_dir / f"{model_name}_{config_index}.json")
    joblib.dump(model, runs_dir / f"{model_name}_{config_index}.joblib")
    log(f"{experiment} {model_name} {parameters} -> validation PR-AUC {entry['validation_pr_auc']:.4f} ({entry['seconds']}s)")


def finalize(experiment: str):
    """Pick best setting per model on validation, freeze threshold on validation, evaluate test once."""
    import json
    runs_dir = RESULTS_DIR / f"{experiment}_runs"
    feature_columns = EXPERIMENT_FEATURES[experiment]
    labels, tree_inputs, linear_inputs, ratio = load_inputs(experiment)
    settings = model_settings(tree_inputs, linear_inputs, ratio)
    title = {"e1": "E1 tabular baselines", "e2": "E2 tabular + Neo4j graph features"}[experiment]
    all_results = {"experiment": title, "features": feature_columns,
                   "split": "train steps 1-446, validation 447-595, test 596-743", "models": {}}
    test_curves = {}
    for model_name, (inputs, grid, _) in settings.items():
        tuning_log = [json.loads((runs_dir / f"{model_name}_{i}.json").read_text()) for i in range(len(grid))]
        best_entry = max(tuning_log, key=lambda e: e["validation_pr_auc"])
        best_model = joblib.load(runs_dir / f"{model_name}_{best_entry['config_index']}.joblib")
        validation_scores = best_model.predict_proba(inputs["validation"])[:, 1]
        threshold = best_f1_threshold(labels["validation"], validation_scores)  # frozen
        test_scores = best_model.predict_proba(inputs["test"])[:, 1]           # test touched once
        validation_metrics = metrics_at_threshold(labels["validation"], validation_scores, threshold)
        test_metrics = metrics_at_threshold(labels["test"], test_scores, threshold)
        high_recall_threshold = threshold_for_minimum_precision(labels["validation"], validation_scores)  # frozen
        result = {"best_parameters": best_entry["parameters"], "tuning": tuning_log,
                  "validation": validation_metrics, "test": test_metrics,
                  "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
                  "validation_precision_floor": metrics_at_threshold(labels["validation"], validation_scores, high_recall_threshold),
                  "test_precision_floor": metrics_at_threshold(labels["test"], test_scores, high_recall_threshold)}
        if model_name == "xgboost":
            result["feature_importance_gain"] = {k: float(v) for k, v in best_model.get_booster().get_score(importance_type="gain").items()}
        if model_name == "random_forest":
            result["feature_importance"] = dict(zip(feature_columns, map(float, best_model.feature_importances_)))
        all_results["models"][model_name] = result
        test_curves[model_name] = (labels["test"], test_scores)
        plot_confusion_matrix(test_metrics, f"{experiment.upper()} {model_name} (test, threshold {threshold:.3f})", FIGURES_DIR / f"{experiment}_confusion_{model_name}.png")
        joblib.dump(best_model, RESULTS_DIR / f"{experiment}_model_{model_name}.joblib")
        pd.DataFrame({"isFraud": labels["test"], "fraud_score": test_scores}).to_parquet(RESULTS_DIR / f"{experiment}_test_scores_{model_name}.parquet")
    plot_pr_curves(test_curves, float(labels["test"].mean()), f"{experiment.upper()} test precision-recall", FIGURES_DIR / f"{experiment}_pr_curves_test.png")
    save_json(all_results, RESULTS_DIR / f"{experiment}_metrics.json")
    log(f"ALL DONE -> results/{experiment}_metrics.json")


if __name__ == "__main__":
    # usage: python src/train_tabular.py <e1|e2> <model_name> <config_index>   |   python src/train_tabular.py <e1|e2> finalize
    if sys.argv[2] == "finalize":
        finalize(sys.argv[1])
    else:
        run_one_config(sys.argv[1], sys.argv[2], int(sys.argv[3]))
