"""Deliberately NAIVE runs that show how much leakage inflates scores (PaySim).
Run on the Mac, from the project root:
    python src/naive_leakage.py balance
Variants (added one at a time):
    balance        : honest E1 features + the 4 balance columns + isFlaggedFraud, same temporal split.
    random_honest  : honest E1 features only, but a random (shuffled) split.
    random_balance : balance columns + random split.
The model settings are FIXED to the honest E1 XGBoost best run (depth 6, learning rate 0.05),
so the only change versus E1 is the leaked inputs. No tuning is done here.
These inputs are forbidden in E1-E3 and appear only in this file."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, MINIMUM_ALARM_PRECISION, best_f1_threshold,
                          metrics_at_threshold, plot_confusion_matrix, save_json,
                          threshold_for_minimum_precision)
from src.features_tabular import MODEL_FEATURE_COLUMNS

RANDOM_SEED = 42
PROCESSED_DIR = Path("data/processed")
BALANCE_LEAK_COLUMNS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]
FIXED_XGBOOST_SETTINGS = {"max_depth": 6, "learning_rate": 0.05}  # best honest E1 setting, chosen on validation


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_split_frames(feature_columns: list) -> dict:
    """Read the E1 feature files (they already hold the original columns, row-aligned with the split)."""
    frames = {}
    for name in ["train", "validation", "test"]:
        frame = pd.read_parquet(PROCESSED_DIR / f"e1_features_{name}.parquet", columns=feature_columns + ["isFraud"])
        frames[name] = frame
        log(f"{name}: shape {frame.shape}, fraud {int(frame['isFraud'].sum())} ({frame['isFraud'].mean():.4%})")
    print(frames["train"].head(5).to_string(), flush=True)
    return frames


def train_and_evaluate(frames: dict, feature_columns: list, experiment_name: str, split_description: str):
    labels = {name: frame["isFraud"].values for name, frame in frames.items()}
    legit_to_fraud_ratio = float((labels["train"] == 0).sum() / (labels["train"] == 1).sum())
    model = XGBClassifier(n_estimators=600, tree_method="hist", scale_pos_weight=legit_to_fraud_ratio,
                          eval_metric="aucpr", early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.9,
                          n_jobs=-1, random_state=RANDOM_SEED, **FIXED_XGBOOST_SETTINGS)
    start = time.time()
    model.fit(frames["train"][feature_columns], labels["train"],
              eval_set=[(frames["validation"][feature_columns], labels["validation"])], verbose=False)
    log(f"trained in {time.time() - start:.0f}s, best iteration {model.best_iteration}")

    validation_scores = model.predict_proba(frames["validation"][feature_columns])[:, 1]
    best_f1_cut = best_f1_threshold(labels["validation"], validation_scores)                 # frozen on validation
    precision_floor_cut = threshold_for_minimum_precision(labels["validation"], validation_scores)  # frozen on validation
    test_scores = model.predict_proba(frames["test"][feature_columns])[:, 1]                 # test touched once

    results = {
        "experiment": experiment_name, "features": feature_columns, "split": split_description,
        "xgboost_settings": FIXED_XGBOOST_SETTINGS, "best_iteration": int(model.best_iteration),
        "validation_pr_auc": float(average_precision_score(labels["validation"], validation_scores)),
        "validation": metrics_at_threshold(labels["validation"], validation_scores, best_f1_cut),
        "test": metrics_at_threshold(labels["test"], test_scores, best_f1_cut),
        "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
        "test_precision_floor": metrics_at_threshold(labels["test"], test_scores, precision_floor_cut),
        "feature_importance_gain": {k: float(v) for k, v in model.get_booster().get_score(importance_type="gain").items()},
    }
    save_json(results, RESULTS_DIR / f"{experiment_name}_metrics.json")
    pd.DataFrame({"isFraud": labels["test"], "fraud_score": test_scores}).to_parquet(
        RESULTS_DIR / f"{experiment_name}_test_scores_xgboost.parquet")
    plot_confusion_matrix(results["test"], f"{experiment_name} XGBoost (test, threshold {best_f1_cut:.3f})",
                          FIGURES_DIR / f"{experiment_name}_confusion_xgboost.png")
    test = results["test"]
    log(f"RESULT {experiment_name}: test PR-AUC {test['pr_auc']:.4f} (random baseline {test['pr_auc_random_baseline']:.4f}), "
        f"precision {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}")
    floor = results["test_precision_floor"]
    log(f"        precision-floor cut: precision {floor['precision']:.3f}, recall {floor['recall']:.3f}")
    top = sorted(results["feature_importance_gain"].items(), key=lambda item: -item[1])[:5]
    log("        top features by gain: " + ", ".join(f"{name} {gain:.0f}" for name, gain in top))
    return results


def run_balance_variant():
    feature_columns = MODEL_FEATURE_COLUMNS + BALANCE_LEAK_COLUMNS
    frames = load_split_frames(feature_columns)
    train_and_evaluate(frames, feature_columns, "naive_balance",
                       "temporal: train steps 1-446, validation 447-595, test 596-743 (same as E1)")


def shuffle_into_random_split(frames: dict) -> dict:
    """Pool train + validation + test, shuffle all rows, cut into the SAME three sizes as the temporal split.
    Rows from every period now land in every split, so the model trains on neighbours of its test rows."""
    pooled = pd.concat([frames["train"], frames["validation"], frames["test"]], ignore_index=True)
    shuffled_order = np.random.default_rng(RANDOM_SEED).permutation(len(pooled))
    train_size, validation_size = len(frames["train"]), len(frames["validation"])
    cut_train, cut_validation = train_size, train_size + validation_size
    random_frames = {"train": pooled.iloc[shuffled_order[:cut_train]].reset_index(drop=True),
                     "validation": pooled.iloc[shuffled_order[cut_train:cut_validation]].reset_index(drop=True),
                     "test": pooled.iloc[shuffled_order[cut_validation:]].reset_index(drop=True)}
    for name, frame in random_frames.items():
        log(f"random {name}: shape {frame.shape}, fraud {int(frame['isFraud'].sum())} ({frame['isFraud'].mean():.4%})")
    print(random_frames["train"].head(5).to_string(), flush=True)
    return random_frames


RANDOM_SPLIT_DESCRIPTION = "RANDOM shuffle of all rows, same sizes as temporal split (2,624,623 / 97,044 / 48,742); fraud prevalence ~0.30% in every split"


def run_random_honest_variant():
    """Honest E1 features only, but a random split: isolates the effect of the random split."""
    feature_columns = MODEL_FEATURE_COLUMNS
    frames = shuffle_into_random_split(load_split_frames(feature_columns))
    train_and_evaluate(frames, feature_columns, "naive_random_honest", RANDOM_SPLIT_DESCRIPTION)


def run_random_balance_variant():
    """Balance columns + random split: the cumulative staircase step."""
    feature_columns = MODEL_FEATURE_COLUMNS + BALANCE_LEAK_COLUMNS
    frames = shuffle_into_random_split(load_split_frames(feature_columns))
    train_and_evaluate(frames, feature_columns, "naive_random_balance", RANDOM_SPLIT_DESCRIPTION)


if __name__ == "__main__":
    variants = {"balance": run_balance_variant, "random_honest": run_random_honest_variant,
                "random_balance": run_random_balance_variant}
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    if variant in variants:
        variants[variant]()
    else:
        sys.exit("usage: python src/naive_leakage.py <balance|random_honest|random_balance>")
