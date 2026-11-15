"""Random-search tuning of XGBoost for the IBM experiments (validation decides everything, incl. scale_pos_weight).
The SAME 30 random settings (fixed seed) are tried for every experiment, so the search budget is equal.
Run on the Mac:  python src/ibm_tune_xgboost.py ibm_e1 ibm_e2 ibm_e2p ibm_e2b      (resumable)
Results: results/<experiment>_tuned_runs/*.json, results/<experiment>_tuned_metrics.json, test scores parquet."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluate import (RESULTS_DIR, FIGURES_DIR, best_f1_threshold, metrics_at_threshold, plot_confusion_matrix,
                          save_json, threshold_for_minimum_precision, MINIMUM_ALARM_PRECISION)
from src.ibm_train_tabular import load_inputs, EXPERIMENTS

NUMBER_OF_SETTINGS = 30
SEARCH_SEED = 2026
RANDOM_SEED = 42


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def search_space():
    """30 random settings, identical for every experiment. Setting 0 = the earlier default-style setting for reference."""
    rng = np.random.default_rng(SEARCH_SEED)
    settings = [{"max_depth": 8, "learning_rate": 0.1, "min_child_weight": 1.0, "reg_lambda": 1.0,
                 "subsample": 0.8, "colsample_bytree": 0.9, "scale_pos_weight": "class_ratio"}]
    while len(settings) < NUMBER_OF_SETTINGS:
        settings.append({"max_depth": int(rng.integers(4, 13)),
                         "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                         "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(100)))),
                         "reg_lambda": float(np.exp(rng.uniform(np.log(0.01), np.log(100)))),
                         "subsample": float(rng.uniform(0.5, 1.0)),
                         "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                         "scale_pos_weight": float(np.exp(rng.uniform(np.log(1), np.log(1500))))})
    return settings


def build_model(setting, class_ratio):
    from xgboost import XGBClassifier
    parameters = dict(setting)
    if parameters["scale_pos_weight"] == "class_ratio":
        parameters["scale_pos_weight"] = class_ratio
    return XGBClassifier(n_estimators=2000, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
                         n_jobs=-1, random_state=RANDOM_SEED, **parameters)


def tune(experiment):
    runs_dir = RESULTS_DIR / f"{experiment}_tuned_runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    labels, inputs, class_ratio, is_tail, feature_columns = load_inputs(experiment, need_linear=False)
    log(f"{experiment}: {inputs['train'].shape[1]} features, class ratio {class_ratio:.0f}")
    for index, setting in enumerate(search_space()):
        done = runs_dir / f"xgboost_{index}.json"
        if done.exists():
            continue
        start = time.time(); model = build_model(setting, class_ratio)
        model.fit(inputs["train"], labels["train"], eval_set=[(inputs["validation"], labels["validation"])], verbose=False)
        validation_pr_auc = float(average_precision_score(labels["validation"], model.predict_proba(inputs["validation"])[:, 1]))
        entry = {"config_index": index, "parameters": setting, "validation_pr_auc": validation_pr_auc,
                 "best_iteration": int(model.best_iteration), "seconds": round(time.time() - start, 1)}
        save_json(entry, done); joblib.dump(model, runs_dir / f"xgboost_{index}.joblib")
        log(f"{experiment} setting {index:2d}: weight {model.get_params()['scale_pos_weight']:8.1f} depth {setting['max_depth']:2d} "
            f"lr {setting['learning_rate']:.3f} -> validation PR-AUC {validation_pr_auc:.4f} ({entry['seconds']}s, {entry['best_iteration']} rounds)")

    tuning = [json.loads((runs_dir / f"xgboost_{i}.json").read_text()) for i in range(NUMBER_OF_SETTINGS)]
    best = max(tuning, key=lambda e: e["validation_pr_auc"])
    model = joblib.load(runs_dir / f"xgboost_{best['config_index']}.joblib")
    validation_scores = model.predict_proba(inputs["validation"])[:, 1]
    test_scores = model.predict_proba(inputs["test"])[:, 1]                       # test touched once
    yv, yt = labels["validation"], labels["test"]
    f1_threshold = best_f1_threshold(yv, validation_scores); floor = threshold_for_minimum_precision(yv, validation_scores)
    result = {"best_parameters": best["parameters"], "best_scale_pos_weight": float(model.get_params()["scale_pos_weight"]),
              "best_iteration": best["best_iteration"], "tuning": tuning,
              "validation": metrics_at_threshold(yv, validation_scores, f1_threshold), "test": metrics_at_threshold(yt, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(yv, validation_scores, floor),
              "test_precision_floor": metrics_at_threshold(yt, test_scores, floor),
              "test_normal_days_precision_floor": metrics_at_threshold(yt[~is_tail], test_scores[~is_tail], floor),
              "test_tail_precision_floor": metrics_at_threshold(yt[is_tail], test_scores[is_tail], floor),
              "feature_importance_gain": {feature_columns[int(k[1:])]: float(v) for k, v in model.get_booster().get_score(importance_type="gain").items()}}
    save_json({"experiment": EXPERIMENTS[experiment]["title"] + " (XGBoost, 30-setting random search)", "models": {"xgboost": result}},
              RESULTS_DIR / f"{experiment}_tuned_metrics.json")
    pd.DataFrame({"label": yt, "fraud_score": test_scores, "is_tail": is_tail}).to_parquet(RESULTS_DIR / f"{experiment}_tuned_test_scores_xgboost.parquet")
    plot_confusion_matrix(result["test_precision_floor"], f"{experiment} tuned XGBoost (test)", FIGURES_DIR / f"{experiment}_tuned_confusion_xgboost.png")
    n = ~is_tail
    log(f"DONE {experiment}: best weight {result['best_scale_pos_weight']:.1f}, validation PR-AUC {best['validation_pr_auc']:.4f}, "
        f"test PR-AUC Sept 9-10 {average_precision_score(yt[n], test_scores[n]):.4f}, all {result['test']['pr_auc']:.4f}, "
        f"F1 all {result['test']['f1']:.3f}")


if __name__ == "__main__":
    for experiment in sys.argv[1:]:
        tune(experiment)
    log("ALL TUNING DONE")
