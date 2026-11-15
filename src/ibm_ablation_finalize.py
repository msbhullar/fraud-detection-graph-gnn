"""Score the ONE final round-2 configuration on test, once (rule in docs/leakage_notes.md, 2026-10-28).
The variant is chosen beforehand from the validation-only table of src/ibm_ablation.py.
Its best validation setting (already trained) is reloaded; both thresholds are frozen on validation; test is scored once.
Output uses the same format as src/ibm_tune_xgboost.py, as experiment 'ibm_e2r' (E2 + patterns + round-2 features),
so src/bootstrap_compare.py works unchanged:  python src/bootstrap_compare.py ibm_e2p_tuned ibm_e2r_tuned xgboost:xgboost --normal-days --resamples=200
Run on Sol:  python src/ibm_ablation_finalize.py <variant>      e.g. plus_all
             python src/ibm_ablation_finalize.py auto           picks plus_all or plus_selected by validation PR-AUC (the written rule)"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import (FIGURES_DIR, RESULTS_DIR, MINIMUM_ALARM_PRECISION, best_f1_threshold, metrics_at_threshold,
                          plot_confusion_matrix, save_json, threshold_for_minimum_precision)
from src.ibm_ablation import ABLATION_DIR, BASE_EXPERIMENT, BASE_FEATURE_COLUMNS, WINDOW_FEATURES_FILE, best_run
from src.ibm_data_prep import PROCESSED_DIR, TAIL_START
from src.ibm_train_tabular import EXPERIMENTS, LABEL

FINAL_EXPERIMENT = "ibm_e2r"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_split(name, feature_columns):
    new_columns = [column for column in feature_columns if column not in BASE_FEATURE_COLUMNS]
    frame = pd.read_parquet(PROCESSED_DIR / f"{EXPERIMENTS[BASE_EXPERIMENT]['files']}_{name}.parquet",
                            columns=["transaction_id", "timestamp", LABEL] + BASE_FEATURE_COLUMNS)
    if new_columns:
        rows_before = len(frame)
        frame = frame.merge(pd.read_parquet(WINDOW_FEATURES_FILE, columns=["transaction_id"] + new_columns), on="transaction_id", how="left")
        assert len(frame) == rows_before and frame[new_columns].notna().all().all()
    log(f"  {name}: {len(frame):,} rows x {len(feature_columns)} features, laundering {int(frame[LABEL].sum()):,}")
    return frame[LABEL].to_numpy(), frame[feature_columns].to_numpy(dtype=np.float32), (frame["timestamp"] >= TAIL_START).to_numpy()


def main(variant):
    output = RESULTS_DIR / f"{FINAL_EXPERIMENT}_tuned_metrics.json"
    assert not output.exists(), f"{output} already exists: the final configuration is scored on test only once"
    best, settings_done = best_run(ABLATION_DIR / variant)
    assert settings_done == 30, f"{variant} has {settings_done} of 30 settings"
    feature_columns = best["feature_columns"]
    log(f"final variant {variant}: best setting {best['config_index']}, validation PR-AUC {best['validation_pr_auc']:.4f}, {len(feature_columns)} features")
    model = joblib.load(ABLATION_DIR / variant / f"xgboost_{best['config_index']}.joblib")
    yv, validation_inputs, _ = load_split("validation", feature_columns)
    validation_scores = model.predict_proba(validation_inputs)[:, 1]
    assert abs(average_precision_score(yv, validation_scores) - best["validation_pr_auc"]) < 1e-3, "reloaded model must match its saved validation score"
    f1_threshold = best_f1_threshold(yv, validation_scores); floor = threshold_for_minimum_precision(yv, validation_scores)
    yt, test_inputs, is_tail = load_split("test", feature_columns)
    test_scores = model.predict_proba(test_inputs)[:, 1]                         # test touched once, here
    result = {"variant": variant, "best_parameters": best["parameters"], "best_scale_pos_weight": float(model.get_params()["scale_pos_weight"]),
              "best_iteration": best["best_iteration"],
              "validation": metrics_at_threshold(yv, validation_scores, f1_threshold), "test": metrics_at_threshold(yt, test_scores, f1_threshold),
              "precision_floor_rule": f"highest recall with validation precision >= {MINIMUM_ALARM_PRECISION}",
              "validation_precision_floor": metrics_at_threshold(yv, validation_scores, floor),
              "test_precision_floor": metrics_at_threshold(yt, test_scores, floor),
              "test_normal_days_precision_floor": metrics_at_threshold(yt[~is_tail], test_scores[~is_tail], floor),
              "test_tail_precision_floor": metrics_at_threshold(yt[is_tail], test_scores[is_tail], floor),
              "feature_importance_gain": {feature_columns[int(k[1:])]: float(v) for k, v in model.get_booster().get_score(importance_type="gain").items()}}
    save_json({"experiment": f"IBM E2 + patterns + round-2 causal features ({variant}, XGBoost, 30-setting random search)",
               "models": {"xgboost": result}}, output)
    pd.DataFrame({"label": yt, "fraud_score": test_scores, "is_tail": is_tail}).to_parquet(RESULTS_DIR / f"{FINAL_EXPERIMENT}_tuned_test_scores_xgboost.parquet")
    plot_confusion_matrix(result["test_precision_floor"], f"{FINAL_EXPERIMENT} tuned XGBoost (test)", FIGURES_DIR / f"{FINAL_EXPERIMENT}_tuned_confusion_xgboost.png")
    normal = ~is_tail; floor_normal = result["test_normal_days_precision_floor"]
    log(f"DONE {FINAL_EXPERIMENT} ({variant}): validation PR-AUC {best['validation_pr_auc']:.4f}, test PR-AUC Sept 9-10 "
        f"{average_precision_score(yt[normal], test_scores[normal]):.4f}, tail {average_precision_score(yt[is_tail], test_scores[is_tail]):.4f}, "
        f"all {result['test']['pr_auc']:.4f}, F1 all {result['test']['f1']:.3f}")
    log(f"  Sept 9-10 at the precision-floor cut: {floor_normal}")
    top = sorted(result["feature_importance_gain"].items(), key=lambda item: -item[1])[:12]
    total = sum(result["feature_importance_gain"].values())
    for name, gain in top:
        print(f"    {name:45s} {gain / total:6.1%}")


SELECTED_VARIANT = "plus_selected_windows_pass_through_pair_history_currency_format_mix"


def choose_by_rule():
    """Rule written in docs/leakage_notes.md before the ablation: the higher VALIDATION PR-AUC of
    plus_all and plus_selected (useful groups only). Test plays no part in this choice."""
    candidates = {}
    for variant in ["plus_all", SELECTED_VARIANT]:
        best, done = best_run(ABLATION_DIR / variant)
        assert done == 30, f"{variant} has {done} of 30 settings"
        candidates[variant] = best["validation_pr_auc"]
        log(f"candidate {variant}: best validation PR-AUC {best['validation_pr_auc']:.4f}")
    chosen = max(candidates, key=candidates.get)
    log(f"chosen by the validation rule: {chosen}")
    return chosen


if __name__ == "__main__":
    main(choose_by_rule() if sys.argv[1] == "auto" else sys.argv[1])
