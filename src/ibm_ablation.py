"""IBM AML round 2: feature-group ablation on VALIDATION ONLY (rule in docs/leakage_notes.md, 2026-10-28).
Base = ibm_e2p (E1 + Neo4j + pattern features, the current best). Each variant adds new causal feature groups
from data/processed/ibm_aml/ibm_window_features.parquet. Every variant gets the SAME 30 XGBoost settings as
src/ibm_tune_xgboost.py. Only the train and validation files are loaded: test is never read here.
Run on Sol (GPU):  python src/ibm_ablation.py                       (all variants, resumable)
                   python src/ibm_ablation.py plus_windows plus_all  (chosen variants)
                   python src/ibm_ablation.py plus_selected=windows,pair_history   (base + chosen groups)
                   python src/ibm_ablation.py summary                (table from saved runs)
                   python src/ibm_ablation.py smoke                  (small subset, 2 settings, CPU: code check only)
Outputs: results/ablation_runs/<variant>/xgboost_<i>.json + .joblib, results/ablation_validation_summary.json"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import RESULTS_DIR, save_json
from src.ibm_data_prep import PROCESSED_DIR
from src.ibm_train_tabular import EXPERIMENTS, LABEL
from src.ibm_tune_xgboost import NUMBER_OF_SETTINGS, build_model, search_space
from src.ibm_window_features import (CYCLE_COLUMNS, MIX_COLUMNS, PAIR_HISTORY_COLUMNS, PASS_THROUGH_COLUMNS,
                                     WINDOW_FEATURE_COLUMNS)

BASE_EXPERIMENT = "ibm_e2p"
BASE_FEATURE_COLUMNS = EXPERIMENTS[BASE_EXPERIMENT]["features"]
WINDOW_FEATURES_FILE = PROCESSED_DIR / "ibm_window_features.parquet"
FEATURE_GROUPS = {"windows": WINDOW_FEATURE_COLUMNS, "pass_through": PASS_THROUGH_COLUMNS,
                  "pair_history": PAIR_HISTORY_COLUMNS, "currency_format_mix": MIX_COLUMNS, "ordered_cycles": CYCLE_COLUMNS}
VARIANTS = {f"plus_{name}": [name] for name in FEATURE_GROUPS}
VARIANTS["plus_all"] = list(FEATURE_GROUPS)
ABLATION_DIR = RESULTS_DIR / "ablation_runs"
SPLITS = ["train", "validation"]                 # test is deliberately not listed


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_train_and_validation(new_columns, row_limit=None):
    """Base ibm_e2p columns + the chosen new columns, joined on transaction_id. Train and validation only."""
    window_features = pd.read_parquet(WINDOW_FEATURES_FILE, columns=["transaction_id"] + new_columns) if new_columns else None
    feature_columns = BASE_FEATURE_COLUMNS + new_columns
    labels, inputs = {}, {}
    for name in SPLITS:
        frame = pd.read_parquet(PROCESSED_DIR / f"{EXPERIMENTS[BASE_EXPERIMENT]['files']}_{name}.parquet",
                                columns=["transaction_id", LABEL] + BASE_FEATURE_COLUMNS)
        if row_limit:
            frame = frame.iloc[:row_limit]
        if new_columns:
            rows_before = len(frame)
            frame = frame.merge(window_features, on="transaction_id", how="left")
            assert len(frame) == rows_before and frame[new_columns].notna().all().all(), "every row needs its new features"
        labels[name] = frame[LABEL].to_numpy()
        inputs[name] = frame[feature_columns].to_numpy(dtype=np.float32)
        log(f"  {name}: shape {inputs[name].shape}, laundering {int(labels[name].sum()):,}")
        if name == "validation":
            print(frame[["transaction_id", LABEL] + feature_columns[-4:]].head(3).to_string())
        del frame
    class_ratio = float((labels["train"] == 0).sum() / (labels["train"] == 1).sum())
    return labels, inputs, class_ratio, feature_columns


def run_variant(variant, device="cuda", settings_to_run=NUMBER_OF_SETTINGS, row_limit=None, runs_root=ABLATION_DIR):
    new_columns = [column for group in VARIANTS[variant] for column in FEATURE_GROUPS[group]]
    log(f"{variant}: base {len(BASE_FEATURE_COLUMNS)} + {len(new_columns)} new = {len(BASE_FEATURE_COLUMNS) + len(new_columns)} features")
    labels, inputs, class_ratio, feature_columns = load_train_and_validation(new_columns, row_limit)
    runs_dir = runs_root / variant; runs_dir.mkdir(parents=True, exist_ok=True)
    for index, setting in enumerate(search_space()[:settings_to_run]):
        done = runs_dir / f"xgboost_{index}.json"
        if done.exists():
            continue
        start = time.time()
        model = build_model(setting, class_ratio); model.set_params(device=device)
        model.fit(inputs["train"], labels["train"], eval_set=[(inputs["validation"], labels["validation"])], verbose=False)
        validation_pr_auc = float(average_precision_score(labels["validation"], model.predict_proba(inputs["validation"])[:, 1]))
        save_json({"variant": variant, "config_index": index, "parameters": setting, "feature_columns": feature_columns,
                   "validation_pr_auc": validation_pr_auc, "best_iteration": int(model.best_iteration),
                   "seconds": round(time.time() - start, 1)}, done)
        joblib.dump(model, runs_dir / f"xgboost_{index}.joblib")
        log(f"{variant} setting {index:2d}: weight {model.get_params()['scale_pos_weight']:8.1f} depth {setting['max_depth']:2d} "
            f"lr {setting['learning_rate']:.3f} -> validation PR-AUC {validation_pr_auc:.4f} ({time.time() - start:.1f}s, {model.best_iteration} rounds)")


def best_run(runs_dir):
    runs = [json.loads(path.read_text()) for path in sorted(runs_dir.glob("xgboost_*.json"))]
    return (max(runs, key=lambda run: run["validation_pr_auc"]), len(runs)) if runs else (None, 0)


def summary():
    """Validation-only ablation table. The base comes from the existing tuned ibm_e2p runs (same 30 settings)."""
    base, base_runs = best_run(RESULTS_DIR / f"{BASE_EXPERIMENT}_tuned_runs")
    rows = [{"variant": "base (ibm_e2p)", "features": len(BASE_FEATURE_COLUMNS), "settings_done": base_runs,
             "best_validation_pr_auc": base["validation_pr_auc"], "gain_vs_base": 0.0, "best_setting": base["config_index"]}]
    for runs_dir in sorted(path for path in ABLATION_DIR.glob("plus_*") if path.is_dir()):
        variant = runs_dir.name
        best, done = best_run(runs_dir)
        if best:
            rows.append({"variant": variant, "features": len(best["feature_columns"]), "settings_done": done,
                         "best_validation_pr_auc": best["validation_pr_auc"],
                         "gain_vs_base": best["validation_pr_auc"] - base["validation_pr_auc"], "best_setting": best["config_index"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_json({"note": "validation PR-AUC only (Sept 7-8); test not scored", "rows": rows}, RESULTS_DIR / "ablation_validation_summary.json")


def smoke():
    """Code check only: 200,000 train rows, all validation rows, 2 settings, CPU, separate folder."""
    run_variant("plus_all", device="cpu", settings_to_run=2, row_limit=200_000, runs_root=RESULTS_DIR / "ablation_smoke_runs")
    log("SMOKE OK")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["smoke"]:
        smoke()
    elif arguments == ["summary"]:
        summary()
    else:
        for variant in (arguments or list(VARIANTS)):
            if variant.startswith("plus_selected="):   # e.g. plus_selected=windows,pair_history (chosen from the validation table)
                groups = variant.split("=")[1].split(",")
                assert all(group in FEATURE_GROUPS for group in groups), groups
                variant = "plus_selected_" + "_".join(groups); VARIANTS[variant] = groups
            run_variant(variant)
        summary()
        log("ALL ABLATION RUNS DONE")
