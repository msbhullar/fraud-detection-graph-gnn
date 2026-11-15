"""Final IBM AML figures from SAVED scores and metrics (trains nothing). Uses the Sol result files in results_sol/
for every model so all curves come from the same platform. Run from the project root:
    python src/ibm_final_figures.py
Outputs go to results/figures/ (existing figures are not touched)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import FIGURES_DIR, plot_confusion_matrix, plot_pr_curves

SOL_DIR = Path("results_sol")
MODELS = {  # legend name -> score file (progression from E1 to the best model, then the GNN)
    "E1 tabular (tuned XGBoost)": "ibm_e1_tuned_test_scores_xgboost",
    "E2 + Neo4j graph features": "ibm_e2_tuned_test_scores_xgboost",
    "E2b + pattern features": "ibm_e2b_tuned_test_scores_xgboost",
    "E2 + patterns": "ibm_e2p_tuned_test_scores_xgboost",
    "E2r + round-2 causal features (best)": "ibm_e2r_tuned_test_scores_xgboost",
    "E3 v2 GraphSAGE": "ibm_e3v2_test_scores_graphsage",
}


def load_scores(file_stem: str) -> pd.DataFrame:
    return pd.read_parquet(SOL_DIR / f"{file_stem}.parquet")


def pr_curve_figures():
    scores = {name: load_scores(stem) for name, stem in MODELS.items()}
    first = next(iter(scores.values()))
    for name, frame in scores.items():
        assert len(frame) == len(first) and (frame["label"].values == first["label"].values).all(), f"rows differ for {name}"
        assert (frame["is_tail"].values == first["is_tail"].values).all(), f"tail flag differs for {name}"
    normal_rows = ~first["is_tail"].values
    for suffix, mask, title in [
        ("normal_days", normal_rows, "IBM AML test, Sept 9-10 (normal days)"),
        ("all_test", normal_rows | True, "IBM AML test, Sept 9-18 (includes simulator tail)")]:
        curves = {name: (frame["label"].values[mask], frame["fraud_score"].values[mask]) for name, frame in scores.items()}
        prevalence = float(first["label"].values[mask].mean())
        plot_pr_curves(curves, prevalence, title, FIGURES_DIR / f"ibm_final_pr_curves_{suffix}.png")
        print(f"saved ibm_final_pr_curves_{suffix}.png | rows {int(mask.sum()):,}, laundering {int(first['label'].values[mask].sum()):,}, prevalence {prevalence:.4%}")


def confusion_figures():
    jobs = [("e2r", "ibm_e2r_tuned_metrics.json", "xgboost", "XGBoost E2r"),
            ("e3v2", "ibm_e3v2_metrics.json", "graphsage", "GraphSAGE v2")]
    for short_name, file_name, model_key, label in jobs:
        result = json.loads((SOL_DIR / file_name).read_text())["models"][model_key]
        for key, suffix, cut_description in [("test_normal_days_precision_floor", "normal_days_floor", "Sept 9-10, precision-floor cut"),
                                             ("test", "all_test_best_f1", "all test, best-F1 cut")]:
            metrics = result[key]
            plot_confusion_matrix(metrics, f"{label}: {cut_description}", FIGURES_DIR / f"ibm_final_{short_name}_confusion_{suffix}.png")
            matrix = metrics["confusion_matrix"]
            print(f"saved ibm_final_{short_name}_confusion_{suffix}.png | caught {matrix['true_positive']:,}, missed {matrix['false_negative']:,}, "
                  f"false alarms {matrix['false_positive']:,}, precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}")


if __name__ == "__main__":
    pr_curve_figures()
    confusion_figures()
