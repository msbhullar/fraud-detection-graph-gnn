"""Robustness rerun (Mac): all XGBoost settings for IBM E1 and E2 with the 2,000-round limit, then finalize and combine.
Usage (inside Build_Project, .venv active):  python src/run_ibm_xgboost.py"""
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")
for experiment in ["ibm_e1", "ibm_e2"]:
    for config_index in range(6):
        if (RESULTS / f"{experiment}_runs" / f"xgboost_{config_index}.json").exists():
            print(f"{experiment} xgboost {config_index}: already done, skipping", flush=True); continue
        subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, "xgboost", str(config_index)], check=True)
    subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, "finalize", "xgboost"], check=True)
    subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, "combine"], check=True)
print("ALL XGBOOST RERUNS DONE", flush=True)
