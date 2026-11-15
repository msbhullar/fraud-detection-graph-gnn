"""Mac runner: every model and setting for the given IBM experiments, then finalize and combine.
Usage:  python src/run_ibm_experiment.py ibm_e2b ibm_e2p        (resumable)"""
import json
import subprocess
import sys
from pathlib import Path

GRID_SIZES = {"logistic_regression": 4, "random_forest": 4, "xgboost": 6}
for experiment in sys.argv[1:]:
    runs = Path("results") / f"{experiment}_runs"
    for model, size in GRID_SIZES.items():
        for config in range(size):
            if (runs / f"{model}_{config}.json").exists():
                print(f"{experiment} {model} {config}: already done, skipping", flush=True); continue
            subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, model, str(config)], check=True)
        if not (runs / f"final_{model}.json").exists():
            subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, "finalize", model], check=True)
    subprocess.run([sys.executable, "src/ibm_train_tabular.py", experiment, "combine"], check=True)
print("ALL EXPERIMENTS DONE", flush=True)
