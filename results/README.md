# EVALUATIONS

Everything here was written by the scripts in `src/`. Model files and score tables are left out of the repository because of their size; the metrics, figures and tests are included.

| Where | What |
|---|---|
| `results/e1_metrics.json`, `e2_metrics.json`, `e3_metrics.json` | PaySim metrics for E1 (tabular), E2 (graph features) and E3 (GraphSAGE), validation and test, with the fraud rate beside every score |
| `results/ibm_*_metrics.json` and `results/ibm_*_tuned_metrics.json` | IBM AML metrics for each experiment arm, including `test_normal_days_precision_floor` (Sept 9-10) and `test_tail_precision_floor` (Sept 11-18) |
| `results_sol/ibm_e2r_tuned_metrics.json`, `ibm_e3v2_metrics.json`, `ibm_e3v2_nograph_metrics.json` | Best IBM runs, trained on the ASU Sol cluster |
| `results_sol/ablation_validation_summary.json` | Feature-group ablation (validation only) |
| `results/*_vs_*_bootstrap.json`, `results_sol/*_bootstrap.json` | Paired bootstrap tests between experiments (200 resamples) |
| `results/naive_leakage_summary.csv` | The deliberately naive PaySim runs next to the honest runs |
| `results/ibm_test_by_*.csv` | Test results for recurring, new and cold-start accounts |
| `results/e4/` | E4 case sets, generated explanations, claim-checker outputs and shape-tagger scores |
| `results/figures/`, `results_sol/figures/` | PR curves and confusion matrices for every arm |
| `results/figures/demo/` | Figures used in the demo presentation, including Neo4j Browser screenshots of the real graph |
| `results/*_runs/`, `results_sol/*_runs/` | One JSON per tuning setting, so every search can be re-read |

Convention: the main metrics are PR-AUC and recall, always shown with the prevalence of the positive class. IBM headline numbers use the normal test days (Sept 9-10).
