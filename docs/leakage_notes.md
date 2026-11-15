# Leakage notes

## Temporal split (decided 2026-10-25, before training)
- Data: TRANSFER and CASH_OUT, all 2,770,409 rows. No subsample: the brief's account-based subsample was tested and reversed the real receiver-activity vs fraud relationship (fraud receivers 5.7 vs legit 11.6 in full data; 5.7 vs 4.5 inside the subsample), so it would create a fake signal.
- Split by step, never shuffled: train steps 1-446, validation 447-595, test 596-743.
- PaySim normal activity collapses after about step 400 while fraud stays steady, so fraud prevalence differs by split. Prevalence is reported beside every score.

## Columns never used as model inputs in E1-E3
- oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest (label leakage), isFlaggedFraud (simulator rule).
- Raw step: test steps are never seen in training, and late steps carry a simulator-driven fraud rate jump. Hour of day (step mod 24) is used instead. Step is still used for ordering and for the split.
- nameOrig, nameDest as raw IDs (only used to compute past-only history counts).

## Fitting rules
- Scalers, encoders, thresholds and tuning: training or validation only. Test is touched once at the end.

## Decision thresholds (fixed 2026-10-25, after E1, applied unchanged to E1, E2, E3)
Both are chosen on validation and frozen before test.
1. Best-F1 cutoff (balanced).
2. Precision-floor cutoff: the highest recall where at least 50% of validation alarms are real fraud.
   Reason: a missed fraud costs more than reviewing an alarm, but fewer than half real alarms is not efficient.
   The 50% floor was chosen from this reasoning, not by reading the test curve.
PR-AUC (threshold-free) stays the main comparison metric.

## E3b run log (2026-10-25)
- First full run was invalid: a smoke-test result (small subset, 2 epochs) was saved in results/e3_runs and was reused as config 0. Moved to results/e3_old_invalid/.
- In that run, config 1's validation PR-AUC was still rising at the 20-epoch cap (best = last epoch). Based on the validation curve only, MAX_EPOCHS was raised to 40 (early stopping patience 4 unchanged). Grid, split, sampling and thresholds unchanged.
- That run's test PR-AUC (0.662) was seen; it is kept for transparency and was not used for any decision.

# IBM AML HI-Small (dataset 2)

## Data and split (decided 2026-10-25, before any IBM model was trained)
- HI-Small_Trans.csv: 5,078,345 transactions, Sept 1-18 2022, 5,177 laundering (0.102%). All rows kept (decision: no rows dropped, same as Altman et al. 2023).
- Known simulator tail: Sept 11-18 has 1,108 transactions of which 655 are laundering (about 59%), because normal activity stops while started laundering patterns finish.
- Split by whole days, never shuffled: train Sept 1-6, validation Sept 7-8, test Sept 9-18.
- Test metrics are also reported separately for Sept 9-10 and Sept 11-18 so the effect of the tail is visible. This sub-report is never used to choose models or thresholds.
- Model inputs never include raw dates or account ids. Thresholds chosen on validation only (same two rules as PaySim).
- Precision-floor rule when unreachable: if no validation cutoff gives at least 50% real alarms, the model gets no cutoff under that rule (threshold = infinity, 0 alarms, recall 0) and this is reported as "not reachable". The rule is never relaxed. (Added 2026-10-25 when IBM E1 logistic regression could not reach 50% precision on validation.)
- IBM XGBoost round limit raised from 600 to 2,000 for BOTH E1 and E2 (2026-10-26). Reason (validation only): the selected E2 setting used all 600 rounds without early stopping, so the cap may have limited tuning. Early stopping (50 rounds on validation PR-AUC), grids, split and thresholds unchanged. The 600-round results are archived in results/ibm_xgboost_cap600_archive/ (E2 test PR-AUC on normal days 0.291 vs E1 0.193 there).

## IBM E3b GraphSAGE (2026-10-26)
- Same design as PaySim E3b: 2-layer GraphSAGE mean aggregator, 10 then 5 most recent neighbor transactions with minute < target minute, plain PyTorch, edge classification, E1 features as transaction inputs. Neighbor messages: log amount (USD), payment format, hours ago, direction.
- Early-stopping monitor uses all validation laundering + a fixed 25% random sample of validation legit rows (seed 42), for speed only. The setting choice, both thresholds and all reported metrics use the full validation and test sets.
- Account-group breakdown (professor comment 2), IBM test Sept 9-10: 862,727 of 862,792 transactions involve an account seen in training; all 956 laundering cases are "recurring" under the brief's definition, so new/cold-start groups contain no laundering. Finer split (sender/receiver seen in training): both known 922 laundering, receiver new 15, sender new 19, both new 0. New-account groups are too small for conclusions. Saved in results/ibm_test_by_account_group.csv and results/ibm_test_by_sender_receiver_known.csv.

## IBM E2b pattern features (2026-10-26, before training)
- Our own Python version of the Graph Feature Preprocessor (Blanuša et al., ICAIF'24; Snap ML has no Apple Silicon build). Streaming in time order; a transaction at minute t uses only transactions from earlier minutes inside a 1-day window (6 hours for scatter-gather paths). No labels.
- Per sender and receiver: fan-out, fan-in (distinct counterparties), sent/received counts, amount sum/mean/std, minutes since last sent/received. Per transaction: prior same-pair count, reverse-pair count (2-cycles), v->w->u paths (3-cycles, order not enforced), u->w->v paths within 6 h (scatter-gather). Neighbor-set intersections capped at 3,000 for hub accounts.
- Verified by brute-force recomputation on 500 random transactions (0 mismatches).
- Experiments: ibm_e2b = E1 + pattern features; ibm_e2p = E1 + Neo4j graph features + pattern features. Same models, grids (XGBoost 2,000-round cap), split and thresholds as E1/E2.

## IBM XGBoost random-search tuning (decided 2026-10-26, before running)
- Reason: scale_pos_weight was fixed at the train class ratio (about 1,280, the XGBoost default suggestion); the GFP paper tuned it in 1-10. Decision: let validation choose it together with other hyperparameters.
- Same 30 random settings (seed 2026) for ibm_e1, ibm_e2, ibm_e2p, ibm_e2b, so every experiment gets an equal search budget. Setting 0 = previous best-style setting with the class-ratio weight.
- Space: scale_pos_weight log-uniform 1-1,500; max_depth 4-12; learning_rate log 0.01-0.2; min_child_weight log 1-100; reg_lambda log 0.01-100; subsample and colsample_bytree 0.5-1.0; up to 2,000 rounds, early stopping 50 on validation PR-AUC.
- Selection by validation PR-AUC; both thresholds frozen on validation; test scored once. Results saved as <experiment>_tuned_*; earlier results kept unchanged.

## Selection rule for the next feature and model rounds (decided 2026-10-28, before any new feature is built)
- Applies to the new causal feature groups (1 h / 6 h / 7 d windows, pass-through ratio, time since last pair payment, currency/format mix, longer cycles), the feature-group ablation, LightGBM, and the GNN upgrade on Sol.
- Which feature groups are kept, which model is chosen, all hyperparameters and both thresholds are decided on validation PR-AUC (Sept 7-8) only.
- Test (Sept 9-18) is not scored during these rounds. Each final configuration is scored on test once, at the end, and all test numbers are reported whether better or worse.
- The ablation table is built from validation PR-AUC. Its test column is filled once, after the final configuration is frozen.
- Same 30-setting random search (seed 2026) for every new experiment, so search budgets stay equal. Any change to this budget is written here with its validation-only reason before running.
- The test-period results already seen (E1, E2, E2b, E2+patterns tuned, Mac and Sol) are not used to choose new feature groups.

## Feature-group ablation decision rule (decided 2026-10-28, before the ablation runs)
- Base = tuned ibm_e2p (E1 + Neo4j + pattern features), best validation PR-AUC from its existing 30 tuned runs.
- Variants (src/ibm_ablation.py): base + one new group each (windows, pass_through, pair_history, currency_format_mix, ordered_cycles), and base + all groups. Same 30 settings (seed 2026). Only train and validation files are loaded.
- A group counts as useful if base + that group beats the base by at least 0.005 validation PR-AUC. Reason: the Mac and Sol runs of the same tuning differed by up to 0.003, so smaller gains may be run-to-run noise.
- Final candidate = whichever is higher on validation: base + all groups, or base + only the useful groups (run as plus_selected). If neither beats the base by 0.005, the base stays the final model and the new features are reported as not helping.
- Only the final candidate is scored on test, once.

## Round 2 final result (2026-10-29)
- Variant chosen on validation only by the rule above: plus_all (0.6515) over plus_selected (0.6504). Saved as experiment ibm_e2r.
- Test scored once (Sol job 66922772): Sept 9-10 PR-AUC 0.594, all test 0.756, tail 0.980. No further choices are made from these test numbers.

## E3 v2 GraphSAGE (2026-10-29)
- Setting chosen on full validation PR-AUC only (setting 0, 0.4723). The no-graph twin uses the same setting. Blend weight chosen on validation only (0.0). Test scored once in jobs/e3v2_finalize.sh; no choice was made from test numbers.
- The early-stopping monitor (all validation laundering + 25% of validation legit, fixed seed) is used for speed only; reported validation numbers use the full validation set.
