# EDA notes and design decisions

Written 2026-10-25, BEFORE any model was trained. These decisions come from data analysis, not from model scores.

## 1. Filter
- Kept TRANSFER and CASH_OUT only: 2,770,409 rows, all 8,213 fraud rows kept (4,097 TRANSFER, 4,116 CASH_OUT).
- Fraud prevalence after filter: 0.2965%. PAYMENT, CASH_IN, DEBIT contain zero fraud.
- Saved with all original columns: data/processed/transfer_cashout_all_columns.parquet

## 2. Do accounts repeat? (brief section 4, first EDA question)
| | unique | seen more than once |
|---|---|---|
| nameOrig (senders) | 2,768,630 | 0.06% |
| nameDest (receivers) | 509,565 | 74.82% (max 75) |
| fraud senders | 8,213 | 0.19% |
| fraud receivers | 8,169 | 62.03% |

- Only 686 accounts appear as both sender and receiver.
- Fraud TRANSFER destination later acting as a CASH_OUT origin: 1 of 4,097. PaySim does not link the two halves of a fraud.
- Conclusion: senders almost never repeat, so the graph is mostly star-shaped around receivers. Graph signal lives on the receiver side.

## 3. Decision: two datasets
- Run E1, E2, E3 on PaySim with a strict temporal split.
- Committed now, regardless of PaySim results: rerun E1, E2, E3 on IBM AML HI-Small (Altman et al., NeurIPS 2023), which has repeat accounts and multi-account laundering patterns. This is the proposal's fallback.
- E4 last. Dataset for E4 to be decided (IBM AML recommended, because PaySim subgraphs are too sparse to test explanation faithfulness).
- E1 includes receiver-side history counts that a non-graph model can compute, so any E2 gain comes from graph structure, not from E1 missing obvious features.

## 4. Supporting literature
- Sharma (2026), arXiv 2607.19266: on PaySim, after removing the balance shortcut, graph features did not improve Average Precision (0.768 vs 0.761). Graph features helped only on injected multi-account rings. Used a random split and full-graph (transductive) features; names temporal evaluation as the key next step. Our project does that.
- Mohawesh et al. (2026), IPM 104860: GraphSAGE + DistilBERT reports gains on PaySim, but applies SMOTE before the split and reports accuracy as a headline. Cited with caution.
- Altman et al. (2023): IBM AML datasets; 60/20/20 temporal split with graph features from past transactions only. Same leakage-safe design as ours.
