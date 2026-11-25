# Nandan Reddy Jalli: GraphSAGE (E3)

## What I worked on
I built the graph neural network experiment (E3). The goal was to test whether a model that learns from the transaction graph directly can beat tabular models given hand-made graph features.

## The model
- Two-layer GraphSAGE with a mean aggregator, written in plain PyTorch, plus a small MLP head that classifies each payment (edge classification).
- Each account's representation is built by averaging the messages from its neighbours. A message carries the log amount, the payment format, how many hours ago the payment happened, and its direction (sent or received).
- Neighbour sampling is time-aware: a payment only sees neighbours from earlier payments, so the model never looks into the future.
- PaySim: samples the 10, then 5, most recent neighbours, with hidden size 32, learning rate 0.003 and dropout 0.2.
- IBM AML (v2, trained on the ASU Sol GPU cluster): samples the 20, then 10, most recent neighbours, with hidden size 128 and learning rate 0.001, and uses the same inputs as our best XGBoost model.
- Settings and the alert threshold are chosen on validation only, and the test set is scored once.

## Measuring what the graph adds
I also trained a twin model with the same network but all graph neighbours removed, so any difference between the two comes from the graph itself. I compared the twins with a paired bootstrap test, and I also tested blending GraphSAGE scores into our XGBoost model.

## Main files
`src/graphsage.py`, `src/ibm_graphsage.py`, `src/ibm_graphsage_v2.py`, `jobs/e3v2_train.sh`, `results/e3_metrics.json`, `results_sol/ibm_e3v2_metrics.json`
