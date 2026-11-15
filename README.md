# Fraud Detection with Graph Databases and Graph Neural Networks, with Checked LLM Explanations

CSE 573 Semantic Web Mining, Fall 2026, Arizona State University. Group 18, Project 10.

Team: Darshitha Gongle, Maninderjit Singh Bhullar, Mithun Menon, Nandan Reddy Jalli, Reshikesh Reddy Puttamreddy.

We compare tabular models, Neo4j graph features and GraphSAGE for fraud and money-laundering detection under one strict, time-ordered protocol, and we add LLM explanations whose every sentence is checked against the graph.

## Repository structure

| Course requirement | Folder | Contents |
|---|---|---|
| CODE | `src/` and `jobs/` | Python scripts (one per step) and batch scripts for the ASU Sol cluster |
| DATA | `data/` | How to get the two public datasets (PaySim, IBM AML) and small samples |
| EVALUATIONS | `results/` and `results_sol/` | Metrics, figures, bootstrap tests and E4 outputs |
| Docs | `docs/` | EDA notes, leakage rules, E4 design notes |
| Slides | `presentation/` | Demo presentation and project presentation |

## Experiments

- E1: tabular baselines (logistic regression, random forest, XGBoost)
- E2: Neo4j graph features added to the tabular models
- E3: GraphSAGE
- E4: LLM explanations with a sentence-level claim checker
