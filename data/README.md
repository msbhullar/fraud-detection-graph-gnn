# DATA

The full data files are large (PaySim 471 MB, IBM AML 476 MB, and several GB of derived feature tables), so they are not stored in this repository. Both datasets are public.

## 1. Download

| Dataset | Source | Put the file here |
|---|---|---|
| PaySim | Kaggle, `ealaxi/paysim1` (https://www.kaggle.com/datasets/ealaxi/paysim1) | `data/raw/PS_20174392719_1491204439457_log.csv` |
| IBM AML, HI-Small | Kaggle, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | `data/raw/ibm_aml/HI-Small_Trans.csv` and `data/raw/ibm_aml/HI-Small_Patterns.txt` |

`HI-Small_Patterns.txt` lists the laundering attempts of the IBM simulator. We use it only to grade and to draw figures, never as a model input.

## 2. Samples

`data/samples/` holds the first 1,000 rows of each file so you can see the format without downloading anything. They are samples only; no result in this repository was computed from them.

- `paysim_first_1000_rows.csv`: columns `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.
- `ibm_hi_small_first_1000_rows.csv`: columns `Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`.

## 3. Preprocessing (done by the scripts in `src/`)

1. PaySim: keep only TRANSFER and CASH_OUT, the only types with fraud. IBM AML: add clear account ids and time columns.
2. Split by time (PaySim by step, IBM AML by whole days). Never shuffle.
3. Original column names are never renamed; new columns get new descriptive names.
4. Build past-only history features for each account.
5. Load accounts and payments into Neo4j. Fraud labels are never stored in the graph.

Derived tables are written to `data/processed/` (ignored by git) and can be rebuilt with the commands in the main [README](../README.md).
