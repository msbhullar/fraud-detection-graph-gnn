# Darshitha Gongle: Data preparation, EDA and temporal splits

## What I worked on
I prepared both datasets for every experiment: cleaning, exploratory data analysis (EDA), deciding whether to subsample, and the time-based train/validation/test splits. These decisions were written down before any model was trained, so they come from the data and not from model scores.

## PaySim preparation
- Kept only TRANSFER and CASH_OUT payments (2,770,409 rows), because PAYMENT, CASH_IN and DEBIT contain no fraud. All 8,213 fraud rows were kept.
- Kept every original column in the saved file, so the deliberately leaky "naive" experiment could use the balance columns later.
- Marked the four balance columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) and isFlaggedFraud as never-allowed model inputs, because they leak the label.
- Replaced the raw time step with hour of day as a model input, so the model can't learn simulator-specific time effects.

## Subsampling decision
I tested the account-based subsample suggested in the project brief and decided not to use it. Inside the subsample, the relationship between receiver activity and fraud reversed compared with the full data, so the subsample would have created a fake signal. Every experiment uses the full filtered data.

## EDA
- Checked whether accounts repeat, which decides how useful a graph can be. In PaySim, senders almost never send again (0.06%), while receivers often do, so the graph is mostly star-shaped around receivers.
- Checked whether the two halves of a fraud (a TRANSFER followed by a CASH_OUT) are linked through the same account. PaySim almost never links them.
- These findings led to the decision to add a second dataset with a denser graph: IBM AML HI-Small.

## IBM AML preparation
- Loaded the 5.08 million IBM payments, built a clear account ID from bank plus account number, parsed the timestamps and sorted everything in time order.
- Linked transactions to IBM's ground-truth laundering pattern file, so we know which pattern (cycle, fan-in and so on) each laundering payment belongs to.

## Temporal splits
- Every split is by time and never shuffled, so a model never trains on the future.
- PaySim: train steps 1-446, validation 447-595, test 596-743 (60/20/20 of the 743 steps).
- IBM AML: train Sept 1-6, validation Sept 7-8, test Sept 9-18. The last days (Sept 11-18) are a simulator quirk with a very high laundering share, so they are reported separately from the normal test days.

## Main files
`src/data_prep.py`, `src/ibm_data_prep.py`, `docs/eda_notes.md`, `docs/leakage_notes.md`, `data/README.md`
