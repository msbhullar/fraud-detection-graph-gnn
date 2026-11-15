# E4 typology case set: selection rule (written 2026-11-12, BEFORE any typology result exists)

Why: the original 50 E4 cases have only 31 pattern-matched true cases, and in only 15 of them is enough of the pattern visible in the case text
(results/e4/pattern_visibility.csv). Too small to score a pattern tagger. This second set is larger.

Rule (fixed now, not tuned on labels or on any typology result):
1. Same pool as the original E4 cases: IBM test transactions on Sept 9-10 (normal days) with ibm_e2r XGBoost score >= the frozen
   precision-floor threshold from validation (0.1198).
2. Exclude the 50 original E4 cases (by transaction_id).
3. True laundering cases = transactions in that pool that are members of an attempt in HI-Small_Patterns.txt (match on timestamp, sender,
   receiver, amount). Draw 150 at random (seed 42), or all of them if fewer exist.
4. False alarms = transactions in that pool with Is Laundering = 0. Draw 50 at random (seed 42).
5. Case text, trimming rules, and fact rules are exactly those of src/e4_subgraph_v2.py. They are NOT changed after looking at which patterns
   are visible (that would tune on test labels).
6. Ground truth for typology = the pattern name of the attempt that contains the transaction ("none" for false alarms). Grading only; never shown to the LLM.
7. Report accuracy twice: (a) on cases where at least 2 earlier members of the same attempt are visible in the case text, (b) on all pattern-matched
   cases. Always state the visibility share. Pool with the original 31 matched cases only as a separate, labelled line.
