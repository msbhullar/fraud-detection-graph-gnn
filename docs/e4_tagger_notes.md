# E4 Option 2 (shape tagger): dated notes (new file, 2026-11-13)

## Design (agreed with Maninder 2026-11-13, before any tagger result)
- Rules in `src/e4_shape_rules.py`, frozen thresholds: 3 distinct partners for FAN-OUT / FAN-IN, 2 for every other shape. Every shape must include the flagged payment A -> B. Self-loops are ignored. RANDOM has no rule (the tagger must say none). Up to 2 tags per case.
- Rule-only baseline priority (fixed): CYCLE, SCATTER-GATHER, GATHER-SCATTER, STACK, BIPARTITE, FAN-IN, FAN-OUT; at most 2 tags.
- Scoring (`src/e4_tagger_score.py`): hit = true shape among the arm's tags; RANDOM truth reported separately; always shown for (a) cases with >= 2 earlier pattern members visible in the case text and (b) all pattern-matched cases; false alarms reported separately.
- Prompt and settings were NOT changed after the 3-case test (those cases belong to the 200 scored cases).

## Bug found and fixed 2026-11-13 (after the first full run, found while drawing figures)
- SCATTER-GATHER rule accepted source == destination (A pays X and B, who both pay A back). That is a loop, not scatter-gather. Fix: source and destination must be different accounts. Self-test case added.
- Effect (re-verified offline from the stored raw model outputs, `src/e4_tagger_reverify.py`; pre-fix files kept in `results/e4/before_sg_fix/`): 1 of 400 model records changed its verification result. Rule-only, visible cases: unchanged (hit rate 0.612, tag precision 0.526). Rule-only, all pattern-matched: hit rate 0.275 -> 0.268, precision 0.376 -> 0.366. qwen2.5:3b verified-any-shown, all pattern-matched: precision 0.225 -> 0.231. No conclusion changed.
- The fix was a conformity fix to the written definition ("one source, middle accounts, one common destination"), made without looking at labels for the decision.

## Results (200 typology cases; 150 pattern-matched true, 50 false alarms; 51 of 150 visible)
- Visible cases (n=49 scorable): rule-only hit 61%, tag precision 53%, 14/50 false alarms tagged. qwen2.5:3b verified on cited edges: hit 16%, precision 44%, 2/50 false alarms tagged. qwen2.5:3b unverified: hit 18%, 48/50 false alarms tagged. llama3.2:3b: 0% hit (answers CYCLE for almost every case, cites only FLAG), 50/50 false alarms tagged before verification.
- Reading: the 3B models collapse to a default label; plain rules beat them; verification removes almost all wrong tags on false alarms but also most correct ones. The LLM adds nothing over code for shape tagging in this setup.
- Visibility limit: only 2-account loops ever appear in the visible case text (top-10 edges per hop hide longer cycles); STACK is 26 of 150 true cases and is essentially never visible.
- Parked: larger-model tagger run on Sol (qwen2.5:32b, llama3.3:70b), to be decided on campus.
