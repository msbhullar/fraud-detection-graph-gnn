# Reshikesh Reddy Puttamreddy: LLM explanation layer (E4)

## What I worked on
I worked on the explanation layer (E4). Its job is to turn a flagged payment into a short card an analyst can trust, without the language model inventing facts.

## How it works
1. **Neighbourhood:** for each flagged payment, we pull its past-only, 2-hop neighbourhood from Neo4j.
2. **Fact lines:** plain rules compute numbered fact lines from that neighbourhood (for example [F6]). The rules, not the model, decide what is true.
3. **Rewriting:** a small local LLM (run with Ollama) only rewrites those facts into readable sentences. A JSON schema constrains its output, it runs at temperature 0 with a fixed seed, and every sentence must cite the fact line it came from.
4. **Checking:** a claim checker written in code compares each sentence with its cited fact. It checks numbers, account names, currencies and negations, for example "no" or "never" being added or dropped. A sentence that fails is shown in red as flagged, never as true.

## Evaluation setup
- We ran four open models (llama3.2:3b, qwen2.5:3b, qwen2.5:32b and llama3.3:70b) on the same 50 flagged cases, and every sentence went through the claim checker.
- We also built a shape tagger that names the laundering pattern (cycle, fan-in, fan-out, scatter-gather and so on) and compared an LLM-based tagger against plain rules.

## Main files
`src/e4_select_cases.py`, `src/e4_subgraph_v2.py`, `src/e4_generate_v2.py`, `src/e4_checker.py`, `src/e4_tagger.py`, `src/e4_tagger_score.py`, `results/e4/`, `docs/e4_tagger_notes.md`
