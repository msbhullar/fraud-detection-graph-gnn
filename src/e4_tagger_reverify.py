"""Re-check stored tagger outputs with the CURRENT shape rules (offline, no LLM). Written 2026-11-13 after the SCATTER-GATHER fix
(source and destination must differ). Reads the raw model outputs, recomputes verified_cited / verified_any_shown, rewrites
results/e4/tagger_generations.jsonl and the rule-only file. The pre-fix files are kept in results/e4/before_sg_fix/.
Run:  python src/e4_tagger_reverify.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e4_tagger import CASES_PATH, FULL_OUTPUT_PATH, run_rule_only, verify_output   # noqa: E402

cases = {json.loads(l)["case_id"]: json.loads(l) for l in CASES_PATH.read_text().splitlines()}
before_path = Path("results/e4/before_sg_fix/tagger_generations.jsonl")
rows, changed = [], 0
for line in before_path.read_text().splitlines():
    record = json.loads(line)
    new = verify_output(record["raw_output"], cases[record["case_id"]])
    changed += (new["verified_cited"], new["verified_any_shown"]) != (record["verified_cited"], record["verified_any_shown"])
    rows.append({**record, **new})
FULL_OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"re-verified {len(rows)} model outputs; {changed} records changed verification result")
run_rule_only(list(cases.values()))
