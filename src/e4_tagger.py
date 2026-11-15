"""E4 Option 2 (typology tagger), part 2: the LLM tagger and the rule-only baseline.
The model reads the SAME case text as the translator arms and names up to 2 money-movement shapes (or none), citing the edge ids that show each.
Code then checks every tag with the frozen shape rules (src/e4_shape_rules.py). The LLM is never trusted on its own.
Ground truth (pattern names) is never shown to the model and is used only by src/e4_tagger_score.py. No tuning on labels.
Run from the project root (Ollama running on the Mac, Docker stopped):
    python src/e4_tagger.py rules                       # rule-only baseline on the 200 typology cases (no LLM) -> results/e4/tagger_rule_only.jsonl
    python src/e4_tagger.py prompt                      # print the system prompt, one user prompt and its schema (no run)
    python src/e4_tagger.py test                        # 3 cases x 2 models -> results/e4/tagger_test.jsonl
    python src/e4_tagger.py full [model1,model2]        # 200 cases x 2 models -> results/e4/tagger_generations.jsonl (resumable)
Settings fixed: temperature 0, seed 42, num_ctx 4096."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e4_shape_rules import SHAPES, edge_ids, parse_case, rule_only_tags, shape_holds   # noqa: E402

_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else "http://" + _host
DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:3b"]
CASES_PATH = Path("results/e4/subgraph_cases_typology_v2.jsonl")
RULE_ONLY_PATH = Path("results/e4/tagger_rule_only.jsonl")
FULL_OUTPUT_PATH = Path("results/e4/tagger_generations.jsonl")
TEST_OUTPUT_PATH = Path("results/e4/tagger_test.jsonl")
TEST_CASE_POSITIONS = [9, 32, 3]   # TY-010 (CYCLE), TY-033 (FAN-OUT), TY-004 (SCATTER-GATHER)
GENERATION_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 300}

SYSTEM_PROMPT = """You label the SHAPE of the money movement around one flagged payment. You are a careful reader, not an investigator.

Use only the edge lines (E01, E02, ...) and the flagged payment, which has the id FLAG. Ignore the risk score and the other facts.
Edges where an account pays itself are not part of any shape.

Shapes (all must include the flagged payment A -> B):
- FAN-OUT: one account pays many different accounts. Here A pays at least 3 different accounts, B among them.
- FAN-IN: many different accounts pay one account. Here at least 3 different accounts pay B, A among them.
- GATHER-SCATTER: money is collected into a middle account, then sent out to several accounts. The middle account (A or B) receives from at least 2 different accounts and pays at least 2 different accounts, and some payment in is earlier than some payment out.
- SCATTER-GATHER: one account pays at least 2 middle accounts, and those middle accounts then all pay the same destination account. The flagged payment is one of those legs.
- CYCLE: money goes round a loop and comes back to where it started. A path of earlier payments leads from B back to A, each payment later than the one before, and the flagged payment A -> B closes the loop.
- BIPARTITE: a group of at least 2 senders each pay the same group of at least 2 receivers, and no account is in both groups.
- STACK: two bipartite groups chained, the receivers of the first group being the senders of the second.

Rules:
1. Return a JSON object with a list "tags" of 1 or 2 items. Each item has a "shape" and the "edge_ids" that show it (include FLAG).
2. If no shape is clearly shown by the edge lines, return one item with shape "none" and an empty edge_ids list. Most payments show no shape; do not guess.
3. Cite only edge ids that exist in the case text."""


def build_schema(case_edge_ids: list) -> dict:
    tag = {"type": "object", "properties": {"shape": {"type": "string", "enum": SHAPES + ["none"]},
                                            "edge_ids": {"type": "array", "items": {"type": "string", "enum": case_edge_ids}}},
           "required": ["shape", "edge_ids"], "additionalProperties": False}
    return {"type": "object", "properties": {"tags": {"type": "array", "items": tag, "minItems": 1, "maxItems": 2}},
            "required": ["tags"], "additionalProperties": False}


def build_user_prompt(case: dict) -> str:
    return f"CASE TEXT:\n{case['text']}\n\nReturn the JSON object now."


def verify_output(raw_output: str, case: dict) -> dict:
    """Parse the model JSON and check each tag with the frozen shape rules, three ways:
    unverified = what the model said; verified_cited = rule holds on the CITED edges + flagged payment (primary);
    verified_any_shown = rule holds on ALL shown edges + flagged payment (looser)."""
    parsed_case = parse_case(case["text"])
    valid_ids = set(edge_ids(parsed_case))
    try:
        tags = json.loads(raw_output).get("tags", [])
        parse_ok = isinstance(tags, list)
    except (json.JSONDecodeError, AttributeError):
        tags, parse_ok = [], False
    unverified, verified_cited, verified_any_shown, bad_ids = [], [], [], 0
    for tag in tags if parse_ok else []:
        shape = tag.get("shape") if isinstance(tag, dict) else None
        cited = [i for i in tag.get("edge_ids", []) if isinstance(i, str)] if isinstance(tag, dict) else []
        if shape not in SHAPES:
            continue
        bad_ids += sum(1 for i in cited if i not in valid_ids)
        if shape not in unverified:
            unverified.append(shape)
        if shape not in verified_cited and shape_holds(shape, parsed_case, allowed_ids=set(cited) - {"FLAG"}):
            verified_cited.append(shape)
        if shape not in verified_any_shown and shape_holds(shape, parsed_case):
            verified_any_shown.append(shape)
    return {"parse_ok": parse_ok, "tags": tags if parse_ok else [], "unverified": unverified, "verified_cited": verified_cited,
            "verified_any_shown": verified_any_shown, "cited_ids_not_in_case": bad_ids}


def ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def generate(model: str, case: dict) -> dict:
    schema = build_schema(edge_ids(parse_case(case["text"])))
    payload = {"model": model, "stream": False, "options": GENERATION_OPTIONS, "format": schema,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(case)}]}
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.load(response)
    return {"output": body["message"]["content"].strip(), "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count"),
            "done_reason": body.get("done_reason"), "seconds": round(time.time() - start, 1)}


def load_cases() -> list:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines()]


def run_rule_only(cases: list):
    with RULE_ONLY_PATH.open("w") as file:
        for case in cases:
            parsed_case = parse_case(case["text"])
            tags = rule_only_tags(parsed_case)
            file.write(json.dumps({"case_id": case["case_id"], "arm": "rule_only", "unverified": tags, "verified_cited": tags,
                                   "verified_any_shown": tags, "n_edges_shown": len(parsed_case["edges"])}) + "\n")
    print("saved:", RULE_ONLY_PATH)


def run_llm(cases: list, models: list, output_path: Path, resumable: bool):
    if not ollama_is_running():
        sys.exit("Ollama is not running. Start it with: brew services start ollama")
    done = set()
    if resumable and output_path.exists():
        done = {(r["case_id"], r["model"]) for r in map(json.loads, output_path.read_text().splitlines())}
    elif output_path.exists():
        output_path.unlink()
    for model in models:
        for case in cases:
            if (case["case_id"], model) in done:
                continue
            result = generate(model, case)
            record = {"case_id": case["case_id"], "model": model, "arm": "tagger", "raw_output": result["output"],
                      **verify_output(result["output"], case), **{k: v for k, v in result.items() if k != "output"}}
            with output_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(f"[{model}] {case['case_id']} {result['seconds']}s  model said: {record['unverified'] or 'none'}  "
                  f"verified(cited): {record['verified_cited'] or '-'}  verified(any shown): {record['verified_any_shown'] or '-'}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT_MODELS
    cases = load_cases()
    if mode == "rules":
        run_rule_only(cases)
    elif mode == "prompt":
        print(SYSTEM_PROMPT, "\n\n--- schema for the first case ---\n", json.dumps(build_schema(edge_ids(parse_case(cases[0]["text"]))), indent=1)[:900])
        print("\n--- user prompt (TY-033) ---\n", build_user_prompt(next(c for c in cases if c["case_id"] == "TY-033")))
    elif mode == "test":
        run_llm([cases[i] for i in TEST_CASE_POSITIONS], models, TEST_OUTPUT_PATH, resumable=False)
    elif mode == "full":
        run_llm(cases, models, FULL_OUTPUT_PATH, resumable=True)
    else:
        sys.exit("usage: python src/e4_tagger.py <rules|prompt|test|full> [model1,model2]")
