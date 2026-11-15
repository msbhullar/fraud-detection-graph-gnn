"""E4 arm A2: schema-constrained translator (step 3 of the 2026-11-12 plan).
The model must return a JSON object with ONE sentence for EVERY required fact label (F1, the fired risk signals, and all graph facts).
The label is the JSON key, so a model cannot skip a required fact, cite a fact that does not exist, or add chatter (Ollama enforces the
JSON schema while it generates). The schema guarantees FORMAT only, not truth: the checker (src/e4_checker.py) tests the sentences.
Code appends the [F#] citation from the key. No examples are in the prompt (the old models copied them word for word).
Run from the project root, on the Mac (Ollama running: brew services start ollama; Docker stopped on the 8 GB Mac):
    python src/e4_generate_v2.py test                     # 3 cases x 2 models -> results/e4/generations_v2_test.jsonl
    python src/e4_generate_v2.py full                     # 50 cases x 2 models -> results/e4/generations_v2.jsonl (resumable)
    python src/e4_generate_v2.py full qwen3:32b           # same cases with another model (e.g. on Sol); appends to the same file
Settings fixed for reproducibility: temperature 0, seed 42."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# On the Mac Ollama listens on localhost. On Sol the ollama module sets OLLAMA_HOST (e.g. http://sg020:11434), so use it when present.
_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else "http://" + _host
DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:3b"]
CASES_PATH = Path("results/e4/subgraph_cases_v2.jsonl")
FULL_OUTPUT_PATH = Path("results/e4/generations_v2.jsonl")
TEST_OUTPUT_PATH = Path("results/e4/generations_v2_test.jsonl")
TEST_CASE_POSITIONS = [0, 3, 6]   # E4-001 (hub account), E4-004 (self-transfer), E4-007 (money flows both ways)
GENERATION_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 700}   # up to ~13 sentences

SYSTEM_PROMPT = """You rewrite rule-based facts about a flagged payment as plain English. You are a strict translator, not an investigator.

Rules:
1. Return a JSON object. It has one key for each label in the REQUIRED LABELS list, and each value is one short plain sentence.
2. Each sentence restates only the fact line with the same label in the case text. Use only the words, accounts and numbers on that line. Copy numbers exactly.
3. Do not add outside knowledge, motives, guesses, advice, or ranking words (strongest, main, most important).
4. Do not say that the payment is or is not money laundering.
5. Refer to accounts only as A, B, H1-xx or H2-xx."""


def required_labels(case: dict) -> list:
    """F1, then every fired risk signal, then every graph fact; each label once, in numeric order."""
    labels = {"F1"} | set(case["risk_signals"]) | set(case["graph_fact_labels"])
    return sorted(labels, key=lambda label: int(label[1:]))


def build_schema(labels: list) -> dict:
    return {"type": "object", "properties": {label: {"type": "string"} for label in labels},
            "required": labels, "additionalProperties": False}


def build_user_prompt(case: dict) -> str:
    labels = ", ".join(required_labels(case))
    return f"CASE TEXT:\n{case['text']}\n\nREQUIRED LABELS: {labels}\n\nReturn the JSON object now."


def clean_output(raw_output: str, case: dict) -> dict:
    """Parse the JSON. Code (not the model) appends each [F#] citation from the key."""
    wanted = required_labels(case)
    try:
        parsed = json.loads(raw_output)
        parse_ok = isinstance(parsed, dict)
    except json.JSONDecodeError:
        parsed, parse_ok = {}, False
    parsed = parsed if parse_ok else {}
    clean_sentences = []
    for label in wanted:
        sentence = parsed.get(label)
        if isinstance(sentence, str) and sentence.strip():
            sentence = re.sub(rf"^\s*\[?{label}\]?\s*[:\-.)]?\s*", "", sentence.strip())   # drop a label the model typed inside the value
            clean_sentences.append({"label": label, "sentence": sentence.strip()})
    explanation_lines = [f"{item['sentence']} [{item['label']}]" for item in clean_sentences]
    if not case["risk_signals"]:
        explanation_lines.insert(1, "No risk-signal rule fired for this payment.")   # added by code, not by the model
    present = {item["label"] for item in clean_sentences}
    return {"parse_ok": parse_ok, "clean_sentences": clean_sentences, "explanation": "\n".join(explanation_lines),
            "missing_labels": [label for label in wanted if label not in present],
            "extra_keys": [key for key in parsed if key not in wanted]}


def ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def generate(model: str, case: dict) -> dict:
    payload = {"model": model, "stream": False, "options": GENERATION_OPTIONS, "format": build_schema(required_labels(case)),
               "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(case)}]}
    if model.startswith("qwen3"):
        payload["think"] = False
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.load(response)
    output = re.sub(r"<think>.*?</think>", "", body["message"]["content"], flags=re.DOTALL).strip()
    return {"output": output, "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count"),
            "done_reason": body.get("done_reason"), "seconds": round(time.time() - start, 1)}


def load_cases() -> list:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines()]


def run(cases: list, models: list, output_path: Path, resumable: bool):
    if not ollama_is_running():
        sys.exit("Ollama is not running. Start it with: brew services start ollama")
    done = set()
    if resumable and output_path.exists():
        done = {(r["case_id"], r["model"]) for r in map(json.loads, output_path.read_text().splitlines())}
    elif output_path.exists():
        output_path.unlink()
    for model in models:                              # one model at a time keeps memory low on the 8 GB Mac
        for case in cases:
            if (case["case_id"], model) in done:
                continue
            result = generate(model, case)
            labels = required_labels(case)
            record = {"case_id": case["case_id"], "model": model, "arm": "A2_schema_translator", "system_prompt": SYSTEM_PROMPT,
                      "user_prompt": build_user_prompt(case), "case_text": case["text"], "trimmed": case["trimmed"],
                      "required_labels": labels, "fact_lines": {f"F{n}": text for n, text in enumerate(case["facts"], start=1)},
                      "raw_output": result["output"], **clean_output(result["output"], case),
                      **{k: v for k, v in result.items() if k != "output"}}
            with output_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(f"\n[{model}] {case['case_id']}  ({result['prompt_tokens']} prompt tokens, {result['output_tokens']} output tokens, {result['seconds']}s, done: {result['done_reason']})")
            print(record["explanation"], flush=True)
            if not record["parse_ok"] or record["missing_labels"] or record["extra_keys"]:
                print(f"   (parse_ok: {record['parse_ok']}; missing: {record['missing_labels']}; extra keys: {record['extra_keys']})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    chosen_models = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT_MODELS
    all_cases = load_cases()
    if mode == "test":
        run([all_cases[position] for position in TEST_CASE_POSITIONS], chosen_models, TEST_OUTPUT_PATH, resumable=False)
    elif mode == "full":
        run(all_cases, chosen_models, FULL_OUTPUT_PATH, resumable=True)
    else:
        sys.exit("usage: python src/e4_generate_v2.py <test|full> [model1,model2]")
