"""E4 step 3: generate short grounded explanations with two local models via Ollama (inference only).
Run from the project root, on the Mac (Ollama must be running: brew services start ollama):
    python src/e4_generate.py test   # 3 cases x 2 models -> results/e4/generations_test.jsonl, prints the outputs
    python src/e4_generate.py full   # 50 cases x 2 models -> results/e4/generations.jsonl (resumable)
The model sees ONLY the case text from results/e4/subgraph_cases.jsonl (past-only subgraph + rule facts, no labels).
Settings are fixed for reproducibility: temperature 0, seed 42."""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODELS = ["llama3.2:3b", "qwen2.5:3b"]   # qwen3:4b was dropped: it wrote out its reasoning even with thinking off
CASES_PATH = Path("results/e4/subgraph_cases.jsonl")
FULL_OUTPUT_PATH = Path("results/e4/generations.jsonl")
TEST_OUTPUT_PATH = Path("results/e4/generations_test.jsonl")
TEST_CASE_POSITIONS = [0, 3, 6]   # E4-001 (hub account), E4-004 (self-transfer), E4-007 (money flows both ways)
GENERATION_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 250}

SYSTEM_PROMPT = """You rewrite rule-based facts about a flagged payment as plain English. You are a strict translator, not an investigator.

Rules:
1. Write exactly one line for each label in the REQUIRED LABELS list, in that order.
2. Each line has the form  LABEL: <one short plain sentence that restates that fact line>
   Use only the facts and numbers on that same fact line. Copy numbers exactly.
3. Do not add outside knowledge, motives, guesses, advice or ranking words (strongest, main, most important).
4. Do not say that the payment is or is not money laundering.
5. Refer to accounts only as A, B, H1-xx or H2-xx.
6. Output only the lines. No introduction, no notes, no extra text."""


def required_labels(case: dict) -> list:
    return ["F1"] + list(case["risk_signals"])


def build_user_prompt(case: dict) -> str:
    labels = ", ".join(required_labels(case))
    return f"CASE TEXT:\n{case['text']}\n\nREQUIRED LABELS: {labels}\n\nWrite the lines now."


LINE_PATTERN = re.compile(r"^\s*\**\[?(F\d+)\]?\**(?:\s*\[?F\d+\]?)*\s*[:\-.)]?\s*(.+?)\s*$")


def clean_output(raw_output: str, case: dict) -> dict:
    """Keep only 'LABEL: sentence' lines whose label is required. Code (not the model) appends the [LABEL] citation."""
    wanted = required_labels(case)
    kept = {}
    dropped_lines = []
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        match = LINE_PATTERN.match(line)
        if match and match.group(1) in wanted and match.group(1) not in kept and "RISK SIGNALS" not in match.group(2):
            sentence = re.sub(r"\s*\[F\d+\]\s*", " ", match.group(2)).strip()   # remove any citation the model typed itself
            kept[match.group(1)] = sentence
        else:
            dropped_lines.append(line.strip())
    clean_sentences = [{"label": label, "sentence": kept[label]} for label in wanted if label in kept]
    explanation_lines = [f"{item['sentence']} [{item['label']}]" for item in clean_sentences]
    if not case["risk_signals"]:
        explanation_lines.append("No rule-based signal fired for this payment.")   # added by code, not by the model
    return {"clean_sentences": clean_sentences, "explanation": "\n".join(explanation_lines),
            "missing_labels": [label for label in wanted if label not in kept], "dropped_lines": dropped_lines}


def ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def generate(model: str, user_prompt: str) -> dict:
    payload = {"model": model, "stream": False, "options": GENERATION_OPTIONS,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]}
    if model.startswith("qwen3"):
        payload["think"] = False          # no hidden reasoning text, short direct answer
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    output = re.sub(r"<think>.*?</think>", "", body["message"]["content"], flags=re.DOTALL).strip()
    return {"output": output, "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count"),
            "seconds": round(time.time() - start, 1)}


def load_cases() -> list:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines()]


def run(cases: list, output_path: Path, resumable: bool):
    if not ollama_is_running():
        sys.exit("Ollama is not running. Start it with: brew services start ollama")
    done = set()
    if resumable and output_path.exists():
        done = {(r["case_id"], r["model"]) for r in map(json.loads, output_path.read_text().splitlines())}
    elif output_path.exists():
        output_path.unlink()
    for model in MODELS:                              # one model at a time keeps memory low on the 8 GB Mac
        for case in cases:
            if (case["case_id"], model) in done:
                continue
            user_prompt = build_user_prompt(case)
            result = generate(model, user_prompt)
            record = {"case_id": case["case_id"], "model": model, "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
                      "case_text": case["text"], "trimmed": case["trimmed"], "required_labels": required_labels(case),
                      "raw_output": result["output"], **clean_output(result["output"], case), **{k: v for k, v in result.items() if k != "output"}}
            with output_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(f"\n[{model}] {case['case_id']}  ({result['prompt_tokens']} prompt tokens, {result['output_tokens']} output tokens, {result['seconds']}s)")
            print(record["explanation"], flush=True)
            if record["missing_labels"] or record["dropped_lines"]:
                print(f"   (missing: {record['missing_labels']}; dropped lines: {len(record['dropped_lines'])})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    all_cases = load_cases()
    if mode == "test":
        run([all_cases[position] for position in TEST_CASE_POSITIONS], TEST_OUTPUT_PATH, resumable=False)
    elif mode == "full":
        run(all_cases, FULL_OUTPUT_PATH, resumable=True)
    else:
        sys.exit("usage: python src/e4_generate.py <test|full>")
