"""E4 claim checker (step 4 of the 2026-11-12 plan) + no-LLM template baseline (arm A0).
Every sentence is compared with the fact line it cites (fact lines are read from the saved case text). Automatic tests:
  1 numbers      every number (and 'once', 'twice', 'two' ...) in the sentence must appear on the cited fact line
  2 accounts     A, B, H1-xx, H2-xx in the sentence must appear on the cited fact line
  3 names        currency and payment-format names in the sentence must appear on the cited fact line
  4 wording      judgement / motive words (suspicious, likely, unusual, ...) that are not on the fact line are errors
  5 polarity     a negation (no, never, not) or NEW must not appear or disappear
  6 inference    comparison words (above, below, higher ...) not on the line: 'true inference' only if code can verify it (F1 score vs threshold),
                 otherwise 'unverified inference' (counted as unsupported, listed for hand-check)
A sentence is SUPPORTED if no test fires (a verified true inference is still supported, but tagged). The checker cannot see every meaning
change, so every flagged sentence plus a random 20% of passed sentences go into a hand-check CSV.
Commands (project root):
    python src/e4_checker.py selftest                      # plants known errors, must catch all and pass the true sentences
    python src/e4_checker.py template                      # arm A0: fact lines used as the sentences -> results/e4/generations_template_v2.jsonl
    python src/e4_checker.py check <generations.jsonl> <arm_name>   # sentence table, summary json, hand-check csv"""
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NUMBER_WORDS = {"once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal",
              "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"]
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
FORBIDDEN_WORDS = {"suspicious", "suspect", "fraud", "fraudulent", "laundering", "launder", "illegal", "criminal", "likely", "unlikely", "probably",
                   "indicates", "indicating", "suggests", "suggesting", "evidence", "risky", "unusual", "unusually", "anomalous", "abnormal",
                   "strongest", "main", "important", "concerning", "alarming", "significant", "significantly", "typical", "normal", "concern", "red"}
INFERENCE_WORDS = {"above", "below", "exceeds", "exceeded", "higher", "lower", "greater", "less", "more", "fewer", "larger", "smaller", "increased"}
NEGATIONS = {"no", "never", "not", "none", "neither", "nobody"}
ALIAS_PATTERN = re.compile(r"(?<![\w-])(H[12]-\d+|A|B)(?![\w-])")
ARTICLE_FOLLOWERS = {"payment", "new", "transaction", "community", "total", "different", "single", "self-transfer", "small", "large", "number"}


def words_of(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def numbers_in(text: str) -> list:
    text = re.sub(r"(?<![\w-])H[12]-\d+(?![\w-])", " ", text)           # alias digits are not numbers
    found = [float(item.replace(",", "")) for item in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
    found += [float(NUMBER_WORDS[word]) for word in words_of(text) if word in NUMBER_WORDS]
    return found


def aliases_in(text: str) -> set:
    found = set()
    for match in ALIAS_PATTERN.finditer(text):
        token = match.group(1)
        if token == "A" and match.start() == 0:                         # sentence-initial "A payment ..." is an article
            next_word = re.match(r"\s*([\w-]+)", text[match.end():])
            if next_word and next_word.group(1).lower() in ARTICLE_FOLLOWERS:
                continue
        found.add(token)
    return found


def names_in(text: str) -> set:
    return {name for name in CURRENCIES + FORMATS if re.search(rf"\b{re.escape(name)}\b", text)}


def has_negation_word(text: str) -> bool:
    return any(word in NEGATIONS for word in words_of(text))


def has_zero_count(text: str) -> bool:
    return 0.0 in numbers_in(text)


def check_sentence(sentence: str, fact_line: str) -> dict:
    flags = []
    fact_numbers = numbers_in(fact_line) + ([0.0] if any(word in {"no", "none", "never"} for word in words_of(fact_line)) else [])
    sentence_numbers = numbers_in(sentence)
    unmatched = [number for number in sentence_numbers if not any(abs(number - known) < 1e-9 for known in fact_numbers)]
    if unmatched:
        flags.append("number_not_on_fact_line")
    if aliases_in(sentence) - aliases_in(fact_line):
        flags.append("account_not_on_fact_line")
    if names_in(sentence) - names_in(fact_line):
        flags.append("name_not_on_fact_line")
    fact_words, sentence_words = set(words_of(fact_line)), set(words_of(sentence))
    if (sentence_words & FORBIDDEN_WORDS) - fact_words:
        flags.append("unsupported_wording")
    # '0 payments' and 'no payments' mean the same. Flag only a real flip: a negation ADDED where the fact has none (word or zero),
    # or a negation word on the fact line (never / no / not) that the sentence drops. Leaving out a '... sent 0 other payments' clause is
    # an omission, not a false claim; it is measured by numbers_recall instead.
    fact_has_word, fact_has_zero = has_negation_word(fact_line), has_zero_count(fact_line)
    if has_negation_word(sentence) and not (fact_has_word or fact_has_zero):
        flags.append("negation_changed")
    elif fact_has_word and not (has_negation_word(sentence) or has_zero_count(sentence)):
        flags.append("negation_changed")
    if ("NEW" in fact_line) != bool(re.search(r"\bnew\b|\bfirst\b", sentence, flags=re.IGNORECASE)) and "NEW" in fact_line:
        flags.append("new_dropped")
    inference_tag = None
    extra_inference = (sentence_words & INFERENCE_WORDS) - fact_words
    if extra_inference:
        score_match = re.search(r"risk score of (\d+\.\d+); the alert threshold is (\d+\.\d+)", fact_line)
        wording_up = bool(extra_inference & {"above", "exceeds", "exceeded", "higher", "greater"})
        wording_down = bool(extra_inference & {"below", "lower", "less", "fewer"})
        if score_match and (wording_up != wording_down):
            score, threshold = float(score_match.group(1)), float(score_match.group(2))
            inference_tag = "inference_true" if ((score > threshold) == wording_up) else "inference_false"
        else:
            inference_tag = "inference_unverified"
        if inference_tag != "inference_true":
            flags.append(inference_tag)
    normalized_sentence, normalized_fact = " ".join(words_of(sentence)), " ".join(words_of(fact_line))
    fact_core = re.sub(r"^f\d+ ", "", normalized_fact)
    return {"supported": not flags, "flags": flags, "inference_tag": inference_tag if inference_tag == "inference_true" else None,
            "numbers_total": len(sentence_numbers), "numbers_matched": len(sentence_numbers) - len(unmatched),
            "fact_numbers_total": len(fact_numbers), "fact_numbers_kept": sum(any(abs(number - known) < 1e-9 for known in sentence_numbers) for number in fact_numbers),
            "identical_to_fact_line": normalized_sentence == fact_core,
            "similarity_to_fact_line": round(SequenceMatcher(None, normalized_sentence, fact_core).ratio(), 3)}


def fact_lines_from_text(case_text: str) -> dict:
    return {f"F{match.group(1)}": match.group(2) for match in re.finditer(r"^F(\d+) (.*)$", case_text, flags=re.MULTILINE)}


def run_selftest():
    fact = "F8 This payment is about 1,887 USD, 11.9x A's median earlier payment (159 USD)."
    true_ok = ["This payment is about 1,887 USD, which is 11.9 times A's median earlier payment of 159 USD."]
    planted = {
        "wrong number": "This payment is about 1,887 USD, 19.9x A's median earlier payment (159 USD).",
        "invented account": "This payment is about 1,887 USD, 11.9x H1-05's median earlier payment (159 USD).",
        "judgement word": "This payment is about 1,887 USD, an unusual 11.9x A's median earlier payment (159 USD).",
        "invented currency": "This payment is about 1,887 Euro, 11.9x A's median earlier payment (159 USD).",
        "invented inference": "This payment is about 1,887 USD, 11.9x higher than A's median earlier payment (159 USD).",
    }
    for sentence in true_ok:
        verdict = check_sentence(sentence, fact)
        assert verdict["supported"], f"true sentence wrongly flagged: {verdict}"
    for name, sentence in planted.items():
        verdict = check_sentence(sentence, fact)
        assert not verdict["supported"], f"planted error '{name}' was NOT caught"
    assert check_sentence("A received 2 payment(s) in the 6 h before.", "F9 In the 6 h before, A received 2 payment(s) and sent 0 other payment(s).")["supported"], "omitting a zero clause wrongly flagged"
    assert not check_sentence("A did not pay B before.", "F2 A paid B 4 time(s) before; the last was 83.6 h ago.")["supported"], "added negation not caught"
    assert check_sentence("A received payments and sent no other payments.", "F9 In the 6 h before, A received 2 payment(s) and sent 0 other payment(s).")["supported"], "'no' versus 0 wrongly flagged"
    new_fact = "F7 ACH is a NEW payment format for A."
    assert not check_sentence("ACH is a payment format for A.", new_fact)["supported"], "dropped NEW not caught"
    never_fact = "F3 B has never paid A before."
    assert not check_sentence("B has paid A before.", never_fact)["supported"], "dropped 'never' not caught"
    score_fact = "F1 The model gave this payment a risk score of 0.18; the alert threshold is 0.12."
    assert check_sentence("The risk score of 0.18 is above the alert threshold of 0.12.", score_fact)["inference_tag"] == "inference_true"
    assert not check_sentence("The risk score of 0.18 is below the alert threshold of 0.12.", score_fact)["supported"], "false inference not caught"
    print(f"SELFTEST PASSED: {len(planted) + 5} planted errors caught, 'no' = 0 accepted, omissions not flagged, correct sentences (incl. a true inference) pass")


def run_template():
    from src.e4_generate_v2 import required_labels
    cases = [json.loads(line) for line in Path("results/e4/subgraph_cases_v2.jsonl").read_text().splitlines()]
    output_path = Path("results/e4/generations_template_v2.jsonl")
    with output_path.open("w") as file:
        for case in cases:
            labels = required_labels(case)
            sentences = [{"label": label, "sentence": case["facts"][int(label[1:]) - 1]} for label in labels]
            file.write(json.dumps({"case_id": case["case_id"], "model": "template", "arm": "A0_template", "case_text": case["text"],
                                   "trimmed": case["trimmed"], "required_labels": labels, "clean_sentences": sentences,
                                   "missing_labels": []}) + "\n")
    print(f"saved {output_path}: {len(cases)} cases (the fact lines themselves, no LLM)")


def bootstrap_interval(per_case: pd.DataFrame, column: str, resamples: int = 1000, seed: int = 42):
    generator = np.random.default_rng(seed)
    case_ids = per_case["case_id"].unique()
    grouped = {case_id: group for case_id, group in per_case.groupby("case_id")}
    values = []
    for _ in range(resamples):
        picked = generator.choice(case_ids, size=len(case_ids), replace=True)
        pooled = pd.concat([grouped[case_id] for case_id in picked])
        values.append(pooled[column].mean())
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_check(generations_path: str, arm_name: str):
    records = [json.loads(line) for line in Path(generations_path).read_text().splitlines()]
    rows = []
    for record in records:
        facts = fact_lines_from_text(record["case_text"])
        for item in record["clean_sentences"]:
            verdict = check_sentence(item["sentence"], facts[item["label"]])
            rows.append({"case_id": record["case_id"], "model": record["model"], "label": item["label"], "sentence": item["sentence"],
                         "fact_line": facts[item["label"]], "flags": ";".join(verdict["flags"]), **{k: v for k, v in verdict.items() if k != "flags"}})
    sentences = pd.DataFrame(rows)
    sentences["unsupported"] = (~sentences["supported"]).astype(int)
    sentences["supported_int"] = sentences["supported"].astype(int)
    sentences["identical_int"] = sentences["identical_to_fact_line"].astype(int)
    sentences.to_csv(f"results/e4/checks_{arm_name}_sentences.csv", index=False)
    summary = {}
    for model, group in sentences.groupby("model"):
        model_records = [r for r in records if r["model"] == model]
        required_total = sum(len(r["required_labels"]) for r in model_records)
        missing_total = sum(len(r["missing_labels"]) for r in model_records)
        low, high = bootstrap_interval(group, "supported_int")
        summary[model] = {
            "cases": len(model_records), "sentences": int(len(group)),
            "supported_claim_rate": round(group["supported_int"].mean(), 4), "supported_rate_95ci_over_cases": [round(low, 4), round(high, 4)],
            "hallucination_rate": round(group["unsupported"].mean(), 4),
            "numeric_accuracy": round(group["numbers_matched"].sum() / max(1, group["numbers_total"].sum()), 4),
            "numbers_recall_detail_kept": round(group["fact_numbers_kept"].sum() / max(1, group["fact_numbers_total"].sum()), 4),
            "true_inference_sentences": int(group["inference_tag"].notna().sum()),
            "coverage_of_required_labels": round(1 - missing_total / max(1, required_total), 4),
            "share_identical_to_fact_line": round(group["identical_int"].mean(), 4),
            "mean_similarity_to_fact_line": round(group["similarity_to_fact_line"].mean(), 4),
            "truncation_rate_cases": round(float(np.mean([r["trimmed"] for r in model_records])), 4),
            "flag_counts": group["flags"][group["flags"] != ""].str.split(";").explode().value_counts().to_dict()}
        if summary[model]["hallucination_rate"] == 0:
            summary[model]["rule_of_three_upper_bound"] = round(3 / len(group), 4)
    Path(f"results/e4/checks_{arm_name}_summary.json").write_text(json.dumps(summary, indent=2))
    flagged = sentences[~sentences["supported"] | sentences["inference_tag"].notna()]
    passed = sentences[sentences["supported"] & sentences["inference_tag"].isna()]
    sample = passed.sample(frac=0.2, random_state=42) if len(passed) else passed
    hand = pd.concat([flagged.assign(reason_in_hand_check="flagged_by_checker"), sample.assign(reason_in_hand_check="random_20_percent_of_passed")])
    hand[["case_id", "model", "label", "sentence", "fact_line", "flags", "reason_in_hand_check"]].assign(
        hand_verdict_supported_yes_no="", hand_note="").to_csv(f"results/e4/hand_check_{arm_name}.csv", index=False)
    print(f"arm {arm_name}: {len(sentences)} sentences from {len(records)} generations")
    print(pd.DataFrame(summary).drop(index=["flag_counts"]).to_string())
    for model, values in summary.items():
        print(f"flags {model}: {values['flag_counts']}")
    print(f"hand-check rows written: {len(hand)} (flagged {len(flagged)} + random sample {len(sample)})")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "selftest":
        run_selftest()
    elif command == "template":
        run_template()
    elif command == "check" and len(sys.argv) == 4:
        run_check(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: python src/e4_checker.py <selftest | template | check <generations.jsonl> <arm_name>>")
