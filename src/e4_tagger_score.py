"""E4 Option 2, part 3: score tagger arms against the IBM pattern ground truth (grading only; never shown to the model).
Rule fixed in docs/e4_typology_selection_rule.md BEFORE any result: report (a) cases with >= 2 earlier pattern members visible in the case text,
(b) all pattern-matched cases, always with the visibility share. Extra decisions written 2026-11-13, before any tagger result:
 - Truth RANDOM has no rule, so it is reported separately and is NOT in the hit-rate denominators ("scorable" truth = the 7 shapes).
 - hit = the true shape is among the arm's tags; tag precision = tags equal to the truth / all tags, over scorable cases.
 - False alarms (truth none): report the share that got no tag (abstain) and the share that got a wrong tag.
Run:  python src/e4_tagger_score.py [tagger_generations.jsonl]     (the rule-only baseline is always scored)
Writes results/e4/tagger_scores.csv and tagger_scores.json"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e4_shape_rules import SHAPES   # noqa: E402

CASES = Path("results/e4/cases_typology.csv")
VISIBILITY = Path("results/e4/pattern_visibility_typology.csv")
RULE_ONLY = Path("results/e4/tagger_rule_only.jsonl")


def load_arms(generations_path: Path) -> dict:
    """arm name -> {case_id: list of tags}"""
    arms = {}
    for line in RULE_ONLY.read_text().splitlines():
        record = json.loads(line)
        arms.setdefault("rule_only", {})[record["case_id"]] = record["verified_any_shown"]
    if generations_path.exists():
        for line in generations_path.read_text().splitlines():
            record = json.loads(line)
            for kind in ("unverified", "verified_cited", "verified_any_shown"):
                arms.setdefault(f"{record['model']} | {kind}", {})[record["case_id"]] = record[kind]
    return arms


def score(frame: pd.DataFrame, tags_by_case: dict) -> dict:
    frame = frame.assign(tags=frame["case_id"].map(tags_by_case))
    frame = frame[frame["tags"].notna()]
    truth = frame["pattern_name"].fillna("none")
    scorable = frame[truth.isin(SHAPES)]
    random_cases = frame[truth == "RANDOM"]
    false_alarms = frame[truth == "none"]
    hits = sum(row.pattern_name in row.tags for row in scorable.itertuples())
    all_tags = sum(len(row.tags) for row in scorable.itertuples())
    correct_tags = sum(sum(tag == row.pattern_name for tag in row.tags) for row in scorable.itertuples())
    return {"cases_scorable": len(scorable), "hit_rate": round(hits / len(scorable), 3) if len(scorable) else None,
            "tag_precision": round(correct_tags / all_tags, 3) if all_tags else None,
            "scorable_cases_with_any_tag": round(sum(len(t) > 0 for t in scorable["tags"]) / len(scorable), 3) if len(scorable) else None,
            "random_cases": len(random_cases), "random_cases_tagged_anyway": int(sum(len(t) > 0 for t in random_cases["tags"])),
            "false_alarm_cases": len(false_alarms), "false_alarms_no_tag": int(sum(len(t) == 0 for t in false_alarms["tags"])),
            "false_alarms_with_tag": int(sum(len(t) > 0 for t in false_alarms["tags"]))}


def main():
    generations_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/e4/tagger_generations.jsonl")
    cases = pd.read_csv(CASES)
    visibility = pd.read_csv(VISIBILITY)[["case_id", "earlier_visible_in_text"]]
    frame = cases.merge(visibility, on="case_id", how="left")
    frame["earlier_visible_in_text"] = frame["earlier_visible_in_text"].fillna(0)
    matched = frame["pattern_name"].notna()
    print(f"200 cases: {int(matched.sum())} pattern-matched true (prevalence of pattern cases {matched.mean():.0%}), {int((~matched).sum())} false alarms")
    print("pattern names:", frame.loc[matched, "pattern_name"].value_counts().to_dict())
    visible = matched & (frame["earlier_visible_in_text"] >= 2)
    print(f"visible (>= 2 earlier members in the case text): {int(visible.sum())} of {int(matched.sum())} pattern-matched ({visible.sum() / matched.sum():.0%})\n")
    subsets = {"(a) visible >= 2": frame[visible | ~matched], "(b) all pattern-matched": frame}
    rows = []
    for arm, tags_by_case in load_arms(generations_path).items():
        for subset_name, subset in subsets.items():
            rows.append({"arm": arm, "subset": subset_name, **score(subset, tags_by_case)})
    result = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(result.to_string(index=False))
    result.to_csv("results/e4/tagger_scores.csv", index=False)
    Path("results/e4/tagger_scores.json").write_text(json.dumps(rows, indent=1))
    print("\nsaved results/e4/tagger_scores.csv and tagger_scores.json")


if __name__ == "__main__":
    main()
