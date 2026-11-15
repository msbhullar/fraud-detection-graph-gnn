"""E4 Option 2 (typology tagger), part 1: the shape rules. Pure code, no LLM, no labels.
Frozen 2026-11-13, BEFORE any tagger result exists (thresholds agreed with Maninder: 3 for fans, 2 for the rest).

Input = the edge lines that are SHOWN in a case text (HOP-1 / HOP-2 lines, past-only) plus the flagged payment A -> B (id FLAG).
Self-loops (X -> X, for example Reinvestment) are dropped: they are not a transfer between two accounts.
Every shape rule must INCLUDE the flagged payment, because the tag is about the flagged payment.
The flagged payment is the latest edge of all (the others are earlier), so a cycle can only be closed by it.

Shapes: FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, BIPARTITE, STACK. RANDOM has no rule (the tagger must say none).
Fixed priority for the rule-only baseline (more specific first): CYCLE, SCATTER-GATHER, GATHER-SCATTER, STACK, BIPARTITE, FAN-IN, FAN-OUT; at most 2 tags.
Run:  python src/e4_shape_rules.py selftest      (planted positive and negative examples)
"""
import re
import sys

SHAPES = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "BIPARTITE", "STACK"]
PRIORITY = ["CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "BIPARTITE", "FAN-IN", "FAN-OUT"]
FAN_MIN = 3        # distinct partners for FAN-OUT / FAN-IN
OTHER_MIN = 2      # distinct partners for every other shape
MAX_CYCLE_PATH = 5  # path B -> ... -> A of at most 5 earlier edges (cycle of at most 6 edges with the flagged payment)
EDGE_LINE = re.compile(r"^(E\d+) t-([\d.]+)h (\S+) -> (\S+) ")


def parse_case(case_text: str) -> dict:
    """Return {'sender': 'A', 'receiver': 'B', 'edges': [(edge_id, time, src, dst), ...]}; time = -hours before (later = bigger)."""
    flagged = re.search(r"FLAGGED PAYMENT at \S+ \S+: (\S+) -> (\S+),", case_text)
    edges = []
    for line in case_text.splitlines():
        match = EDGE_LINE.match(line)
        if match:
            edges.append((match.group(1), -float(match.group(2)), match.group(3), match.group(4)))
    return {"sender": flagged.group(1), "receiver": flagged.group(2), "edges": edges}


def _usable(parsed: dict, allowed_ids=None) -> list:
    """Edges used by the rules: shown edges (optionally only the cited ids), no self-loops, plus the flagged payment."""
    chosen = [e for e in parsed["edges"] if e[2] != e[3] and (allowed_ids is None or e[0] in allowed_ids)]
    return chosen + [("FLAG", 0.0, parsed["sender"], parsed["receiver"])]


def _out(edges):
    table = {}
    for _, _, src, dst in edges:
        table.setdefault(src, set()).add(dst)
    return table


def _in(edges):
    table = {}
    for _, _, src, dst in edges:
        table.setdefault(dst, set()).add(src)
    return table


def _fan_out(a, b, edges):
    return len(_out(edges).get(a, set())) >= FAN_MIN


def _fan_in(a, b, edges):
    return len(_in(edges).get(b, set())) >= FAN_MIN


def _gather_scatter(a, b, edges):
    for middle in (a, b):                       # the flagged payment is an outflow of A or an inflow of B
        ins = [e for e in edges if e[3] == middle]
        outs = [e for e in edges if e[2] == middle]
        if (len({e[2] for e in ins}) >= OTHER_MIN and len({e[3] for e in outs}) >= OTHER_MIN
                and min(e[1] for e in ins) < max(e[1] for e in outs)):   # some inflow is earlier than some outflow
            return True
    return False


def _scatter_gather(a, b, edges):
    out_table, in_table = _out(edges), _in(edges)
    for source, middles_of_source in out_table.items():
        for destination, payers in in_table.items():
            if source == destination:                  # 2026-11-13 fix: source and destination must be different accounts (A->X->A is a loop, not scatter-gather)
                continue
            middles = (middles_of_source & payers) - {source, destination}
            if len(middles) < OTHER_MIN:
                continue
            if (source == a and b in middles) or (destination == b and a in middles):   # flagged payment is one of the legs
                return True
    return False


def _cycle(a, b, edges):
    """Time-ordered path from B back to A (strictly increasing times, no repeated account); with A -> B it closes a loop."""
    earlier = [e for e in edges if e[0] != "FLAG"]
    def walk(account, last_time, visited, length):
        if length >= MAX_CYCLE_PATH:
            return False
        for _, time_value, src, dst in earlier:
            if src == account and time_value > last_time and dst not in visited:
                if dst == a or walk(dst, time_value, visited | {dst}, length + 1):
                    return True
        return False
    return a != b and walk(b, float("-inf"), {b}, 0)


def _block_receivers(a, s2, edges):
    out_table = _out(edges)
    return out_table.get(a, set()) & out_table.get(s2, set())


def _bipartite(a, b, edges):
    out_table = _out(edges)
    for s2 in out_table:
        if s2 == a:
            continue
        common = _block_receivers(a, s2, edges)
        if b in common and len(common) >= OTHER_MIN and not (common & {a, s2}):
            return True
    return False


def _stack(a, b, edges):
    """Two bipartite layers sharing a middle group; the flagged payment sits in layer 1 (A in S, B in M) or layer 2 (A in M, B in R)."""
    out_table, in_table = _out(edges), _in(edges)
    for s2 in out_table:                              # layer 1: flagged payment A -> B, middle group M (>= 2 accounts incl. B)
        if s2 == a:
            continue
        middle = _block_receivers(a, s2, edges)
        if b in middle and len(middle) >= OTHER_MIN and not (middle & {a, s2}):
            for m1 in middle:
                for m2 in middle:
                    if m1 < m2:
                        top = _block_receivers(m1, m2, edges)
                        if len(top) >= OTHER_MIN and not (top & ({a, s2} | middle)):
                            return True
    for m2 in out_table:                              # layer 2: flagged payment A -> B with A in M, B in R
        if m2 == a:
            continue
        top = _block_receivers(a, m2, edges)
        if b in top and len(top) >= OTHER_MIN and not (top & {a, m2}):
            feeders = in_table.get(a, set()) & in_table.get(m2, set())
            if len(feeders) >= OTHER_MIN and not (feeders & (top | {a, m2})):
                return True
    return False


RULES = {"FAN-OUT": _fan_out, "FAN-IN": _fan_in, "GATHER-SCATTER": _gather_scatter, "SCATTER-GATHER": _scatter_gather,
         "CYCLE": _cycle, "BIPARTITE": _bipartite, "STACK": _stack}


def shape_holds(shape: str, parsed: dict, allowed_ids=None) -> bool:
    """Does the shape rule hold on the shown edges (or only the cited edge ids) plus the flagged payment?"""
    if shape not in RULES or parsed["sender"] == parsed["receiver"]:
        return False
    return RULES[shape](parsed["sender"], parsed["receiver"], _usable(parsed, allowed_ids))


def rule_only_tags(parsed: dict) -> list:
    """Rule-only baseline: every shape that holds on all shown edges, first two by the fixed priority."""
    return [shape for shape in PRIORITY if shape_holds(shape, parsed)][:2]


def edge_ids(parsed: dict) -> list:
    return ["FLAG"] + [e[0] for e in parsed["edges"]]


def _selftest():
    def mk(flag, *rows):   # rows: (id, hours_before, src, dst)
        text = f"FLAGGED PAYMENT at 2022-09-09 09:02: {flag[0]} -> {flag[1]}, ACH, 1.00 US Dollar\n" + "\n".join(
            f"{i} t-{h}h {s} -> {d} ACH 1.00 US Dollar" for i, h, s, d in rows)
        return parse_case(text)
    cases = {
        "FAN-OUT": (mk(("A", "B"), ("E01", 3, "A", "X"), ("E02", 2, "A", "Y")), True),
        "FAN-OUT-neg": (mk(("A", "B"), ("E01", 3, "A", "X")), False),
        "FAN-IN": (mk(("A", "B"), ("E01", 3, "X", "B"), ("E02", 2, "Y", "B")), True),
        "GATHER-SCATTER": (mk(("A", "B"), ("E01", 5, "X", "A"), ("E02", 4, "Y", "A"), ("E03", 3, "A", "Z")), True),
        "GATHER-SCATTER-wrong-order": (mk(("A", "B"), ("E01", 1, "X", "A"), ("E02", 1, "Y", "A")), False),
        "SCATTER-GATHER": (mk(("A", "B"), ("E01", 5, "A", "X"), ("E02", 4, "X", "D"), ("E03", 3, "B", "D")), True),
        "SCATTER-GATHER-neg": (mk(("A", "B"), ("E01", 5, "A", "X"), ("E02", 4, "X", "D")), False),
        "SCATTER-GATHER-same-source-and-destination": (mk(("A", "B"), ("E01", 5, "A", "X"), ("E02", 4, "X", "A"), ("E03", 3, "B", "A")), False),
        "CYCLE": (mk(("A", "B"), ("E01", 5, "B", "X"), ("E02", 3, "X", "A")), True),
        "CYCLE-2": (mk(("A", "B"), ("E01", 5, "B", "A")), True),
        "CYCLE-wrong-time": (mk(("A", "B"), ("E01", 3, "B", "X"), ("E02", 5, "X", "A")), False),
        "BIPARTITE": (mk(("A", "B"), ("E01", 5, "A", "R2"), ("E02", 4, "S2", "B"), ("E03", 3, "S2", "R2")), True),
        "BIPARTITE-neg": (mk(("A", "B"), ("E01", 5, "A", "R2"), ("E02", 4, "S2", "B")), False),
        "STACK": (mk(("A", "B"), ("E01", 9, "A", "M2"), ("E02", 8, "S2", "B"), ("E03", 7, "S2", "M2"),
                     ("E04", 6, "B", "R1"), ("E05", 5, "B", "R2"), ("E06", 4, "M2", "R1"), ("E07", 3, "M2", "R2")), True),
        "self-loops-ignored": (mk(("A", "B"), ("E01", 3, "A", "A"), ("E02", 2, "A", "X")), False),
    }
    names = {"FAN-OUT-neg": "FAN-OUT", "GATHER-SCATTER-wrong-order": "GATHER-SCATTER", "SCATTER-GATHER-neg": "SCATTER-GATHER", "SCATTER-GATHER-same-source-and-destination": "SCATTER-GATHER",
             "CYCLE-2": "CYCLE", "CYCLE-wrong-time": "CYCLE", "BIPARTITE-neg": "BIPARTITE", "self-loops-ignored": "FAN-OUT"}
    failures = 0
    for name, (parsed, expected) in cases.items():
        shape = names.get(name, name)
        got = shape_holds(shape, parsed)
        failures += got != expected
        print(f"{'ok  ' if got == expected else 'FAIL'} {name:28s} expected {expected!s:5s} got {got}")
    parsed = mk(("A", "B"), ("E01", 5, "B", "X"), ("E02", 3, "X", "A"))
    print("cited-edges-only (cite E01 only) on the CYCLE example:", shape_holds("CYCLE", parsed, allowed_ids={"E01"}), "(expected False)")
    print("rule_only_tags on the CYCLE example:", rule_only_tags(parsed))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    sys.exit("usage: python src/e4_shape_rules.py selftest")
