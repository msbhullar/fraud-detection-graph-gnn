"""Demo-presentation figures (new file, 2026-11-13). Real data only; accounts stay aliased (A, B, H1-xx, H2-xx). No model is trained here.
Writes PNGs to results/figures/demo/. Run from the project root:  python src/demo_figures.py [shapes|neighborhood|charts|e4|all]"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from e4_shape_rules import PRIORITY, _usable, parse_case, shape_holds   # noqa: E402

OUT = Path("results/figures/demo")
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, RED, BLUE, GREEN, AMBER, LIGHT = "#1f2937", "#6b7280", "#d62839", "#2f6fb0", "#2e8b57", "#e09f3e", "#e5e7eb"
plt.rcParams.update({"font.family": "Lato", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK})
CASES = [json.loads(line) for line in Path("results/e4/subgraph_cases_typology_v2.jsonl").read_text().splitlines()]
TRUTH = pd.read_csv("results/e4/cases_typology.csv").set_index("case_id")["pattern_name"]


def _support_edges(shape, parsed):
    """Edges that make the (already verified) shape rule true, for drawing only. Mirrors the rule logic in e4_shape_rules.py."""
    a, b, edges = parsed["sender"], parsed["receiver"], _usable(parsed)
    if shape == "FAN-OUT":
        return [e for e in edges if e[2] == a]
    if shape == "FAN-IN":
        return [e for e in edges if e[3] == b]
    if shape == "GATHER-SCATTER":
        for middle in (a, b):
            ins, outs = [e for e in edges if e[3] == middle], [e for e in edges if e[2] == middle]
            if len({e[2] for e in ins}) >= 2 and len({e[3] for e in outs}) >= 2 and min(e[1] for e in ins) < max(e[1] for e in outs):
                return ins + outs
    if shape == "SCATTER-GATHER":
        for s in {e[2] for e in edges}:
            for d in {e[3] for e in edges}:
                mids = ({e[3] for e in edges if e[2] == s} & {e[2] for e in edges if e[3] == d}) - {s, d}
                if len(mids) >= 2 and ((s == a and b in mids) or (d == b and a in mids)):
                    return [e for e in edges if (e[2] == s and e[3] in mids) or (e[3] == d and e[2] in mids)]
    if shape == "CYCLE":
        earlier = [e for e in edges if e[0] != "FLAG"]
        def walk(node, last_time, path, seen):
            for e in earlier:
                if e[2] == node and e[1] > last_time and e[3] not in seen:
                    if e[3] == a:
                        return path + [e]
                    found = walk(e[3], e[1], path + [e], seen | {e[3]})
                    if found:
                        return found
            return None
        path = walk(b, float("-inf"), [], {b})
        return (path or []) + [e for e in edges if e[0] == "FLAG"]
    return edges


def _draw_graph(ax, parsed, edges, layout="spring", title=""):
    graph = nx.MultiDiGraph()
    for _, _, src, dst in edges:
        graph.add_edge(src, dst)
    if layout == "circular":
        positions = nx.circular_layout(graph)
    elif layout == "layered":                         # left-to-right by longest path, so collect/spread shapes read clearly
        simple = nx.DiGraph(graph)
        layers = {node: 0 for node in simple.nodes}
        for _ in range(len(simple)):
            for u, v in simple.edges:
                if layers[v] < len(simple):
                    layers[v] = max(layers[v], layers[u] + 1)
        nx.set_node_attributes(simple, layers, "layer")
        positions = nx.multipartite_layout(simple, subset_key="layer")
    else:
        positions = nx.spring_layout(graph, seed=3, k=2.2 / max(len(graph) ** 0.5, 1), iterations=200)
    a, b = parsed["sender"], parsed["receiver"]
    colors = [RED if n in (a, b) else (BLUE if n.startswith("H1") else (AMBER if n.startswith("H2") else MUTED)) for n in graph.nodes]
    for edge_id, _, src, dst in edges:
        flagged = edge_id == "FLAG"
        nx.draw_networkx_edges(nx.DiGraph([(src, dst)]), positions, ax=ax, edge_color=RED if flagged else "#9ca3af", width=3.2 if flagged else 1.6,
                               arrowsize=18, node_size=900, connectionstyle="arc3,rad=0.12", style="solid")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=900, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8, font_color="white", font_weight="bold")
    ax.set_title(title, fontsize=12.5, loc="left", color=INK)
    ax.margins(0.18)
    ax.axis("off")


def fig_shapes():
    chosen = {}
    for shape in ["CYCLE", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER"]:
        for case in CASES:
            parsed = parse_case(case["text"])
            if TRUTH.get(case["case_id"]) == shape and shape_holds(shape, parsed):
                chosen[shape] = (case, parsed)
                break
    print("shape examples (real flagged cases where the rule holds AND the pattern file agrees):", {s: c["case_id"] for s, (c, _) in chosen.items()})
    descriptions = {"CYCLE": "money returns to its start", "FAN-OUT": "one pays many", "FAN-IN": "many pay one",
                    "GATHER-SCATTER": "collected, then spread out", "SCATTER-GATHER": "spread out, then collected"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, shape in zip(axes.flat, chosen):
        case, parsed = chosen[shape]
        _draw_graph(ax, parsed, _support_edges(shape, parsed), "circular" if (shape == "CYCLE" and len(_support_edges(shape, parsed)) > 2) else ("layered" if shape in ("GATHER-SCATTER", "SCATTER-GATHER") else "spring"), f"{shape}: {descriptions[shape]}\n(real case {case['case_id']}" + (", short 2-account loop)" if shape == "CYCLE" else ")"))
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    for i, (label, color) in enumerate([("flagged payment (A to B)", RED), ("account A or B", RED), ("traded directly with A or B (hop 1)", BLUE)]):
        legend_ax.scatter([], [], color=color, s=140, label=label)
    legend_ax.plot([], [], color=RED, lw=3, label="flagged payment edge")
    legend_ax.plot([], [], color="#9ca3af", lw=2, label="earlier payments (past only)")
    legend_ax.legend(loc="center", frameon=False, fontsize=13, title="How to read these graphs", title_fontsize=14)
    fig.suptitle("Laundering shapes found around real flagged payments (IBM AML, Sept 9-10; accounts anonymised)", fontsize=16, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shapes_panel.png", dpi=160)
    plt.close(fig)


def fig_patterns():
    """Full laundering attempts exactly as recorded in IBM's pattern file (ground truth, NOT what the model sees; includes later payments)."""
    from e4_pattern_check import load_pattern_rows_with_attempt_ids
    rows = load_pattern_rows_with_attempt_ids()
    wanted = ["CYCLE", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "BIPARTITE", "STACK", "RANDOM"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, name in zip(axes.flat, wanted):
        attempts = rows[rows["pattern_name"] == name].groupby("pattern_attempt_number")
        sizes = attempts.apply(lambda g: g[["sender_account_id", "receiver_account_id"]].drop_duplicates().shape[0])
        usable = sizes[(sizes >= 4) & (sizes <= 12)]
        if name == "CYCLE":      # want a loop through at least 3 accounts
            usable = usable[[a for a in usable.index if len(set(rows[rows["pattern_attempt_number"] == a]["sender_account_id"])) >= 3]]
        attempt = (usable - 8).abs().sort_values(kind="stable").index[0]
        group = rows[rows["pattern_attempt_number"] == attempt].sort_values("timestamp")
        names = {}
        for account in list(group["sender_account_id"]) + list(group["receiver_account_id"]):
            names.setdefault(account, f"N{len(names) + 1}")
        graph = nx.MultiDiGraph()
        for sender, receiver in zip(group["sender_account_id"], group["receiver_account_id"]):
            graph.add_edge(names[sender], names[receiver])
        simple = nx.DiGraph(graph)
        if name == "CYCLE":
            positions = nx.circular_layout(simple)
        elif name in ("BIPARTITE", "STACK", "SCATTER-GATHER", "GATHER-SCATTER"):
            layers = {}
            for node in simple.nodes:                     # layer = longest distance from a source, so stacks and gathers read left to right
                layers[node] = 0
            for _ in range(len(simple)):
                for u, v in simple.edges:
                    layers[v] = max(layers[v], layers[u] + 1) if layers[v] < len(simple) else layers[v]
            nx.set_node_attributes(simple, layers, "layer")
            positions = nx.multipartite_layout(simple, subset_key="layer")
        else:
            positions = nx.spring_layout(simple, seed=3, k=2.0 / max(len(simple) ** 0.5, 1), iterations=200)
        nx.draw_networkx_edges(simple, positions, ax=ax, edge_color="#9ca3af", width=1.7, arrowsize=16, node_size=620, connectionstyle="arc3,rad=0.10")
        nx.draw_networkx_nodes(simple, positions, ax=ax, node_color=BLUE, node_size=620, edgecolors="white", linewidths=1.5)
        nx.draw_networkx_labels(simple, positions, ax=ax, font_size=8, font_color="white", font_weight="bold")
        ax.set_title(f"{name}\n({len(group)} payments, {len(simple)} accounts)", fontsize=13, loc="left")
        ax.margins(0.15)
        ax.axis("off")
    fig.suptitle("What the laundering shapes look like: full real attempts from IBM's pattern file (ground truth, accounts anonymised)", fontsize=16, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ground_truth_patterns.png", dpi=160)
    plt.close(fig)


def fig_neighborhood(case_id="TY-033"):
    case = next(c for c in CASES if c["case_id"] == case_id)
    parsed = parse_case(case["text"])
    edges = [e for e in _usable(parsed) if e[2] != e[3]]
    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_graph(ax, parsed, edges, "spring", f"Past-only 2-hop neighbourhood of flagged payment {case_id}: A pays B (red)")
    ax.text(0.0, -0.02, f"Shown: {len(edges) - 1} of the largest earlier payments per hop (self-payments hidden); the case text also reports how many are not shown.",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_neighborhood_TY-033.png", dpi=160)
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.01):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dy, fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=12, fontweight="bold")


def fig_charts():
    scores = pd.read_parquet("results_sol/ibm_e2r_tuned_test_scores_xgboost.parquet")
    normal = scores[~scores["is_tail"].astype(bool)]
    prevalence = normal["label"].mean()
    n_laundering, n_rows = int(normal["label"].sum()), len(normal)
    # 1. improvement chain (numbers from docs/project_progress*.md; Mac results for the first four, Sol for the last)
    steps = ["E1\nuntuned", "E1\ntuned", "E2\n+ Neo4j graph", "E2 + pattern\nfeatures", "E2 + round-2\ncausal features"]
    values = [0.229, 0.284, 0.368, 0.458, 0.594]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(steps, values, color=[LIGHT, "#9db7d5", "#6c97c4", BLUE, GREEN], edgecolor="white")
    bar_labels(ax, bars)
    ax.set_ylabel("PR-AUC (higher is better)")
    ax.set_ylim(0, 0.72)
    ax.set_title("Graph and causal features raised normal-day PR-AUC from 0.23 to 0.59", loc="left", fontsize=15)
    ax.text(0, -0.2, f"IBM AML HI-Small, test Sept 9-10, strict past-only. {n_laundering} laundering of {n_rows:,} transactions (prevalence {prevalence:.2%}); a random guess scores about {prevalence:.3f}.",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ibm_improvement_chain.png", dpi=160)
    plt.close(fig)
    # 2. leakage inflation (PaySim)
    leak = pd.read_csv("results/naive_leakage_summary.csv")
    order = {"Honest E1 XGBoost": "Honest\n(temporal split)", "Naive 1: + balance columns": "+ balance\ncolumns", "Naive 2a: random split only": "random split\nonly",
             "Naive 2b: balance columns + random split": "balance + \nrandom split", "Naive 3a: full-graph features only": "full-graph\nfeatures only", "Naive 3b: everything (brief's naive run)": "everything\n(naive run)"}
    leak = leak.set_index("run").loc[list(order)]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    bars = ax.bar([order[r] for r in leak.index], leak["pr_auc"], color=[GREEN, RED, AMBER, RED, AMBER, RED], edgecolor="white")
    bar_labels(ax, bars)
    for bar, prevalence_value in zip(bars, leak["test_fraud_prevalence"]):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"fraud rate\n{prevalence_value:.2%}", ha="center", fontsize=9.5, color="white", fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("PR-AUC")
    ax.set_title("PaySim: balance columns are the real leak (0.66 to 0.99 on the same split)", loc="left", fontsize=15)
    ax.text(0, -0.2, "Same XGBoost settings in every bar. Random splits change the fraud rate, so those bars are not comparable with the temporal ones (137 test frauds).",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paysim_leakage.png", dpi=160)
    plt.close(fig)
    # 3. graph helps on rich networks, not on sparse ones
    fig, ax = plt.subplots(figsize=(8.5, 6))
    width = 0.36
    labels = ["PaySim\n(thin, star-shaped graph)", "IBM AML\n(rich, connected graph)"]
    e1, e2 = [0.658, 0.284], [0.636, 0.368]
    bars1 = ax.bar([0 - width / 2, 1 - width / 2], e1, width, color=LIGHT, label="E1 tabular", edgecolor="white")
    bars2 = ax.bar([0 + width / 2, 1 + width / 2], e2, width, color=BLUE, label="E2 + Neo4j graph features", edgecolor="white")
    bar_labels(ax, bars1)
    bar_labels(ax, bars2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("PR-AUC")
    ax.legend(frameon=False)
    ax.set_title("Graph features help only when the network is rich", loc="left", fontsize=14)
    ax.text(0, -0.24, "XGBoost, test sets: PaySim fraud rate 3.37%; IBM Sept 9-10 fraud rate 0.11% (tuned models).\nBootstrap: IBM +0.083 [+0.069, +0.101]; PaySim -0.022 (graph features slightly worse).",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_graph_helps_when_rich.png", dpi=160)
    plt.close(fig)
    # 4. models on IBM normal days
    names = ["XGBoost\n(round-2 features)", "GraphSAGE v2", "GraphSAGE v2\n(graph removed)", "GraphSAGE v1\n(old inputs)"]
    vals = [0.594, 0.428, 0.408, 0.173]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, vals, color=[GREEN, BLUE, "#9db7d5", LIGHT], edgecolor="white")
    bar_labels(ax, bars)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Explicit graph features in XGBoost beat learned message passing here", loc="left", fontsize=14)
    ax.text(0, -0.24, f"IBM AML test Sept 9-10, prevalence {prevalence:.2%}. GraphSAGE vs same network without neighbours: +0.021 [+0.007, +0.034].\nBlending GraphSAGE into XGBoost added nothing (validation chose weight 0).",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xgboost_vs_graphsage.png", dpi=160)
    plt.close(fig)
    # 5. E4 results: translator faithfulness and tagger vs rules
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6))
    models = ["llama3.2:3b", "qwen2.5:3b", "qwen2.5:32b", "llama3.3:70b"]
    kept, flagged = [88.4, 91.8, 92.4, 84.3], [0, 2, 0, 0]
    bars = left.bar(models, kept, color=[GREEN, AMBER, GREEN, GREEN], edgecolor="white")
    left.set_ylim(0, 112)
    left.set_ylabel("% of fact-line details kept in the sentence")
    left.set_title("E4 translator: 419 sentences per model, checked claim by claim", loc="left", fontsize=13)
    for bar, k, w in zip(bars, kept, flagged):
        left.text(bar.get_x() + bar.get_width() / 2, k + 1.5, f"{k:.0f}%", ha="center", fontsize=13, fontweight="bold")
        left.text(bar.get_x() + bar.get_width() / 2, 4, f"wrong claims\nflagged: {w}", ha="center", fontsize=9.5, color="white", fontweight="bold")
    left.tick_params(axis="x", labelsize=10)
    arms = ["rule-only\n(code)", "qwen 3B\nverified", "qwen 3B\nunverified", "llama 3B\nunverified"]
    hit = [0.612, 0.163, 0.184, 0.0]
    bars = right.bar(arms, hit, color=[GREEN, BLUE, AMBER, RED], edgecolor="white")
    for bar, h, fa in zip(bars, hit, ["14", "2", "48", "50"]):
        right.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.0%}", ha="center", fontsize=13, fontweight="bold")
        right.text(bar.get_x() + bar.get_width() / 2, 0.02 if h > 0.1 else h + 0.075, f"false alarms\ntagged: {fa}/50", ha="center", fontsize=9, color="white" if h > 0.1 else INK)
    right.set_ylim(0, 0.8)
    right.set_ylabel("true shape found (visible cases, n=49)")
    right.set_title("E4 shape tagger: 3B models trail plain rules", loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e4_results.png", dpi=160)
    plt.close(fig)
    print(f"normal-day prevalence {prevalence:.4%} ({n_laundering}/{n_rows})")


def _card(case_id, model, source_csv, generations_path, filename, title_note):
    records = [json.loads(line) for line in Path(generations_path).read_text().splitlines()]
    record = next(r for r in records if r["case_id"] == case_id and r["model"] == model)
    checks = pd.read_csv(source_csv)
    checks = checks[(checks["case_id"] == case_id) & (checks["model"] == model)].set_index("label")
    header = [line for line in record["case_text"].splitlines() if line.startswith("FLAGGED PAYMENT")][0].replace("FLAGGED PAYMENT at", "Flagged payment,")
    sentences = record["clean_sentences"][:8]
    fig_height = 1.6 + 0.78 * len(sentences)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 11.8, fig_height - 0.2, boxstyle="round,pad=0.02,rounding_size=0.15", facecolor="white", edgecolor=LIGHT, linewidth=2))
    ax.text(0.4, fig_height - 0.45, f"{case_id}: {header}", fontsize=12.5, fontweight="bold", va="center")
    ax.text(0.4, fig_height - 0.9, title_note, fontsize=10, color=MUTED, va="center")
    y = fig_height - 1.5
    for item in sentences:
        label = item["label"]
        ok = bool(checks.loc[label, "supported"]) if label in checks.index else True
        ax.add_patch(FancyBboxPatch((0.4, y - 0.22), 1.45, 0.44, boxstyle="round,pad=0.01,rounding_size=0.1", facecolor=GREEN if ok else RED, edgecolor="none"))
        ax.text(1.12, y, ("supported" if ok else "NOT supported"), color="white", fontsize=10, fontweight="bold", ha="center", va="center")
        ax.text(2.05, y, textwrap.fill(f"{item['sentence']}  [{label}]", 100), fontsize=11, va="center")
        y -= 0.78
    fig.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_e4():
    _card("E4-004", "llama3.2:3b", "results/e4/checks_A2_full_sentences.csv", "results/e4/generations_v2.jsonl", "fig_e4_explanation_card.png",
          "Each sentence cites a rule-made fact line and is checked against it by code.")
    _card("E4-017", "qwen2.5:3b", "results/e4/checks_A2_full_sentences.csv", "results/e4/generations_v2.jsonl", "fig_e4_explanation_card_caught_error.png",
          "A real case where the checker caught a sentence whose numbers are not on its cited fact line.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("shapes", "all"):
        fig_shapes()
    if mode in ("patterns", "all"):
        fig_patterns()
    if mode in ("neighborhood", "all"):
        fig_neighborhood()
    if mode in ("charts", "all"):
        fig_charts()
    if mode in ("e4", "all"):
        fig_e4()
    print("figures in", OUT, sorted(p.name for p in OUT.glob("*.png")))
