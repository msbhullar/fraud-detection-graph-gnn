"""Deliberately NAIVE full-graph experiments (PaySim). The graph features here are computed ONCE on ALL
transactions, including the test period, so a test row can see edges from its own future. This is the leak
that E2 avoids with daily past-only snapshots. Reuses the GDS code in src/graph_features.py unchanged.

Run from the project root:
    python src/naive_full_graph.py features    # Mac only, needs Neo4j running with the PaySim graph loaded
    python src/naive_full_graph.py fullgraph   # 3a: E1 features + full-graph features, temporal split
    python src/naive_full_graph.py all         # 3b: E1 + balance columns + full-graph features, random split
The model settings are fixed (same as src/naive_leakage.py). No tuning."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_e2_features import GRAPH_FEATURE_COLUMNS
from src.features_tabular import MODEL_FEATURE_COLUMNS
from src.naive_leakage import (BALANCE_LEAK_COLUMNS, PROCESSED_DIR, load_split_frames, log,
                               shuffle_into_random_split, train_and_evaluate)

FULL_GRAPH_FEATURES_PATH = PROCESSED_DIR / "naive_full_graph_features.parquet"
FULL_GRAPH_CUTOFF_STEP = 10_000  # larger than any step (max 743), so the "snapshot" holds every edge
SPLIT_ORDER = ["train", "validation", "test"]


def compute_full_graph_features():
    """One projection of the WHOLE graph (all steps), same features as E2: degrees, money in/out, PageRank, Louvain size."""
    from neo4j.exceptions import ClientError, ServiceUnavailable
    from src import graph_features as snapshot_code

    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet",
                                   columns=["step", "nameOrig", "nameDest"])
    transactions["transaction_id"] = transactions.index
    log(f"transactions: {transactions.shape}")

    driver = snapshot_code.connect_when_ready()
    with driver.session() as session:
        edge_count = session.run("MATCH ()-[t:TRANSACTION]->() RETURN count(t) AS c").single()["c"]
    if edge_count != len(transactions):
        sys.exit(f"Neo4j holds {edge_count:,} TRANSACTION edges but the data has {len(transactions):,}. "
                 f"Load the PaySim graph first: python src/graph_features.py load")
    log(f"Neo4j holds all {edge_count:,} PaySim transactions")

    account_names = sorted(set(transactions["nameOrig"]) | set(transactions["nameDest"]))
    log(f"{len(account_names):,} accounts; projecting the full graph (this is the slow step)")
    start = time.time()
    for attempt in range(1, 4):
        try:
            with driver.session() as session:
                account_features, nodes, relationships = snapshot_code.compute_snapshot_account_features(
                    session, FULL_GRAPH_CUTOFF_STEP, account_names)
            break
        except (ClientError, ServiceUnavailable) as error:
            if attempt == 3 or ("memory" not in str(error).lower() and not isinstance(error, ServiceUnavailable)):
                raise
            driver.close()
            snapshot_code.restart_neo4j()
            driver = snapshot_code.connect_when_ready()
    driver.close()
    log(f"full graph: {nodes:,} nodes, {relationships:,} edges, {time.time() - start:.0f}s")

    features = snapshot_code.attach_to_transactions(transactions, account_features)
    features.to_parquet(FULL_GRAPH_FEATURES_PATH, index=False)
    log(f"saved {FULL_GRAPH_FEATURES_PATH}: shape {features.shape}")
    print(features.head(5).to_string(), flush=True)


def attach_full_graph_features(frames: dict) -> dict:
    """Join the full-graph features to the split frames by row position, after checking the rows line up."""
    graph_features = pd.read_parquet(FULL_GRAPH_FEATURES_PATH).sort_values("transaction_id").reset_index(drop=True)
    original = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet", columns=["step", "amount"])
    pooled_steps = np.concatenate([frames[name]["step"].values for name in SPLIT_ORDER])
    pooled_amounts = np.concatenate([frames[name]["amount"].values for name in SPLIT_ORDER])
    assert len(graph_features) == len(pooled_steps), "row counts differ"
    assert np.array_equal(original["step"].values, pooled_steps), "steps do not line up"
    assert np.array_equal(original["amount"].values, pooled_amounts), "amounts do not line up"
    log("alignment check passed: graph features line up with the split rows")
    start = 0
    joined = {}
    for name in SPLIT_ORDER:
        size = len(frames[name])
        graph_part = graph_features.iloc[start:start + size][GRAPH_FEATURE_COLUMNS].reset_index(drop=True)
        joined[name] = pd.concat([frames[name].reset_index(drop=True), graph_part], axis=1)
        start += size
        log(f"{name}: shape {joined[name].shape}")
    print(joined["train"].head(5).to_string(), flush=True)
    return joined


def show_how_much_the_full_graph_sees_extra(test_frame: pd.DataFrame):
    """Compare with the honest E2 daily snapshots (past-only) for the same test rows."""
    honest = pd.read_parquet(PROCESSED_DIR / "e2_features_test.parquet",
                             columns=["receiver_graph_in_degree", "sender_graph_out_degree"])
    for column in ["receiver_graph_in_degree", "sender_graph_out_degree"]:
        full_values, honest_values = test_frame[column].values, honest[column].values
        log(f"test {column}: full graph sees MORE than past-only snapshot for {(full_values > honest_values).mean():.1%} of rows, "
            f"equal for {(full_values == honest_values).mean():.1%}, fewer for {(full_values < honest_values).mean():.1%}")


def run_fullgraph_variant():
    base_columns = MODEL_FEATURE_COLUMNS + ["step"]
    frames = attach_full_graph_features(load_split_frames(base_columns))
    show_how_much_the_full_graph_sees_extra(frames["test"])
    train_and_evaluate(frames, MODEL_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS, "naive_fullgraph",
                       "temporal: train steps 1-446, validation 447-595, test 596-743; graph features from the FULL graph")


def run_all_variant():
    base_columns = MODEL_FEATURE_COLUMNS + BALANCE_LEAK_COLUMNS + ["step"]
    frames = attach_full_graph_features(load_split_frames(base_columns))
    frames = shuffle_into_random_split(frames)
    train_and_evaluate(frames, MODEL_FEATURE_COLUMNS + BALANCE_LEAK_COLUMNS + GRAPH_FEATURE_COLUMNS, "naive_all",
                       "NAIVE: balance columns + isFlaggedFraud + full-graph features + random split (same sizes as temporal)")


if __name__ == "__main__":
    variants = {"features": compute_full_graph_features, "fullgraph": run_fullgraph_variant, "all": run_all_variant}
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    if variant in variants:
        variants[variant]()
    else:
        sys.exit("usage: python src/naive_full_graph.py <features|fullgraph|all>")
