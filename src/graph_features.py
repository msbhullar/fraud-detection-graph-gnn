"""E2: load PaySim into Neo4j and compute leakage-safe graph features.
Run on the Mac (needs Neo4j at localhost:7687):
    python src/graph_features.py load
Fraud labels are NOT stored in the graph, so no graph computation can see them."""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, ServiceUnavailable

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "fraudgraph2026")
PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 20_000


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_transactions_into_neo4j():
    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet",
                                   columns=["step", "type", "amount", "nameOrig", "nameDest"])
    transactions["transaction_id"] = transactions.index  # row number in the filtered file, used to join back
    log(f"loaded parquet: {transactions.shape}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
    with driver.session() as session:
        session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 50000 ROWS")  # start clean
        session.run("CREATE CONSTRAINT account_name IF NOT EXISTS FOR (a:Account) REQUIRE a.name IS UNIQUE")
        session.run("CREATE INDEX transaction_step IF NOT EXISTS FOR ()-[t:TRANSACTION]-() ON (t.step)")
        log("database cleared, constraint and index ready")

        account_names = pd.unique(pd.concat([transactions["nameOrig"], transactions["nameDest"]]))
        for start in range(0, len(account_names), BATCH_SIZE * 5):
            batch = [{"name": n} for n in account_names[start:start + BATCH_SIZE * 5]]
            session.run("UNWIND $rows AS row CREATE (:Account {name: row.name})", rows=batch)
        log(f"created {len(account_names):,} Account nodes")

        records = transactions.to_dict("records")
        for start in range(0, len(records), BATCH_SIZE):
            session.run("""
                UNWIND $rows AS row
                MATCH (sender:Account {name: row.nameOrig})
                MATCH (receiver:Account {name: row.nameDest})
                CREATE (sender)-[:TRANSACTION {transaction_id: row.transaction_id, type: row.type,
                                               amount: row.amount, step: row.step}]->(receiver)
                """, rows=records[start:start + BATCH_SIZE])
            if (start // BATCH_SIZE) % 10 == 0:
                log(f"  transactions loaded: {min(start + BATCH_SIZE, len(records)):,} / {len(records):,}")

        node_count = session.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        edge_count = session.run("MATCH ()-[t:TRANSACTION]->() RETURN count(t) AS c").single()["c"]
        log(f"DONE: {node_count:,} accounts, {edge_count:,} transactions in Neo4j")
    driver.close()


STEPS_PER_DAY = 24
NUMBER_OF_DAYS = 31  # steps 1-743
DAILY_FEATURES_DIR = PROCESSED_DIR / "e2_daily_graph_features"
ACCOUNT_FEATURE_NAMES = ["in_degree", "out_degree", "weighted_in_amount", "weighted_out_amount",
                         "pagerank", "community_size"]
DIRECTED_GRAPH = "snapshot_directed"
UNDIRECTED_GRAPH = "snapshot_undirected"


def first_step_of_day(day: int) -> int:
    return day * STEPS_PER_DAY + 1


def drop_graph(session, graph_name: str):
    session.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", name=graph_name).consume()


def fetch_node_properties(session, graph_name: str, property_names: list, account_names: list) -> pd.DataFrame:
    returned = ",\n".join(f"gds.util.nodeProperty('{graph_name}', id(a), '{p}') AS {p}" for p in property_names)
    rows = []
    for start in range(0, len(account_names), BATCH_SIZE):
        result = session.run(f"UNWIND $names AS name MATCH (a:Account {{name: name}}) RETURN name, {returned}",
                             names=account_names[start:start + BATCH_SIZE])
        rows.extend(record.data() for record in result)
    return pd.DataFrame(rows, columns=["name"] + property_names)


def compute_snapshot_account_features(session, cutoff_step: int, account_names: list):
    """One in-memory graph at a time (keeps memory low). Both graphs use only transactions with step < cutoff_step."""
    # 1) directed graph: degrees, money in/out, PageRank
    drop_graph(session, DIRECTED_GRAPH)
    directed = session.run("""
        MATCH (sender:Account)-[t:TRANSACTION]->(receiver:Account) WHERE t.step < $cutoff
        WITH gds.graph.project($name, sender, receiver, {relationshipProperties: {amount: t.amount}}) AS g
        RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships""",
        cutoff=cutoff_step, name=DIRECTED_GRAPH).single()
    session.run(f"CALL gds.degree.mutate('{DIRECTED_GRAPH}', {{mutateProperty: 'out_degree'}})").consume()
    session.run(f"CALL gds.degree.mutate('{DIRECTED_GRAPH}', {{orientation: 'REVERSE', mutateProperty: 'in_degree'}})").consume()
    session.run(f"CALL gds.degree.mutate('{DIRECTED_GRAPH}', {{relationshipWeightProperty: 'amount', mutateProperty: 'weighted_out_amount'}})").consume()
    session.run(f"CALL gds.degree.mutate('{DIRECTED_GRAPH}', {{orientation: 'REVERSE', relationshipWeightProperty: 'amount', mutateProperty: 'weighted_in_amount'}})").consume()
    session.run(f"CALL gds.pageRank.mutate('{DIRECTED_GRAPH}', {{mutateProperty: 'pagerank', maxIterations: 20}})").consume()
    directed_features = fetch_node_properties(session, DIRECTED_GRAPH,
        ["in_degree", "out_degree", "weighted_in_amount", "weighted_out_amount", "pagerank"], account_names)
    drop_graph(session, DIRECTED_GRAPH)

    # 2) undirected graph: Louvain communities
    drop_graph(session, UNDIRECTED_GRAPH)
    session.run("""
        MATCH (sender:Account)-[t:TRANSACTION]->(receiver:Account) WHERE t.step < $cutoff
        WITH gds.graph.project($name, sender, receiver, {}, {undirectedRelationshipTypes: ['*']}) AS g
        RETURN g.nodeCount""", cutoff=cutoff_step, name=UNDIRECTED_GRAPH).consume()
    session.run(f"CALL gds.louvain.mutate('{UNDIRECTED_GRAPH}', {{mutateProperty: 'community', concurrency: 1}})").consume()
    community_features = fetch_node_properties(session, UNDIRECTED_GRAPH, ["community"], account_names)
    community_sizes = pd.DataFrame([r.data() for r in session.run(f"""
        CALL gds.graph.nodeProperty.stream('{UNDIRECTED_GRAPH}', 'community') YIELD propertyValue
        RETURN propertyValue AS community, count(*) AS community_size""")])
    drop_graph(session, UNDIRECTED_GRAPH)

    community_features = community_features.merge(community_sizes, on="community", how="left").drop(columns=["community"])
    account_features = directed_features.merge(community_features, on="name", how="left")
    return account_features, directed["nodes"], directed["relationships"]


def attach_to_transactions(day_transactions: pd.DataFrame, account_features: pd.DataFrame) -> pd.DataFrame:
    output = day_transactions[["transaction_id"]].copy()
    for role, column in [("sender", "nameOrig"), ("receiver", "nameDest")]:
        role_features = account_features.add_prefix(f"{role}_graph_").rename(columns={f"{role}_graph_name": column})
        merged = day_transactions[[column]].merge(role_features, on=column, how="left")
        output[f"{role}_is_new_account"] = merged[f"{role}_graph_in_degree"].isna().astype(int).values
        for feature in ACCOUNT_FEATURE_NAMES:
            output[f"{role}_graph_{feature}"] = merged[f"{role}_graph_{feature}"].fillna(0.0).values
    return output


def connect_when_ready(max_wait_seconds: int = 180):
    """Open a driver, waiting until Neo4j accepts connections."""
    waited = 0
    while True:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
        try:
            driver.verify_connectivity()
            return driver
        except (ServiceUnavailable, OSError):
            driver.close()
            if waited >= max_wait_seconds:
                raise
            time.sleep(5); waited += 5


def restart_neo4j():
    """Neo4j on an 8 GB Mac runs with a 2 GB heap; restarting clears leftover memory between snapshots."""
    log("Neo4j is low on memory -> restarting the container (docker compose restart)")
    subprocess.run(["docker", "compose", "restart"], check=True)
    time.sleep(20)


def compute_one_day(session, transactions: pd.DataFrame, day: int, output_path: Path):
    start_time = time.time()
    cutoff_step = first_step_of_day(day)
    day_transactions = transactions[(transactions.step >= cutoff_step) & (transactions.step < cutoff_step + STEPS_PER_DAY)]
    past_transactions = transactions[transactions.step < cutoff_step]
    accounts_with_past = set(past_transactions.nameOrig) | set(past_transactions.nameDest)
    accounts_needed = sorted((set(day_transactions.nameOrig) | set(day_transactions.nameDest)) & accounts_with_past)
    if len(past_transactions) == 0:
        account_features = pd.DataFrame(columns=["name"] + ACCOUNT_FEATURE_NAMES)
        nodes, relationships = 0, 0
    else:
        account_features, nodes, relationships = compute_snapshot_account_features(session, cutoff_step, accounts_needed)
    attach_to_transactions(day_transactions, account_features).to_parquet(output_path, index=False)
    log(f"day {day:02d}: snapshot steps < {cutoff_step} ({nodes:,} nodes, {relationships:,} edges), "
        f"{len(day_transactions):,} transactions, {len(accounts_needed):,} accounts with history, "
        f"{time.time() - start_time:.0f}s")


def compute_daily_graph_features(days: list):
    transactions = pd.read_parquet(PROCESSED_DIR / "transfer_cashout_all_columns.parquet",
                                   columns=["step", "nameOrig", "nameDest"])
    transactions["transaction_id"] = transactions.index
    DAILY_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    driver = connect_when_ready()
    for day in days:
        output_path = DAILY_FEATURES_DIR / f"day_{day:02d}.parquet"
        if output_path.exists():
            log(f"day {day:02d}: already done, skipping"); continue
        for attempt in range(1, 4):
            try:
                with driver.session() as session:
                    compute_one_day(session, transactions, day, output_path)
                break
            except (ClientError, ServiceUnavailable) as error:
                if "memory" not in str(error).lower() and not isinstance(error, ServiceUnavailable):
                    raise
                if attempt == 3:
                    raise
                driver.close()
                restart_neo4j()
                driver = connect_when_ready()
    with driver.session() as session:
        for graph_name in [DIRECTED_GRAPH, UNDIRECTED_GRAPH]:
            drop_graph(session, graph_name)
    driver.close()
    log("ALL DAYS DONE")


if __name__ == "__main__":
    # python src/graph_features.py load
    # python src/graph_features.py features            (all 31 days, resumable)
    # python src/graph_features.py features 30         (one day, for a timing test)
    if sys.argv[1] == "load":
        load_transactions_into_neo4j()
    elif sys.argv[1] == "features":
        chosen_days = [int(d) for d in sys.argv[2:]] or list(range(NUMBER_OF_DAYS))
        compute_daily_graph_features(chosen_days)
