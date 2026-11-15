"""IBM AML E2: load transactions into Neo4j (as IbmAccount / IBM_TRANSACTION, next to the PaySim graph)
and compute leakage-safe daily-snapshot graph features. Laundering labels are NOT stored in the graph.
Run on the Mac with Neo4j running:
    python src/ibm_graph_features.py load
    python src/ibm_graph_features.py features        (resumable, restarts Neo4j itself if memory runs out)"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))  # appended (not inserted) so the real neo4j package wins over ./neo4j
from neo4j.exceptions import ClientError, ServiceUnavailable
from src.graph_features import connect_when_ready, restart_neo4j, drop_graph, log, BATCH_SIZE
from src.ibm_data_prep import PROCESSED_DIR, split_name
from src.ibm_features_tabular import usd_rates

DIRECTED_GRAPH, UNDIRECTED_GRAPH = "ibm_snapshot_directed", "ibm_snapshot_undirected"
DAILY_DIR = PROCESSED_DIR / "ibm_e2_daily_graph_features"
ACCOUNT_FEATURE_NAMES = ["in_degree", "out_degree", "weighted_in_amount", "weighted_out_amount", "pagerank", "community_size"]


def load_transactions_table() -> pd.DataFrame:
    columns = ["transaction_id", "timestamp", "sender_account_id", "receiver_account_id", "Amount Paid", "Amount Received",
               "Payment Currency", "Receiving Currency", "Payment Format"]
    transactions = pd.read_parquet(PROCESSED_DIR / "ibm_transactions_all_columns.parquet", columns=columns)
    rates = usd_rates(transactions, split_name(transactions.timestamp) == "train")
    transactions["amount_usd"] = transactions["Amount Paid"] * transactions["Payment Currency"].astype(str).map(rates).fillna(1.0).values
    start = transactions.timestamp.min().normalize()
    transactions["minute"] = ((transactions.timestamp - start).dt.total_seconds() // 60).astype(np.int64)
    transactions["day"] = (transactions.minute // 1440).astype(np.int64)
    return transactions


def load_into_neo4j():
    transactions = load_transactions_table()
    log(f"loaded table {transactions.shape}")
    driver = connect_when_ready()
    with driver.session() as session:
        session.run("MATCH (a:IbmAccount) CALL (a) { DETACH DELETE a } IN TRANSACTIONS OF 20000 ROWS").consume()
        session.run("CREATE CONSTRAINT ibm_account_id IF NOT EXISTS FOR (a:IbmAccount) REQUIRE a.account_id IS UNIQUE").consume()
        session.run("CREATE INDEX ibm_transaction_minute IF NOT EXISTS FOR ()-[t:IBM_TRANSACTION]-() ON (t.minute)").consume()
        accounts = pd.unique(pd.concat([transactions.sender_account_id, transactions.receiver_account_id]))
        for start in range(0, len(accounts), 100_000):
            session.run("UNWIND $rows AS id CREATE (:IbmAccount {account_id: id})", rows=accounts[start:start + 100_000].tolist()).consume()
        log(f"created {len(accounts):,} IbmAccount nodes")
        records = transactions[["transaction_id", "sender_account_id", "receiver_account_id", "amount_usd", "minute", "day"]]
        records = records.assign(payment_format=transactions["Payment Format"].astype(str)).to_dict("records")
        for start in range(0, len(records), BATCH_SIZE):
            session.run("""
                UNWIND $rows AS row
                MATCH (s:IbmAccount {account_id: row.sender_account_id})
                MATCH (r:IbmAccount {account_id: row.receiver_account_id})
                CREATE (s)-[:IBM_TRANSACTION {transaction_id: row.transaction_id, amount_usd: row.amount_usd,
                                              minute: row.minute, day: row.day, payment_format: row.payment_format}]->(r)""",
                rows=records[start:start + BATCH_SIZE]).consume()
            if (start // BATCH_SIZE) % 25 == 0:
                log(f"  transactions loaded: {min(start + BATCH_SIZE, len(records)):,} / {len(records):,}")
        nodes = session.run("MATCH (a:IbmAccount) RETURN count(a) AS c").single()["c"]
        edges = session.run("MATCH ()-[t:IBM_TRANSACTION]->() RETURN count(t) AS c").single()["c"]
        log(f"DONE: {nodes:,} IBM accounts, {edges:,} IBM transactions in Neo4j")
    driver.close()


def fetch_properties(session, graph_name, property_names, account_ids):
    returned = ", ".join(f"gds.util.nodeProperty('{graph_name}', id(a), '{p}') AS {p}" for p in property_names)
    rows = []
    for start in range(0, len(account_ids), BATCH_SIZE):
        result = session.run(f"UNWIND $ids AS id MATCH (a:IbmAccount {{account_id: id}}) RETURN id AS account_id, {returned}",
                             ids=account_ids[start:start + BATCH_SIZE])
        rows.extend(r.data() for r in result)
    return pd.DataFrame(rows, columns=["account_id"] + property_names)


def snapshot_features(session, cutoff_minute: int, account_ids: list) -> pd.DataFrame:
    """One in-memory graph at a time; both use only transactions with minute < cutoff_minute."""
    drop_graph(session, DIRECTED_GRAPH)
    session.run("""
        MATCH (s:IbmAccount)-[t:IBM_TRANSACTION]->(r:IbmAccount) WHERE t.minute < $cutoff
        WITH gds.graph.project($name, s, r, {relationshipProperties: {amount: t.amount_usd}}) AS g RETURN g.nodeCount""",
        cutoff=cutoff_minute, name=DIRECTED_GRAPH).consume()
    for config in ["{mutateProperty: 'out_degree'}", "{orientation: 'REVERSE', mutateProperty: 'in_degree'}",
                   "{relationshipWeightProperty: 'amount', mutateProperty: 'weighted_out_amount'}",
                   "{orientation: 'REVERSE', relationshipWeightProperty: 'amount', mutateProperty: 'weighted_in_amount'}"]:
        session.run(f"CALL gds.degree.mutate('{DIRECTED_GRAPH}', {config})").consume()
    session.run(f"CALL gds.pageRank.mutate('{DIRECTED_GRAPH}', {{mutateProperty: 'pagerank', maxIterations: 20}})").consume()
    directed = fetch_properties(session, DIRECTED_GRAPH, ACCOUNT_FEATURE_NAMES[:5], account_ids)
    drop_graph(session, DIRECTED_GRAPH)

    drop_graph(session, UNDIRECTED_GRAPH)
    session.run("""
        MATCH (s:IbmAccount)-[t:IBM_TRANSACTION]->(r:IbmAccount) WHERE t.minute < $cutoff
        WITH gds.graph.project($name, s, r, {}, {undirectedRelationshipTypes: ['*']}) AS g RETURN g.nodeCount""",
        cutoff=cutoff_minute, name=UNDIRECTED_GRAPH).consume()
    session.run(f"CALL gds.louvain.mutate('{UNDIRECTED_GRAPH}', {{mutateProperty: 'community', concurrency: 1}})").consume()
    communities = fetch_properties(session, UNDIRECTED_GRAPH, ["community"], account_ids)
    sizes = pd.DataFrame([r.data() for r in session.run(f"""
        CALL gds.graph.nodeProperty.stream('{UNDIRECTED_GRAPH}', 'community') YIELD propertyValue
        RETURN propertyValue AS community, count(*) AS community_size""")])
    drop_graph(session, UNDIRECTED_GRAPH)
    communities = communities.merge(sizes, on="community", how="left").drop(columns="community")
    return directed.merge(communities, on="account_id", how="left")


def attach(day_rows: pd.DataFrame, account_features: pd.DataFrame) -> pd.DataFrame:
    output = day_rows[["transaction_id"]].copy()
    for role, column in [("sender", "sender_account_id"), ("receiver", "receiver_account_id")]:
        merged = day_rows[[column]].merge(account_features.rename(columns={"account_id": column}), on=column, how="left")
        output[f"{role}_is_new_account"] = merged["in_degree"].isna().astype(np.int8).values
        for feature in ACCOUNT_FEATURE_NAMES:
            output[f"{role}_graph_{feature}"] = merged[feature].fillna(0.0).values
    return output


def compute_one_day(session, transactions, day):
    start_time = time.time(); cutoff = day * 1440
    day_rows = transactions[transactions.day == day]
    past = transactions[transactions.minute < cutoff]
    with_past = set(past.sender_account_id) | set(past.receiver_account_id)
    needed = sorted((set(day_rows.sender_account_id) | set(day_rows.receiver_account_id)) & with_past)
    if len(past) == 0:
        features = pd.DataFrame(columns=["account_id"] + ACCOUNT_FEATURE_NAMES)
    else:
        features = snapshot_features(session, cutoff, needed)
    attach(day_rows, features).to_parquet(DAILY_DIR / f"day_{day:02d}.parquet", index=False)
    log(f"day {day:02d}: snapshot of {len(past):,} earlier transactions, {len(day_rows):,} transactions today, "
        f"{len(needed):,} accounts with history, {time.time() - start_time:.0f}s")


def compute_all_days():
    transactions = load_transactions_table()
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    driver = connect_when_ready()
    for day in sorted(transactions.day.unique()):
        if (DAILY_DIR / f"day_{day:02d}.parquet").exists():
            log(f"day {day:02d}: already done, skipping"); continue
        for attempt in range(1, 4):
            try:
                with driver.session() as session:
                    compute_one_day(session, transactions, int(day))
                break
            except (ClientError, ServiceUnavailable) as error:
                if attempt == 3 or ("memory" not in str(error).lower() and not isinstance(error, ServiceUnavailable)):
                    raise
                driver.close(); restart_neo4j(); driver = connect_when_ready()
    driver.close(); log("ALL DAYS DONE")


if __name__ == "__main__":
    {"load": load_into_neo4j, "features": compute_all_days}[sys.argv[1]]()
