"""LlamaIndex semantic SQL over telecom_ops.db using ObjectIndex for table selection."""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "telecom_ops.db"))

_query_engine = None

TABLE_DESCRIPTIONS = {
    "network_towers": (
        "Telecom tower inventory. Columns: tower_id, region, city, technology (LTE/5G), "
        "status (OPERATIONAL/DEGRADED/OUTAGE/MAINTENANCE), capacity_subscribers, last_maintenance. "
        "Use for tower availability, counts by region or status, geographic coverage questions."
    ),
    "network_outages": (
        "Historical network outage records. Columns: outage_id, region, tower_id, start_time, "
        "end_time, severity (CRITICAL/HIGH/MEDIUM/LOW), affected_customers, root_cause, incident_id. "
        "Use for outage trends, severity analysis, regional comparisons, and incident history."
    ),
    "tower_performance": (
        "Time-series performance metrics per tower. Columns: record_id, tower_id, avg_latency_ms, "
        "packet_loss_pct, throughput_mbps, signal_strength_dbm, recorded_at. "
        "Use for network quality, latency analysis, packet loss, throughput comparisons."
    ),
    "open_incidents": (
        "Active NOC incidents linked to towers. Columns: incident_id, tower_id, severity, "
        "status (OPEN/IN_PROGRESS/RESOLVED), description, eta_hours, opened_at. "
        "Use for live incidents, open tickets, estimated resolution times."
    ),
    "customer_subscriptions": (
        "Customer plan details. Columns: customer_id, account_type (Consumer/Business/Enterprise), "
        "plan_name, data_limit_gb, monthly_fee, contract_end, region. "
        "Use for subscription counts, plan distribution, revenue, customer segmentation."
    ),
    "billing_accounts": (
        "Customer billing account balances. Columns: customer_id, current_balance, billing_cycle_day. "
        "Use for account balance lookups and billing cycle information."
    ),
    "billing_charges": (
        "Individual billing line items. Columns: charge_id, customer_id, description, amount, "
        "charge_date, billing_period, is_duplicate_flag. "
        "Use for charge history, duplicate detection, billing amount analysis."
    ),
    "billing_credits": (
        "Applied or pending billing credits. Columns: credit_id, customer_id, amount, reason, "
        "status (APPLIED/PENDING_APPROVAL), reference_number, applied_at. "
        "Use for credit history and pending approval items."
    ),
    "billing_disputes": (
        "Billing dispute records. Columns: dispute_id, customer_id, charge_id, dispute_reason, "
        "status (OPEN/RESOLVED/REJECTED), opened_at, resolved_at. "
        "Use for dispute status, resolution tracking."
    ),
}


def _build_query_engine():
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
    from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
    from llama_index.core.query_engine import SQLTableRetrieverQueryEngine
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI
    from sqlalchemy import create_engine

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    engine = create_engine(f"sqlite:///{DB_PATH}")
    sql_database = SQLDatabase(engine, include_tables=list(TABLE_DESCRIPTIONS.keys()))

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=name, context_str=desc)
        for name, desc in TABLE_DESCRIPTIONS.items()
    ]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

    _query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        obj_index.as_retriever(similarity_top_k=2),
    )
    return _query_engine


def query_sql(question: str) -> str:
    """Run a natural language query against telecom_ops.db using semantic table selection."""
    qe = _build_query_engine()
    response = qe.query(question)
    return str(response)
