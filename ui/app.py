"""Prodapt AI Operations Center — Streamlit UI."""
import os
import sys

import httpx
import streamlit as st

# Ensure project root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

# ── Constants ────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(_ROOT, "data", "telecom_ops.db")
VECTOR_INDEX_PATH = os.path.join(_ROOT, "data", "vector_index")
NETWORK_ADK_URL = os.environ.get("NETWORK_ADK_URL", "http://localhost:8001")
BILLING_ADK_URL = os.environ.get("BILLING_ADK_URL", "http://localhost:8002")

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prodapt AI Operations Center",
    page_icon="📡",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_adk_service(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/.well-known/agent-card.json", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _status_badge(ok: bool, ok_label: str = "Ready", fail_label: str = "Not ready") -> str:
    if ok:
        return f"🟢 {ok_label}"
    return f"🔴 {fail_label}"


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("System Status")
    st.divider()

    db_ok = os.path.exists(DB_PATH)
    st.markdown(f"**Database** {_status_badge(db_ok, 'Ready', 'Run init_db.py')}")

    vec_ok = os.path.exists(VECTOR_INDEX_PATH) and bool(os.listdir(VECTOR_INDEX_PATH))
    st.markdown(
        f"**Vector Index** {_status_badge(vec_ok, 'Built', 'Builds on first RAG query')}"
    )

    net_ok = _check_adk_service(NETWORK_ADK_URL)
    st.markdown(f"**Network Diagnostics ADK** {_status_badge(net_ok, 'Running', 'Not running')}")

    bill_ok = _check_adk_service(BILLING_ADK_URL)
    st.markdown(f"**Billing Resolution ADK** {_status_badge(bill_ok, 'Running', 'Not running')}")

    if not net_ok or not bill_ok:
        st.warning(
            "One or more ADK services are offline. "
            "Start them in separate terminals:\n\n"
            "```\npython adk-services/network_diagnostics/agent.py\n"
            "python adk-services/billing_resolution/agent.py\n```"
        )

    st.divider()
    st.subheader("Framework Map")
    st.markdown(
        """
| Capability | Framework |
|---|---|
| Policy FAQ | LlamaIndex RAG |
| Network Analytics | LlamaIndex Semantic SQL |
| Network Diagnostics | Google ADK (A2A) |
| Billing Resolution | Google ADK (A2A) |
| Orchestration | LangGraph |
| Customer Comms | CrewAI |
"""
    )

    st.divider()
    st.caption("Prodapt AI Operations Center v1.0")

# ── Main area ────────────────────────────────────────────────────────────────

st.title("📡 Prodapt AI Operations Center")
st.caption(
    "Powered by **LangGraph** · **LlamaIndex** · **Google ADK** · **CrewAI** · **Streamlit**"
)
st.divider()

# Sample queries for convenience
with st.expander("Sample queries (click to copy)", expanded=False):
    st.markdown(
        """
**Scenario 1 — Policy FAQ:**
> What is Prodapt's roaming policy for Western Europe?

**Scenario 2 — Network Analytics:**
> Which region had the most CRITICAL network outages recently?

**Scenario 3 — Network Diagnostics:**
> My 5G keeps dropping in Austin near tower TX-512. Please diagnose.

**Scenario 4 — Billing Dispute:**
> Customer CUST-10002 was charged twice for Unlimited Plus. Investigate and apply credit.

**Scenario 5 — Combined:**
> We had a 6-hour outage in the Midwest. Am I eligible for an SLA credit?
"""
    )

query = st.text_area(
    "Customer Inquiry",
    placeholder="Type a customer inquiry in plain English, e.g.:\n"
    "'My 5G drops near tower TX-512 in Austin — please diagnose.'",
    height=120,
    key="query_input",
)

submit = st.button("Submit", type="primary", disabled=not query.strip())

if submit and query.strip():
    with st.spinner("Processing inquiry through the AI Operations Center…"):
        try:
            from orchestration.graph import run_telecom_assistant

            result = run_telecom_assistant(query.strip())
            st.session_state["last_result"] = result
            st.session_state["last_query"] = query.strip()
        except Exception as e:
            st.error(f"Error running the AI Operations Center: {e}")
            st.session_state["last_result"] = None

# ── Results ──────────────────────────────────────────────────────────────────

if st.session_state.get("last_result"):
    result = st.session_state["last_result"]

    st.divider()
    st.subheader("Customer Response")
    st.info(result["final_response"])

    # Agent Execution Trace
    st.divider()
    with st.expander("Agent Execution Trace", expanded=True):
        trace = result.get("execution_trace", [])
        if not trace:
            st.write("No workers executed.")
        else:
            for i, step in enumerate(trace, 1):
                worker = step.get("worker", "Unknown")
                output = step.get("output", "")
                truncated = output[:500] + ("…" if len(output) > 500 else "")

                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**Step {i}**")
                    st.markdown(f"`{worker}`")
                with col2:
                    st.text(truncated)
                st.divider()

    # Optional debug expander
    with st.expander("Full Agent Context (debug)", expanded=False):
        st.text(result.get("agent_context", "(empty)"))
