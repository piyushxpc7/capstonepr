"""Prodapt AI Operations Center — Streamlit UI with ChatGPT-like conversation."""
import os
import sys
import json

import httpx
import streamlit as st

# Ensure project root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from conversation_manager import (
    create_conversation,
    get_conversations,
    get_conversation_messages,
    add_message,
    get_conversation_info,
    delete_conversation,
    get_conversation_context,
)

# ── Constants ────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(_ROOT, "data", "telecom_ops.db")
VECTOR_INDEX_PATH = os.path.join(_ROOT, "data", "vector_index")
NETWORK_ADK_URL = os.environ.get("NETWORK_ADK_URL", "http://localhost:8001")
BILLING_ADK_URL = os.environ.get("BILLING_ADK_URL", "http://localhost:8002")
MAX_FOLLOW_UPS = 5

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prodapt AI Operations Center",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS for animations and styling ────────────────────────────────────

st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

[data-testid="stMainBlockContainer"] {
    animation: fadeIn 0.6s ease-out;
}

button {
    transition: all 0.3s ease !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

button:active {
    transform: translateY(0) !important;
}

[data-testid="stTextInput"] input {
    transition: all 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 0 3px rgba(59, 89, 152, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

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


# ── Session state initialization ──────────────────────────────────────────────

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "show_sample_queries" not in st.session_state:
    st.session_state.show_sample_queries = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 Conversations")
    st.divider()

    # New conversation button
    if st.button("➕ New Conversation", use_container_width=True):
        conv_id = create_conversation()
        st.session_state.current_conversation_id = conv_id
        st.rerun()

    st.divider()

    # Conversation list
    conversations = get_conversations()
    if conversations:
        st.subheader("History", divider=False)
        for conv in conversations:
            conv_id = conv["conversation_id"]
            title = conv["title"][:30]  # Truncate long titles
            follow_ups = conv["follow_up_count"]

            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{title}\n_({follow_ups} follow-ups)_",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                ):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{conv_id}"):
                    delete_conversation(conv_id)
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                    st.rerun()
    else:
        st.caption("No conversations yet. Start a new one!")

    st.divider()
    st.subheader("System Status", divider=False)

    db_ok = os.path.exists(DB_PATH)
    st.markdown(f"**Database** {_status_badge(db_ok, 'Ready', 'Run init_db.py')}")

    vec_ok = os.path.exists(VECTOR_INDEX_PATH) and bool(os.listdir(VECTOR_INDEX_PATH))
    st.markdown(
        f"**Vector Index** {_status_badge(vec_ok, 'Built', 'Builds on first RAG query')}"
    )

    net_ok = _check_adk_service(NETWORK_ADK_URL)
    st.markdown(f"**Network ADK** {_status_badge(net_ok, 'Running', 'Not running')}")

    bill_ok = _check_adk_service(BILLING_ADK_URL)
    st.markdown(f"**Billing ADK** {_status_badge(bill_ok, 'Running', 'Not running')}")

    if not net_ok or not bill_ok:
        st.warning(
            "One or more ADK services are offline. "
            "Start them in separate terminals:\n\n"
            "```\npython adk-services/network_diagnostics/agent.py\n"
            "python adk-services/billing_resolution/agent.py\n```"
        )

    st.divider()
    st.subheader("Framework Map", divider=False)
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
    st.caption("Prodapt AI Operations Center v2.0")

# ── Main area ────────────────────────────────────────────────────────────────

st.title("📡 Prodapt AI Operations Center")
st.caption(
    "Powered by **LangGraph** · **LlamaIndex** · **Google ADK** · **CrewAI** · **Streamlit**"
)

# Initialize conversation if needed
if not st.session_state.current_conversation_id:
    st.info("👈 Select or create a conversation to start")
    st.stop()

conv_id = st.session_state.current_conversation_id
conv_info = get_conversation_info(conv_id)

if not conv_info:
    st.error("Conversation not found")
    st.stop()

st.divider()

# Check if conversation is at max follow-ups
follow_up_count = conv_info["follow_up_count"]
at_max = follow_up_count >= MAX_FOLLOW_UPS

if at_max:
    st.warning(
        f"⚠️ Maximum {MAX_FOLLOW_UPS} follow-ups reached for this conversation. "
        "👈 Start a new conversation to continue."
    )

# Display conversation title
st.subheader(conv_info["title"], divider=True)

# Load and display message history
messages = get_conversation_messages(conv_id)

if not messages:
    # First message in conversation
    with st.expander("💡 Sample queries (click to copy)", expanded=True):
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
else:
    # Display message history in chat format
    for msg in messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.info(msg["content"])
                if msg.get("execution_trace"):
                    with st.expander("🔍 Execution Trace", expanded=False):
                        try:
                            trace = json.loads(msg["execution_trace"])
                            for i, step in enumerate(trace, 1):
                                worker = step.get("worker", "Unknown")
                                output = step.get("output", "")
                                st.markdown(
                                    f"**Step {i}** — "
                                    f"<span style='color:#4CAF50;font-weight:600'>{worker}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(output)
                                if i < len(trace):
                                    st.divider()
                        except json.JSONDecodeError:
                            st.markdown(msg["execution_trace"])

st.divider()

# Input area (disabled if at max follow-ups)
placeholder = (
    "Ask a follow-up..." if messages else "Type your question here..."
)
user_input = st.text_input(
    "Your message:",
    placeholder=placeholder,
    disabled=at_max,
    key="user_message_input",
)

submit = st.button(
    "🚀 Send",
    type="primary",
    disabled=not user_input.strip() or at_max,
)

if submit and user_input.strip():
    # Add user message to conversation
    add_message(conv_id, "user", user_input.strip())

    # Update title if it's the first message
    if len(messages) == 0:
        first_words = user_input[:50]
        import sqlite3

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (first_words, conv_id)
        )
        conn.commit()
        conn.close()

    # Run the assistant
    with st.spinner("Processing inquiry through the AI Operations Center…"):
        try:
            from orchestration.graph import run_telecom_assistant_with_context

            # Get conversation context for multi-turn
            conv_context = get_conversation_context(conv_id)

            result = run_telecom_assistant_with_context(
                user_input.strip(),
                conv_context
            )

            # Add assistant response to conversation
            add_message(
                conv_id,
                "assistant",
                result["final_response"],
                final_response=result["final_response"],
                execution_trace=result.get("execution_trace", [])
            )

            st.session_state["user_message_input"] = ""
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
