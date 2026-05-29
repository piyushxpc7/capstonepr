"""LangGraph supervisor + worker nodes for the Prodapt AI Operations Center."""
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

load_dotenv()

# Ensure project root is on path for sibling imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestration.state import AgentState

WORKER_NAMES = Literal[
    "PolicyRAG",
    "NetworkAnalytics",
    "NetworkDiagnosticsADK",
    "BillingResolutionADK",
    "CustomerCommsCrew",
    "FINISH",
]


class RouterOutput(BaseModel):
    next: WORKER_NAMES


SUPERVISOR_SYSTEM = """\
You are the Prodapt AI Operations Center supervisor.

Your job is to route each customer inquiry to the correct specialist worker.
Review the accumulated agent_context and the already-run workers to decide who should run next.

Routing rules:
- PolicyRAG: policy questions, FAQ, roaming charges, SLA rules, device upgrade eligibility, 5G FAQ
- NetworkAnalytics: outage trends, tower counts, packet loss stats, latency comparisons, analytics/SQL
- NetworkDiagnosticsADK: live tower diagnostics, signal drops, connectivity problems — especially when a tower ID (e.g. TX-512) is mentioned
- BillingResolutionADK: billing disputes, duplicate charges, credit requests, customer IDs starting with CUST-
- CustomerCommsCrew: ALWAYS route here after all relevant specialists have run — it drafts the final customer response
- FINISH: ONLY after CustomerCommsCrew has run and the final response is ready

CRITICAL RULES:
1. NEVER route to a worker that has already run. Already-run workers: [{already_run}]
2. Once a specialist has run, move on — do NOT repeat it.
3. Route CustomerCommsCrew exactly once, then FINISH immediately after.

For multi-domain questions (e.g. "outage + SLA credit"):
  route NetworkAnalytics first, then PolicyRAG, then CustomerCommsCrew.

Current accumulated context:
{agent_context}
"""


SPECIALISTS = {"PolicyRAG", "NetworkAnalytics", "NetworkDiagnosticsADK", "BillingResolutionADK"}


def _supervisor_node(state: AgentState) -> AgentState:
    execution_trace = state.get("execution_trace", [])
    already_run = {step["worker"] for step in execution_trace}

    # Hard guards — no LLM needed for terminal conditions
    if "CustomerCommsCrew" in already_run:
        return {"next": "FINISH"}
    if already_run & SPECIALISTS and not (SPECIALISTS - already_run):
        # All 4 specialists ran, CustomerCommsCrew hasn't → force it
        return {"next": "CustomerCommsCrew"}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(RouterOutput)

    agent_context = state.get("agent_context", "")
    system_content = SUPERVISOR_SYSTEM.format(
        agent_context=agent_context or "(none yet)",
        already_run=", ".join(already_run) if already_run else "none",
    )

    response: RouterOutput = structured_llm.invoke(
        [{"role": "system", "content": system_content}]
        + [{"role": "user", "content": state["user_query"]}]
    )

    # Python-level enforcement: never re-run an already-run worker
    chosen = response.next
    if chosen in already_run:
        # LLM tried to repeat — if specialists have run, go to CustomerCommsCrew, else FINISH
        chosen = "CustomerCommsCrew" if (already_run & SPECIALISTS) else "FINISH"

    return {"next": chosen}


def _route_decision(state: AgentState) -> str:
    return state["next"]


def _policy_rag_node(state: AgentState) -> AgentState:
    from llamaindex_rag.document_rag import query_policy

    result = query_policy(state["user_query"])
    context_entry = f"\n[PolicyRAG]\n{result}"
    trace_entry = {"worker": "PolicyRAG", "output": result[:500]}
    return {
        "agent_context": state.get("agent_context", "") + context_entry,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=result, name="PolicyRAG")],
        "next": "",
    }


def _network_analytics_node(state: AgentState) -> AgentState:
    from llamaindex_rag.sql_semantic_search import query_sql

    result = query_sql(state["user_query"])
    context_entry = f"\n[NetworkAnalytics]\n{result}"
    trace_entry = {"worker": "NetworkAnalytics", "output": result[:500]}
    return {
        "agent_context": state.get("agent_context", "") + context_entry,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=result, name="NetworkAnalytics")],
        "next": "",
    }


def _network_diagnostics_adk_node(state: AgentState) -> AgentState:
    from orchestration.adk_remote_client import call_network_diagnostics_adk

    result = call_network_diagnostics_adk(state["user_query"])
    context_entry = f"\n[NetworkDiagnosticsADK]\n{result}"
    trace_entry = {"worker": "NetworkDiagnosticsADK", "output": result[:500]}
    return {
        "agent_context": state.get("agent_context", "") + context_entry,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=result, name="NetworkDiagnosticsADK")],
        "next": "",
    }


def _billing_resolution_adk_node(state: AgentState) -> AgentState:
    from orchestration.adk_remote_client import call_billing_resolution_adk

    result = call_billing_resolution_adk(state["user_query"])
    context_entry = f"\n[BillingResolutionADK]\n{result}"
    trace_entry = {"worker": "BillingResolutionADK", "output": result[:500]}
    return {
        "agent_context": state.get("agent_context", "") + context_entry,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=result, name="BillingResolutionADK")],
        "next": "",
    }


def _customer_comms_crew_node(state: AgentState) -> AgentState:
    from orchestration.crew_nodes import run_customer_comms_crew

    result = run_customer_comms_crew(
        user_query=state["user_query"],
        agent_context=state.get("agent_context", ""),
    )
    context_entry = f"\n[CustomerCommsCrew]\n{result}"
    trace_entry = {"worker": "CustomerCommsCrew", "output": result[:500]}
    return {
        "agent_context": state.get("agent_context", "") + context_entry,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=result, name="CustomerCommsCrew")],
        "final_response": result,
        "next": "",
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("PolicyRAG", _policy_rag_node)
    graph.add_node("NetworkAnalytics", _network_analytics_node)
    graph.add_node("NetworkDiagnosticsADK", _network_diagnostics_adk_node)
    graph.add_node("BillingResolutionADK", _billing_resolution_adk_node)
    graph.add_node("CustomerCommsCrew", _customer_comms_crew_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_decision,
        {
            "PolicyRAG": "PolicyRAG",
            "NetworkAnalytics": "NetworkAnalytics",
            "NetworkDiagnosticsADK": "NetworkDiagnosticsADK",
            "BillingResolutionADK": "BillingResolutionADK",
            "CustomerCommsCrew": "CustomerCommsCrew",
            "FINISH": END,
        },
    )

    for worker in [
        "PolicyRAG",
        "NetworkAnalytics",
        "NetworkDiagnosticsADK",
        "BillingResolutionADK",
        "CustomerCommsCrew",
    ]:
        graph.add_edge(worker, "supervisor")

    return graph.compile()


_app = None


def _get_app():
    global _app
    if _app is None:
        _app = _build_graph()
    return _app


def run_telecom_assistant(user_query: str) -> dict:
    """Invoke the LangGraph supervisor loop and return structured results for the UI."""
    app = _get_app()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
        "user_query": user_query,
        "agent_context": "",
        "execution_trace": [],
        "final_response": "",
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 20})

    return {
        "final_response": final_state.get("final_response", "No response generated."),
        "execution_trace": final_state.get("execution_trace", []),
        "agent_context": final_state.get("agent_context", ""),
    }
