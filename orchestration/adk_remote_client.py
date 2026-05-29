"""Wrappers for calling remote ADK A2A agents from LangGraph nodes."""
import asyncio
import os
import uuid

import httpx

NETWORK_ADK_URL = os.environ.get("NETWORK_ADK_URL", "http://localhost:8001")
BILLING_ADK_URL = os.environ.get("BILLING_ADK_URL", "http://localhost:8002")


def _is_service_up(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/.well-known/agent-card.json", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


async def _call_adk_agent(base_url: str, query: str, agent_label: str) -> str:
    """Send a message to a remote ADK A2A agent and return the final text response."""
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    remote = RemoteA2aAgent(
        name=agent_label,
        agent_card=f"{base_url}/.well-known/agent-card.json",
    )
    runner = Runner(
        agent=remote,
        session_service=session_service,
        app_name=agent_label,
    )

    user_id = "langgraph_user"
    session_id = uuid.uuid4().hex
    await session_service.create_session(
        app_name=agent_label, user_id=user_id, session_id=session_id
    )

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

    return final_text or f"[{agent_label}] No response received."


def call_network_diagnostics_adk(query: str) -> str:
    """Call the Network Diagnostics ADK agent on port 8001."""
    if not _is_service_up(NETWORK_ADK_URL):
        return (
            "[NetworkDiagnosticsADK] Service unavailable. "
            "Start with: python adk-services/network_diagnostics/agent.py"
        )
    try:
        return asyncio.run(_call_adk_agent(NETWORK_ADK_URL, query, "NetworkDiagnosticsADK"))
    except Exception as e:
        return f"[NetworkDiagnosticsADK] Error: {e}"


def call_billing_resolution_adk(query: str) -> str:
    """Call the Billing Resolution ADK agent on port 8002."""
    if not _is_service_up(BILLING_ADK_URL):
        return (
            "[BillingResolutionADK] Service unavailable. "
            "Start with: python adk-services/billing_resolution/agent.py"
        )
    try:
        return asyncio.run(_call_adk_agent(BILLING_ADK_URL, query, "BillingResolutionADK"))
    except Exception as e:
        return f"[BillingResolutionADK] Error: {e}"
