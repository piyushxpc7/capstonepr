"""LangGraph shared state definition."""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    user_query: str
    agent_context: str           # accumulated specialist outputs
    execution_trace: list[dict]  # [{"worker": "PolicyRAG", "output": "..."}]
    final_response: str
