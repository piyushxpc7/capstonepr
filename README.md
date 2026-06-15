# nexel

**Telecom AI operations center.** A multi-agent system that triages network issues and customer-care requests for a telecom operator — routing each request to the right specialist agent and surfacing the resolution through an ops console.

**Stack:** Google ADK · LlamaIndex (RAG) · LangGraph · Streamlit · Python

## What it does
- **Orchestration** — a router classifies incoming requests (network fault, billing, roaming, …) and dispatches to specialist agents.
- **RAG knowledge** — LlamaIndex retrieves over policy, network, and product docs to ground answers.
- **Conversation state** — `conversation_manager.py` tracks multi-turn context.
- **Ops console** — Streamlit UI for operators to watch agent reasoning and outcomes.

See [`PROJECT_EXPLANATION.md`](PROJECT_EXPLANATION.md) for a full walkthrough (technical + non-technical).

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your keys
python init_db.py
streamlit run ui/app.py
```

## Layout
| Path | Purpose |
|---|---|
| `adk-services/` | Google ADK agent services |
| `llamaindex_rag/` | RAG indexing & retrieval |
| `orchestration/` | Router + multi-agent graph |
| `conversation_manager.py` | Multi-turn state |
| `ui/` | Streamlit ops console |
| `sql/`, `init_db.py` | Schema & seed |
