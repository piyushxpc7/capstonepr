# Prodapt AI Operations Center — Complete Project Explanation

---

# PART 1: EXPLAINING IT TO A NON-TECHNICAL PERSON

---

## What Is This Project?

Imagine you work at a phone company's customer support center. Every day, customers call in with all kinds of problems:

- "My 5G keeps dropping near downtown Austin — what's going on?"
- "I got charged twice this month. I want a refund."
- "Can I use my plan while travelling in Europe?"
- "There was a big network outage last week — do I get a discount on my bill?"

Now imagine you have **five expert colleagues** sitting at desks around you:

1. A **Policy Expert** who has read every company rulebook and can instantly answer any policy question
2. A **Network Data Analyst** who has access to every outage, every tower, every performance report and can pull stats in seconds
3. A **Network Engineer** who can remotely diagnose any specific cell tower and tell you exactly what's wrong
4. A **Billing Specialist** who can look up any customer's account, find duplicate charges, and issue refunds on the spot
5. A **Communications Writer** who takes all their findings and turns it into a clean, professional email to send to the customer

This project builds all five of those experts as **AI agents** — software that thinks and acts on its own. And it builds a **smart manager** on top who reads your question, decides which experts need to be involved, sends the query to each of them, waits for all the answers, and then has the writer package everything into one professional response.

You type your question. In about 30 seconds, you get a complete answer — as if a whole team of specialists worked on it together.

---

## How Does It Work? (A Simple Step-by-Step Walk-Through)

Let's walk through a real example:

**Customer question:** *"We had a 6-hour outage in the Midwest. Am I eligible for an SLA credit?"*

*(An SLA credit is a refund or discount the company owes you if the network was down for too long — it's a legal commitment in the service contract.)*

### Step 1: You type the question into the app

There's a simple website (think of it like a Google search box for customer inquiries). You type the question and press Submit.

### Step 2: The Manager reads the question

An AI called the **Supervisor** reads the question. It understands that this question has TWO parts:
- "Was there really a 6-hour outage in the Midwest?" — that needs real data from the network records
- "Does a 6-hour outage qualify for a credit under the SLA policy?" — that needs the policy rulebook

So the Manager says: "I need BOTH the Network Analyst AND the Policy Expert on this one."

### Step 3: The Network Analyst runs

The Network Analyst searches the company's database for Midwest outages. It finds: yes, there was a CRITICAL outage in the Midwest that lasted 6 hours and 14 minutes, affecting 12,000 customers. It reports this back to the Manager.

### Step 4: The Policy Expert runs

The Policy Expert searches the company rulebook. It finds the SLA policy document which says: "For outages exceeding 4 hours with CRITICAL severity, customers on Business and Enterprise plans are eligible for a 20% credit on their monthly bill."

### Step 5: The Communications Writer takes over

Now the Writer has everything: the outage data AND the policy rules. It drafts a professional response:

> *"Dear Customer, we sincerely apologize for the network disruption you experienced in the Midwest. Our records confirm a CRITICAL outage on [date] lasting 6 hours and 14 minutes affecting your area. Per our SLA policy, Business and Enterprise customers are eligible for a 20% credit on their monthly bill for outages exceeding 4 hours of CRITICAL severity. We have applied a credit of $X to your account (reference: CR-A1B2C3D4). Thank you for your patience."*

A second AI on the writing team then **quality-checks** the draft — verifying the tone is empathetic, the facts are correct, and the policy is applied correctly — before it's finalized.

### Step 6: You see the result

The final polished response appears on your screen, along with a step-by-step log showing exactly which "experts" ran and what they found.

---

## The Five Scenarios This System Handles

| Customer Problem | Which Expert(s) Respond |
|---|---|
| "What is the roaming policy for Europe?" | Policy Expert |
| "Which region had the worst outages this month?" | Network Data Analyst |
| "My 5G keeps dropping near tower TX-512 in Austin" | Network Engineer (live tower check) |
| "Customer CUST-10002 was charged twice — apply a refund" | Billing Specialist (actually processes the refund) |
| "Midwest outage — am I eligible for an SLA credit?" | Network Analyst + Policy Expert together |

---

## Why Is This Impressive?

- **It's not a simple chatbot.** It doesn't just make up an answer. Every response is grounded in real company documents and real database records.
- **It handles complexity.** Questions that need multiple kinds of expertise are handled by multiple agents working together, not just one.
- **It actually takes action.** The billing agent doesn't just say "you should get a refund" — it actually writes the credit into the database with a reference number.
- **It knows its own limits.** If a network service is offline, it tells you clearly instead of guessing.

---
---

# PART 2: EXPLAINING IT TO A TECHNICAL PERSON

---

## System Overview

This is a **Python-based multi-agent AI orchestration system** that combines four AI frameworks — LangGraph, LlamaIndex, Google ADK (Agent Development Kit), and CrewAI — into a unified telecom operations center. The UI is Streamlit. The backend data store is SQLite. The LLM backbone is GPT-4o-mini (via OpenAI API) for orchestration and writing, and Gemini 2.0 Flash (via Google Generative AI) for the ADK microservice agents.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit UI  (ui/app.py)                       │
│         - text_area input, st.button submit                         │
│         - sidebar: DB/vector-index/ADK service health checks        │
│         - renders final_response + execution_trace from state       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  calls run_telecom_assistant(query)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LangGraph Supervisor Loop  (orchestration/graph.py)    │
│                                                                     │
│   StateGraph with AgentState (TypedDict):                           │
│     messages, next, user_query, agent_context, execution_trace,     │
│     final_response                                                   │
│                                                                     │
│   Entry → supervisor_node → conditional_edges → worker nodes        │
│   Each worker → back to supervisor (looped until FINISH)            │
└──────┬──────────┬───────────┬───────────┬──────────────────────────┘
       │          │           │           │
       ▼          ▼           ▼           ▼
  PolicyRAG  NetAnalytics  NetDiagADK  BillingADK   CustomerCommsCrew
 (LlamaIndex)(LlamaIndex) (Google ADK)(Google ADK)     (CrewAI)
  doc RAG    semantic SQL   port 8001   port 8002    draft → QA review
```

---

## The State Machine: LangGraph

**File:** `orchestration/graph.py` | `orchestration/state.py`

### AgentState (TypedDict)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # append-only via reducer
    next: str                  # routing decision from supervisor
    user_query: str            # original user input, immutable
    agent_context: str         # accumulated plaintext from all specialists
    execution_trace: list[dict]  # [{worker: str, output: str}] for UI display
    final_response: str        # set only by CustomerCommsCrew
```

`messages` uses `operator.add` as an annotation so LangGraph knows to append rather than overwrite on each state update — this is the standard LangGraph reducer pattern.

### Graph Topology

```
supervisor ──(conditional edges)──> PolicyRAG
                                 ──> NetworkAnalytics
                                 ──> NetworkDiagnosticsADK
                                 ──> BillingResolutionADK
                                 ──> CustomerCommsCrew
                                 ──> END

Each worker node ──> supervisor  (unconditional back-edge)
```

This forms a **hub-and-spoke loop**: the supervisor always regains control after each worker, re-evaluates accumulated context, and decides the next step. `recursion_limit=20` on `.invoke()` prevents runaway loops.

### Supervisor Node — Routing Logic (3-tier priority)

The `_supervisor_node` function uses a layered decision strategy:

**Tier 1 — Hard Python guards (no LLM call):**
```python
if "CustomerCommsCrew" in already_run:
    return {"next": "FINISH"}
if already_run & SPECIALISTS and not (SPECIALISTS - already_run):
    return {"next": "CustomerCommsCrew"}  # all 4 specialists ran → force comms
```

**Tier 2 — Regex-based required-specialist enforcement:**
```python
def _required_specialists(query: str) -> set[str]:
    q = query.lower()
    required = set()
    if any(k in q for k in ("sla", "eligible", "policy", "roaming", "upgrade")):
        required.add("PolicyRAG")
    if any(k in q for k in ("outage", "latency", "packet loss", "throughput")):
        required.add("NetworkAnalytics")
    return required
```
This catches the multi-domain routing failure mode where GPT-4o-mini routes to one specialist and immediately tries to go to CustomerCommsCrew.

**Tier 3 — LLM structured output routing:**
```python
class RouterOutput(BaseModel):
    next: WORKER_NAMES   # Literal type enforces valid choices

structured_llm = llm.with_structured_output(RouterOutput)
response: RouterOutput = structured_llm.invoke([system_prompt, user_query])
```
`with_structured_output` uses function calling under the hood to guarantee the LLM can only return a valid worker name — no free-text parsing needed.

**Post-LLM safety net:**
```python
if chosen in already_run:
    chosen = "CustomerCommsCrew" if (already_run & SPECIALISTS) else "FINISH"
if chosen == "FINISH" and "CustomerCommsCrew" not in already_run:
    chosen = "CustomerCommsCrew"
```

---

## PolicyRAG: LlamaIndex Vector RAG

**File:** `llamaindex_rag/document_rag.py`

### Pipeline

1. **Documents** — Six `.txt` files in `data/documents/`: billing disputes policy, SLA policy, roaming policy, device upgrade policy, 5G FAQ, network outage procedures.

2. **Embedding model** — `BAAI/bge-small-en-v1.5` loaded locally via `llama_index.embeddings.huggingface.HuggingFaceEmbedding`. No API call for embeddings — runs entirely on CPU. Vectors are 384-dimensional.

3. **Index** — `VectorStoreIndex` built from `SimpleDirectoryReader`. Persisted to `data/vector_index/` as JSON files (docstore, vector store, index store, graph store) via LlamaIndex's `StorageContext`. On subsequent runs it loads from disk — no re-embedding.

4. **Query engine** — `index.as_query_engine(similarity_top_k=3)`: the query is embedded, the 3 most similar document chunks are retrieved, and GPT-4o-mini synthesizes a response from those chunks.

```python
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAI(model="gpt-4o-mini")
# index persisted to data/vector_index/
engine = index.as_query_engine(similarity_top_k=3)
response = engine.query(question)
```

---

## NetworkAnalytics: LlamaIndex Semantic SQL

**File:** `llamaindex_rag/sql_semantic_search.py`

### Pipeline

This is **Text-to-SQL with semantic table routing** — not direct SQL generation over all tables.

1. **SQLAlchemy engine** — wraps `data/telecom_ops.db` (SQLite). LlamaIndex's `SQLDatabase` wraps this.

2. **Table descriptions** — Each of the 9 tables is annotated with a human-readable schema description string. These descriptions are embedded into a `VectorStoreIndex` via `ObjectIndex`.

3. **Semantic table selection** — When a query comes in, it's embedded and matched against the table description vectors (`similarity_top_k=2`). Only the 2 most relevant tables are passed to the SQL generation step — preventing the LLM from having to reason over all 9 tables.

4. **SQL generation + execution** — `SQLTableRetrieverQueryEngine` takes the selected tables, generates SQL (GPT-4o-mini), runs it against SQLite, and synthesizes a natural-language answer from the results.

```python
obj_index = ObjectIndex.from_objects(
    table_schema_objs,       # [SQLTableSchema(name, description)]
    table_node_mapping,
    VectorStoreIndex,
)
_query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=2),
)
```

Example: "Which region had the most CRITICAL outages?" → embeddings match `network_outages` table → SQL generated: `SELECT region, COUNT(*) FROM network_outages WHERE severity='CRITICAL' GROUP BY region ORDER BY COUNT(*) DESC` → natural language answer returned.

---

## NetworkDiagnosticsADK & BillingResolutionADK: Google ADK Microservices

**Files:** `adk-services/network_diagnostics/agent.py`, `adk-services/billing_resolution/agent.py`

### What Google ADK Is

Google's Agent Development Kit is a framework for building agents with tool use, backed by Gemini models. The key feature used here is **A2A (Agent-to-Agent) protocol** — each agent exposes an HTTP server that any other agent (in any framework) can call using a standardized JSON protocol.

### Agent Setup

```python
root_agent = Agent(
    name="network_diagnostics_agent",
    model="gemini-2.0-flash-exp",
    description="Prodapt NOC specialist for tower and connectivity diagnostics.",
    instruction="You are a Network Operations Center specialist...",
    tools=[check_tower_status, run_connectivity_diagnostics, get_regional_network_summary],
)

# Expose as A2A HTTP server
app = to_a2a(root_agent, host="localhost", port=8001, protocol="http")
uvicorn.run(app, host="0.0.0.0", port=8001)
```

The **agent card** (`/.well-known/agent-card.json`) is a JSON manifest describing the agent's capabilities — standard A2A discovery mechanism.

### Tools (Network Diagnostics)

All three tools are `async` functions that open a SQLite connection, run a query, and return a JSON string:

- `check_tower_status(tower_id)` — JOINs `network_towers`, `tower_performance`, `open_incidents`. Returns current status, tech (LTE/5G), performance metrics, and any open incident.
- `run_connectivity_diagnostics(tower_id, symptom)` — Pulls the same data, then applies **rule-based diagnosis logic** in Python: packet loss > 5% → flag congestion; latency > 100ms → flag backhaul issue; signal < -90 dBm → flag coverage edge. Returns recommendations list.
- `get_regional_network_summary(region)` — GROUP BY query on tower status/technology for a region.

### Tools (Billing Resolution)

- `lookup_billing_account(customer_id)` — JOINs `billing_accounts` + `customer_subscriptions`, returns last 10 charges.
- `check_duplicate_charges(customer_id)` — Queries `billing_charges WHERE is_duplicate_flag = 1`.
- `apply_billing_credit(customer_id, amount, reason)` — Inserts into `billing_credits`. If `amount <= $50`, status is `APPLIED` and `billing_accounts.current_balance` is decremented via `UPDATE`. If `amount > $50`, status is `PENDING_APPROVAL` — no balance change. Returns reference number `CR-{uuid hex}`.

The Gemini model decides which tools to call and in what order, based on the query. Gemini sees the tool definitions (docstrings are the schemas) and calls them via function calling.

---

## ADK Remote Client: How LangGraph Calls ADK Agents

**File:** `orchestration/adk_remote_client.py`

This is where the inter-framework communication happens. LangGraph is synchronous; the ADK runner is async. The bridge:

```python
def _run_async(coro) -> str:
    """Run async coroutine in a fresh thread to avoid event loop conflicts."""
    result_holder = {}

    def target():
        result_holder["value"] = asyncio.run(coro)

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=120)
    return result_holder.get("value", "No response.")
```

`asyncio.run()` inside a thread creates a **fresh event loop** isolated from any event loop Streamlit or other code might be running. This solved the event loop conflict that caused hangs.

The actual A2A call uses Google ADK's own `RemoteA2aAgent`:

```python
remote = RemoteA2aAgent(
    name=agent_label,
    agent_card=f"{base_url}/.well-known/agent-card.json",
)
runner = Runner(agent=remote, session_service=InMemorySessionService(), ...)
async for event in runner.run_async(...):
    if event.is_final_response():
        final_text += part.text
```

It fetches the agent card to discover the endpoint, establishes a session (UUID-based, in-memory), sends the query as a `types.Content` message, and streams events back — collecting the final response event.

Before calling, a lightweight health check hits `/.well-known/agent-card.json` with a 3-second timeout. If the service is down, a human-readable error is returned to LangGraph rather than raising an exception.

---

## CustomerCommsCrew: CrewAI Sequential Pipeline

**File:** `orchestration/crew_nodes.py`

### Why CrewAI Here?

CrewAI is used specifically for its **sequential multi-agent task pipeline with context chaining**. The `context=[draft_task]` on `review_task` means the reviewer agent automatically receives the drafter's output as part of its input — no manual passing needed.

### Two-Agent Setup

```
comms_specialist (draft_task) → quality_reviewer (review_task) → final text
```

**Agent 1 — Communications Specialist:**
- Role: translate technical/billing findings into plain customer language
- LLM: GPT-4o-mini, temperature=0.3 (slight creativity for natural prose)
- `allow_delegation=False` — cannot hand off mid-task
- Input: user query + full `agent_context` string (all prior specialist outputs concatenated)

**Agent 2 — Quality Assurance Reviewer:**
- Role: check accuracy, tone, completeness, policy compliance
- `expected_output`: "Final polished customer response text only" — forces the agent to output only the final text, no meta-commentary
- `context=[draft_task]` — receives draft_task output automatically

```python
crew = Crew(
    agents=[comms_specialist, quality_reviewer],
    tasks=[draft_task, review_task],
    process=Process.sequential,
    verbose=False,
)
result = crew.kickoff()
return str(result)
```

`str(result)` on a `CrewOutput` object extracts the final task's output as a string.

---

## The Database

**File:** `data/telecom_ops.db` | Schema: `sql/01_schema.sql`

SQLite database with 9 tables across two domains:

**Network Operations:**
```
network_towers       tower_id (PK), region, city, technology, status, capacity, maintenance
network_outages      outage_id (PK), region, tower_id (FK), start_time, end_time, severity, affected_customers, root_cause
tower_performance    record_id (PK), tower_id (FK), latency_ms, packet_loss_pct, throughput_mbps, signal_dbm, recorded_at
open_incidents       incident_id (PK), tower_id (FK), severity, status, description, eta_hours, opened_at
```

**Billing & Customers:**
```
customer_subscriptions  customer_id (PK), account_type, plan_name, data_limit_gb, monthly_fee, contract_end, region
billing_accounts        customer_id (PK/FK), current_balance, billing_cycle_day
billing_charges         charge_id (PK), customer_id (FK), description, amount, charge_date, is_duplicate_flag
billing_credits         credit_id (PK), customer_id (FK), amount, reason, status, reference_number, applied_at
billing_disputes        dispute_id (PK), customer_id (FK), charge_id (FK), dispute_reason, status, opened_at
```

`is_duplicate_flag` is an INTEGER (0/1) — SQLite boolean. The billing agent's `check_duplicate_charges` queries `WHERE is_duplicate_flag = 1`.

`billing_credits.status` is either `APPLIED` (balance decremented immediately) or `PENDING_APPROVAL` (above $50 auto-approval threshold — no balance change until human approval).

---

## Streamlit UI

**File:** `ui/app.py`

Three things worth noting technically:

1. **Service health checks** — On every page render, `httpx.get(url/.well-known/agent-card.json, timeout=2)` is called for both ADK services. This is synchronous and fast — renders red/green badges in the sidebar.

2. **`st.session_state`** — Results are stored in `st.session_state["last_result"]` so they persist across Streamlit's re-render cycle (Streamlit re-runs the entire script on each interaction).

3. **`run_telecom_assistant` is called inside `st.spinner`** — This blocks the Streamlit thread while the graph runs. The full graph can take 20-40 seconds depending on how many specialists run. Streamlit shows a spinner during this time.

---

## Full Request Lifecycle (End to End, Technical)

1. User submits query via `st.text_area` + `st.button`
2. `run_telecom_assistant(query)` called → builds `AgentState`, calls `graph.invoke(initial_state, config={"recursion_limit": 20})`
3. **LangGraph enters supervisor node** → checks `execution_trace` for already-run workers → `_required_specialists(query)` regex check → LLM structured-output routing call
4. Supervisor returns `{"next": "NetworkAnalytics"}` (example)
5. **LangGraph routes to `_network_analytics_node`** → calls `query_sql(user_query)` → LlamaIndex embeds query → matches `network_outages` table description → GPT-4o-mini generates SQL → SQLite runs it → answer synthesized → node appends to `agent_context` and `execution_trace`
6. `_network_analytics_node` returns updated state → LangGraph routes back to supervisor (unconditional back-edge)
7. Supervisor re-evaluates → `_required_specialists` finds `PolicyRAG` still required → routes to `_policy_rag_node`
8. **`_policy_rag_node`** → `query_policy(user_query)` → LlamaIndex embeds query → finds top-3 chunks from SLA policy document → GPT-4o-mini synthesizes policy answer → appends to state
9. Returns to supervisor → both required specialists have run → LLM routes to `CustomerCommsCrew`
10. **`_customer_comms_crew_node`** → `run_customer_comms_crew(query, agent_context)` → CrewAI kicks off `comms_specialist` (draft) → `quality_reviewer` (review) → `final_response` set in state
11. Returns to supervisor → `CustomerCommsCrew in already_run` → hard guard returns `{"next": "FINISH"}`
12. LangGraph hits `END` node → `graph.invoke()` returns `final_state`
13. `run_telecom_assistant` extracts `final_response`, `execution_trace`, `agent_context` into dict → returned to Streamlit
14. Streamlit renders `st.info(final_response)` + trace table in expander

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| Local HuggingFace embeddings (no API) | No cost per embedding call; `bge-small-en-v1.5` is fast and accurate enough for 6 documents |
| Thread-based async bridge for ADK | Streamlit runs in a context where `asyncio.run()` in the main thread fails; a fresh thread gets a clean event loop |
| Regex pre-filter for required specialists | GPT-4o-mini with structured output is good but can still skip a needed specialist on first pass; deterministic Python guardrails are cheaper and more reliable than retrying the LLM |
| `operator.add` reducer on `messages` | LangGraph state updates are merges, not replacements — without the reducer annotation, each node would overwrite the messages list instead of appending |
| SQLite over Postgres | Zero setup for a capstone demo; the A2A agents open fresh connections per call rather than using a connection pool, which is fine for SQLite's file-lock model |
| `recursion_limit=20` | A safety cap; in practice, even the most complex query (all 4 specialists + comms crew) uses 11 graph steps — well under the limit |