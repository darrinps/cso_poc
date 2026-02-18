# Cognitive Singularity Orchestrator (CSO) — Proof of Concept

## Thesis

Multi-agent AI architectures lose context at every handoff. This proof of concept demonstrates a measurably better alternative: **centralized AI reasoning with deterministic execution**, where a single LLM pass preserves full request context across domains, policies, and constraints.

The CSO processes complex, multi-intent guest requests through one reasoning pass (Claude Sonnet), producing a deterministic execution plan enforced through policy-gated MCP tool calls. A real Claude-powered agentic mesh (Claude Haiku agents) runs the same scenarios in parallel, providing a controlled, side-by-side comparison that quantifies the context degradation inherent in multi-agent architectures.

**Key finding:** In scenarios requiring cross-domain reasoning (checkout policy + compensatory benefits + room constraints), the CSO consistently preserves context that the mesh loses at Coordinator handoffs — measurable in missed compensatory actions, incorrect room assignments, and cascaded denials.

---

## Architecture

```
  dashboard-net           orchestration-net           backend-net
+--------------+       +------------------+       +--------------+
|  Dashboard   |       |  CSO             |       |  Postgres    |
|  (Streamlit) +--+ +--+  Orchestrator    +--+ +--+  Database    |
|  :9501       |  | |  |  (FastAPI) :9001 |  | |  |  (Layer 7)  |
+--------------+  | |  +------------------+  | |  +--------------+
                  | |                        | |
                  | |  +------------------+  | |
                  | +--+  MCP Gateway     +--+ |
                  +----+  (SSE) :9000     +----+
                       +------------------+
```

### Design Principles

| Principle | Implementation |
|---|---|
| **Network-enforced policy gates** | The orchestrator has no route to the database. All writes pass through the MCP gateway's policy enforcement layer. This is a Docker network-level guarantee, not a code convention. |
| **Centralized reasoning, deterministic execution** | Claude reasons once; the orchestrator executes the resulting Canonical Intent Envelope sequentially via MCP tool calls. No agent-to-agent negotiation. |
| **Immutable audit trail** | Every policy decision — approvals, denials, compromises, escalations — is captured as a Decision Breadcrumb with trace ID, policy reference, and timestamp. |
| **Three-block memory with TTL decay** | Core (permanent), Tactical (48h), Transient (10min). Zombie context is scrubbed before every reasoning cycle to prevent stale facts from influencing new decisions. |
| **Defense in depth** | Policy enforcement occurs at two levels: the CSO's reasoning engine (Cognitive Fallback) and the MCP gateway (database-level constraints). |

### Layer Architecture

| Layer | Component | Responsibility |
|---|---|---|
| **Layer 2** | Streamlit Dashboard | Executive demo UI with memory vault visualization and live breadcrumb feed |
| **Layer 4** | CSO Orchestrator (FastAPI) | Claude reasoning, Cognitive Fallback, envelope execution, memory management |
| **Layer 5** | MCP Gateway (FastMCP/SSE) | Tool-level policy enforcement, Decision Breadcrumb logging, database mutations |
| **Layer 6** | Decision Breadcrumbs | Immutable audit records for every tool call (cross-cutting via decorator) |
| **Layer 7** | Postgres 16 | System of record — guest profiles, stays, room inventory, allocated benefits |

---

## The CSO Pipeline

Each guest request passes through five phases:

```
Phase 0: SCRUB      →  Purge expired memory facts (prevent zombie context)
Phase 1: DECOMPOSE  →  Claude decomposes request into structured sub-intents
Phase 2: REASON     →  Cognitive Fallback cascade: no-tool → tier-gate → ceiling → execute
Phase 3: EXECUTE    →  Dispatch Canonical Intent Envelope via MCP tool calls
Phase 4: MEMORIZE   →  Store facts in core/tactical/transient blocks, expire transient state
```

### Cognitive Fallback Cascade

When a sub-intent cannot be fulfilled as requested, the CSO applies a priority-ordered fallback:

1. **No MCP tool exists** (e.g., wine delivery) → Escalate to human staff
2. **Tier gate violation** (e.g., Gold guest requesting Diamond benefit) → Deny + escalate
3. **Policy ceiling exceeded** (e.g., 5 PM checkout when max is 4 PM) → Clamp to ceiling + issue compensatory benefit
4. **All within policy** → Execute normally

Each branch produces a structured audit trail, making every outcome explainable and reviewable.

---

## CSO vs. Agentic Mesh Comparison

The mesh pipeline uses **real Claude Haiku agents** (not hardcoded simulations) that communicate through text summaries. Context degrades naturally because each Coordinator handoff compresses the previous agent's output to 2-3 sentences.

### Agent Chains by Scenario

| Scenario | Mesh Agent Chain | Expected Degradation |
|---|---|---|
| Single Benefit | Profile → Coord → Loyalty (3 agents) | None — control case |
| Tier-Gated Denial | Profile → Coord → Reservation → Coord → Loyalty (5 agents) | Coordinator interprets denial as blanket rejection, drops breakfast |
| Multi-Intent Compromise | Profile → Coord → Res → Coord → Loyalty → Coord (6 agents) | Coordinator reports "4PM confirmed" without mentioning 5PM was denied; no voucher |
| Proactive Recovery | Profile → Coord → Rooms (3 agents) | Pet details (breed, weight, near_exit) compressed out of handoff |
| VIP Concierge Bundle | Profile → Coord → Res → Coord → Rooms → Coord → Loyalty (7 agents) | 3 Coordinator compressions; voucher almost always lost, room constraints may degrade |
| Contradictory Intent | Profile → Coord → Reservation (3 agents) | Agents process contradictory requests independently without detecting conflict |
| Ambiguous Escalation | Profile → Coord (2 agents) | Coordinator may over-interpret vague sentiment |
| Mesh-Favorable Baseline | Profile → Coord (2 agents) | None — included for intellectual honesty |

### Quantitative Scoring

Both pipelines are scored against predefined criteria using a weighted multi-dimensional scorecard:

| Dimension | Weight | What It Measures |
|---|---|---|
| Intent Detection | 25% | Did the system identify all sub-intents in the request? |
| Tool Accuracy | 35% | Did it call the correct MCP tools with correct parameters? |
| Escalation Accuracy | 20% | Did it correctly flag unresolvable items for human staff? |
| Compromise Detection | 20% | Did it detect policy gaps and issue compensatory benefits? |

Policy violations are penalized as deductions, ensuring errors are never masked by high sub-scores.

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **CSO Reasoning** | Claude Sonnet 4 (temp=0) | Deterministic single-pass decomposition with deep policy reasoning |
| **Mesh Agents** | Claude Haiku 4.5 (temp=0.2) | Realistic small specialist agents with natural variation |
| **Orchestrator API** | FastAPI + Uvicorn | Async HTTP with embedded single-page scenario runner UI |
| **MCP Gateway** | FastMCP (SSE transport) | Anthropic's Model Context Protocol for tool execution with policy enforcement |
| **Database** | Postgres 16 (Alpine) | ACID guarantees for guest data, idempotent benefit allocation via UNIQUE constraints |
| **Dashboard** | Streamlit | Rapid prototyping for memory vault visualization and breadcrumb feeds |
| **Test Framework** | pytest + httpx | Integration tests against live Docker stack with structural assertions |
| **Containerization** | Docker Compose v2 | Three isolated bridge networks enforce write-path security at the infrastructure level |

---

## Getting Started

### Prerequisites

- **Docker Desktop** (with Docker Compose v2)
- **Python 3.11+** (for running tests locally)
- **Anthropic API key** (Claude powers both CSO reasoning and mesh agents)

### 1. Set Your API Key

```bash
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

Docker Compose reads this file automatically and passes the key to the orchestrator container.

### 2. Build and Start the Stack

```bash
docker compose up -d --build
```

This builds three images and starts four containers:

| Container | Port | Role |
|---|---|---|
| `mock-pms-db` | (internal) | Postgres 16 — guest/stay/room seed data |
| `mcp-hospitality-gateway` | `localhost:9000` | MCP tools + business policy enforcement |
| `cso-orchestrator` | `localhost:9001` | FastAPI — Claude reasoning + mesh comparison |
| `cso-dashboard` | `localhost:9501` | Streamlit UI — multi-turn chat demo |

### 3. Verify Health

```bash
docker compose ps
curl http://localhost:9001/health
# Expected: {"status":"ok","turns_processed":0}
```

If the health check fails, wait 10 seconds — services start in dependency order (Postgres → MCP Gateway → Orchestrator → Dashboard).

### 4. Open the UI

- **Scenario Runner:** [http://localhost:9001](http://localhost:9001) — Eight scenario buttons with side-by-side CSO vs. mesh comparison and quantitative scorecard
- **Streamlit Dashboard:** [http://localhost:9501](http://localhost:9501) — Multi-turn chat with live breadcrumb feed and memory vault inspector

---

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# All CSO scenario tests + comparison tests
pytest tests/test_scenarios.py -v

# Mesh agent pipeline tests
pytest tests/test_mesh.py -v

# Everything
pytest tests/ -v
```

### Test Design Philosophy

- **Integration tests against live Docker stack** — validates the full pipeline (Claude → MCP → Postgres) end-to-end
- **Autouse reset fixture** — every test gets clean memory and database state via `POST /reset`
- **CSO assertions are strict** — deterministic pipeline (temp=0) allows exact checks on tool names, room numbers, benefit types
- **Mesh assertions are loosened** — real Haiku agents (temp=0.2) are non-deterministic; tests check structural properties, not exact values
- **Informational-only mesh checks** — some observations are logged but not asserted, preventing flaky tests while preserving evidence
- **Timeouts:** 180s per scenario, 300s for comparisons (real Claude API calls)

---

## Test Reference

### CSO Pipeline Tests (`tests/test_scenarios.py`)

Tests validate that the CSO's single-pass reasoning correctly decomposes, reasons about, and executes guest requests. All assertions are strict.

#### Scenario 1: Single Benefit (`TestScenario1SingleBenefit`)

**Guest G-2002 (Gold)** asks for complimentary breakfast. Baseline scenario — single intent, no policy conflicts.

| Test | Assertion |
|---|---|
| `test_status_executable` | Envelope status is `Executable` |
| `test_single_action` | Exactly 1 action produced |
| `test_action_is_loyalty_allocate` | Action calls `loyalty_allocate_benefit` |
| `test_benefit_type_breakfast` | Benefit type is `ComplimentaryBreakfast` |
| `test_no_escalations` | No escalation notes |
| `test_no_compromises` | No actions flagged as compromises |

#### Scenario 2: Tier-Gated Denial (`TestScenario2TierGatedDenial`)

**Guest G-2002 (Gold)** asks for late checkout at 3 PM AND breakfast. CSO denies checkout (Gold can't extend past 11 AM) while independently fulfilling the breakfast request.

| Test | Assertion |
|---|---|
| `test_status_includes_escalation` | `Human_Escalation_Required` or `Partial_Fulfillment` |
| `test_breakfast_allocated` | Breakfast allocated despite checkout denial |
| `test_late_checkout_denied_or_escalated` | Denial in escalation notes, assertions, or sub-intent violations |
| `test_mesh_annotation_present` | Mesh comparison annotation populated |

#### Scenario 3: Multi-Intent Compromise (`TestScenario3MultiIntentCompromise`)

**Guest G-1001 (Diamond)** asks for 5 PM checkout, Suite Night Award, and Chateau Margaux. CSO clamps checkout to 4 PM, issues compensatory drink voucher, allocates SNA, and escalates wine to staff.

| Test | Assertion |
|---|---|
| `test_status_human_escalation` | `Human_Escalation_Required` (wine needs staff) |
| `test_checkout_clamped_to_4pm` | Checkout contains `T16:00` |
| `test_drink_voucher_compromise` | `ComplimentaryDrinkVoucher` issued, flagged as compromise |
| `test_suite_night_award_allocated` | SNA allocated normally |
| `test_wine_escalated` | Wine/Margaux in escalation notes or assertions |
| `test_compromise_breadcrumbs` | Breadcrumb references `COMPROMISE` or `CEILING` |

#### Scenario 4: Proactive Recovery (`TestScenario4ProactiveRecovery`)

**Guest G-3003 (Titanium)** with 2 Cane Corso dogs (225 lbs). Flight delayed +3h, arriving 1 AM. CSO proactively reassigns from floor 14 to ground-floor pet-friendly suite near exit.

| Test | Assertion |
|---|---|
| `test_status_partial_fulfillment` | `Partial_Fulfillment` or `Executable` |
| `test_room_query_executed` | `pms_query_rooms` executed |
| `test_reassigned_to_room_101` | Reassigned to room `101` |
| `test_old_room_was_1415` | Old room was `1415` |
| `test_core_memory_has_pet_facts` | Core memory contains pet/dog/Cane Corso facts |
| `test_mesh_annotation_explains_telephone_game` | Annotation mentions context degradation |

#### Scenario 5: VIP Concierge Bundle (`TestScenario5VIPConciergeBundle`)

**Guest G-1001 (Diamond)** requests checkout extension, room change for visiting service dog, Suite Night Award, and breakfast. The hardest scenario — 6 actions in one pass.

| Test | Assertion |
|---|---|
| `test_checkout_clamped_to_4pm` | Checkout clamped to `T16:00` |
| `test_drink_voucher_compromise` | Compensatory voucher issued |
| `test_suite_night_award` | SNA allocated |
| `test_breakfast_allocated` | Breakfast allocated |
| `test_room_query_executed` | Room query executed |
| `test_reassigned_to_room_101` | Reassigned to room `101` |

#### Scenarios 6-8: Adversarial and Control Cases

| Scenario | Key Test | What It Proves |
|---|---|---|
| **Contradictory Intent** — late checkout + early check-in same day | `test_no_conflicting_actions_executed` | CSO detects logical contradiction; mesh processes independently |
| **Ambiguous Escalation** — vague complaint with no actionable intent | `test_no_hallucinated_actions` | CSO escalates to staff; mesh may hallucinate tool calls |
| **Mesh-Favorable Baseline** — "What time is checkout?" | `test_no_mutation_tools_called` | Both architectures handle simple queries equivalently (intellectual honesty) |

### CSO vs. Mesh Comparison Tests (`tests/test_scenarios.py`)

| Comparison | Key Assertion | CSO Advantage |
|---|---|---|
| **Single Benefit** | `both_succeed: true` | None — control case |
| **Tier-Gated Denial** | `cso_action_count >= mesh_action_count` | CSO preserves independent intents |
| **Multi-Intent** | `cso_has_voucher: true` | CSO issues compensatory drink voucher |
| **Proactive Recovery** | `cso_room: "101"` | CSO preserves all constraint parameters |
| **VIP Bundle** | `cso_has_voucher + cso_room: "101"` | CSO handles 5 sub-intents in one pass |

### Mesh Agent Pipeline Tests (`tests/test_mesh.py`)

Validates real Haiku-powered agents produce structurally sound results. Assertions are loosened for non-determinism; some checks are informational-only.

| Test Class | Agent Chain | Key Checks |
|---|---|---|
| `TestMeshScenario1` | 3 agents | Has breakfast, status Executable |
| `TestMeshScenario2` | 5 agents | Successful actions <= CSO count |
| `TestMeshScenario3` | 6 agents | Has checkout + SNA; voucher logged but not asserted |
| `TestMeshScenario4` | 3 agents | Has room reassignment (any room); CSO room 101 strict |
| `TestMeshScenario5` | 7 agents | At least 5 degradation entries; CSO has 5+ actions |
| `TestMesh6-8` | 2-3 agents | Structure validation, degradation chain populated |

---

## API Reference

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check with turn count |
| `POST` | `/reset` | Reset memory + database to seed state |
| `POST` | `/scenario/{name}` | Run CSO pipeline for named scenario |
| `POST` | `/scenario/{name}/mesh` | Run mesh pipeline only |
| `POST` | `/scenario/{name}/compare` | Side-by-side CSO vs. mesh comparison |
| `GET` | `/breadcrumbs` | Fetch Decision Breadcrumbs |
| `GET` | `/memory-vault` | Three-block memory state with TTL data |

### Available Scenarios

| Name | Guest | Description |
|---|---|---|
| `single_benefit` | G-2002 (Gold) | Complimentary breakfast |
| `tier_gated_denial` | G-2002 (Gold) | Late checkout denied + breakfast allocated |
| `multi_intent_compromise` | G-1001 (Diamond) | 5PM checkout clamped + drink voucher + wine escalated |
| `proactive_recovery` | G-3003 (Titanium) | Flight delay + 2 dogs + room reassignment |
| `vip_concierge_bundle` | G-1001 (Diamond) | All capabilities combined (7-agent mesh stress test) |
| `contradictory_intent` | G-1001 (Diamond) | Logically impossible request — contradiction detection |
| `ambiguous_escalation` | G-2002 (Gold) | Vague complaint — no hallucinated actions |
| `mesh_favorable_baseline` | G-2002 (Gold) | Simple lookup — both architectures equivalent |

---

## Operations

### Stopping the Stack

```bash
docker compose down        # Stop containers, preserve data
docker compose down -v     # Stop containers + remove Postgres volume (full reset)
```

### Rebuilding After Code Changes

```bash
docker compose up -d --build
```

### Troubleshooting

| Symptom | Resolution |
|---|---|
| Container exits immediately | `docker compose logs cso-orchestrator` / `docker compose logs mcp-hospitality-gateway` |
| Tests fail with connection errors | Verify stack is running: `docker compose ps`, then `curl http://localhost:9001/health` |
| "Duplicate benefit" / stale state | `curl -X POST http://localhost:9001/reset` (autouse fixture should handle this) |
| Claude API errors (401/429) | Verify key: `docker compose exec cso-orchestrator printenv ANTHROPIC_API_KEY` |

---

## Project Structure

```
cso_poc/
  model_config.py     Centralized model version pinning (Sonnet for CSO, Haiku for mesh)
  schemas.py          Canonical Intent Envelope, ProposedAction, Decision Breadcrumb
  memory.py           Three-block memory with TTL decay (core/tactical/transient)
  reasoning.py        Claude decomposition engine with policy-encoded system prompt
  orchestrator.py     FastAPI app — Cognitive Fallback, envelope execution, embedded UI
  mcp_server.py       MCP gateway — policy enforcement, breadcrumb decorator, DB mutations
  mesh_agents.py      Haiku agent infrastructure — tool-use loop, agent prompts, handoffs
  mesh.py             Agent chain pipelines per scenario, structural comparison builder
  scenarios.py        8 scenario configs + run_scenario pipeline with two-phase room handling
  scorecard.py        Weighted multi-dimensional scoring (intent/tool/escalation/compromise)
  dashboard.py        Streamlit UI — chat, breadcrumb feed, memory vault inspector
  db.py               Async Postgres connection pool (singleton, lazy initialization)
tests/
  conftest.py         Fixtures: session client, autouse reset, helper functions
  test_scenarios.py   CSO pipeline tests (strict) + comparison tests
  test_mesh.py        Mesh agent tests (loosened) + structural validation
db/
  init.sql            Schema + seed data (3 guests, 4 stays, 6 rooms)
docker-compose.yml    4 services, 3 networks, health checks
```
