# Cognitive Singularity Orchestrator — Proof of Concept

A 4-layer Docker stack demonstrating centralized AI reasoning vs. traditional agentic mesh architectures in a hospitality domain. The CSO processes multi-intent guest requests through a single reasoning pass (Claude), preserving full context — while the mesh simulator shows how context degrades at each agent handoff.

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

- **Orchestrator** has no direct DB access — all writes go through the MCP Gateway's policy gates.
- **Dashboard** can only talk to the Orchestrator's HTTP API.
- **MCP Gateway** is the only service that bridges orchestration and backend networks.

## Prerequisites

- **Docker Desktop** (with Docker Compose v2)
- **Python 3.11+** (for running tests locally)
- **Anthropic API key** (Claude powers the sub-intent decomposition)

## Step 1: Set Your API Key

Create a `.env` file in the project root (if one does not already exist):

```bash
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

Docker Compose reads this file automatically and passes the key to the orchestrator container.

## Step 2: Build and Start the Stack

From the project root directory:

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

## Step 3: Verify the Stack Is Running

Check that all four containers are healthy:

```bash
docker compose ps
```

You should see all four services with status `Up`. Then confirm the orchestrator API responds:

```bash
curl http://localhost:9001/health
```

Expected response:

```json
{"status":"ok","turns_processed":0}
```

If the health check fails, wait 10 seconds and retry — the MCP gateway needs the Postgres healthcheck to pass before it starts, and the orchestrator needs the gateway.

## Step 4: Open the UI

Open your browser to:

- **Scenario Runner (FastAPI):** [http://localhost:9001](http://localhost:9001)
- **Streamlit Dashboard:** [http://localhost:9501](http://localhost:9501)

The Scenario Runner at port 9001 has five preset scenario buttons and a "Compare with Mesh" button under each one. Click any Compare button to see a side-by-side split view of CSO vs. mesh results with a scorecard.

## Step 5: Install Test Dependencies

Tests run on your host machine against the live Docker stack. Install the project and test dependencies:

```bash
pip install -e ".[dev]"
```

This installs `pytest`, `httpx`, and all project dependencies locally.

## Step 6: Run the Tests

### Run all CSO scenario tests (existing + comparison):

```bash
pytest tests/test_scenarios.py -v
```

### Run mesh-only tests:

```bash
pytest tests/test_mesh.py -v
```

### Run all tests at once:

```bash
pytest tests/ -v
```

### Important notes about tests:

- Tests require the Docker stack to be running (`docker compose up -d --build`).
- Tests hit the live orchestrator at `http://localhost:9001`.
- Each test automatically resets orchestrator memory and database state via `POST /reset` before running.
- Both CSO and mesh tests call Claude's API (requires a valid `ANTHROPIC_API_KEY`), so they incur API usage. CSO uses Sonnet for decomposition; the mesh uses Haiku for each agent in the chain.
- Comparison tests run both CSO and mesh sequentially with a DB reset between them, so they take roughly twice as long per scenario.
- Test timeout is 180 seconds per request (300 seconds for comparisons).
- **CSO assertions are strict** — the CSO pipeline is deterministic once Claude produces sub-intents, so we check exact tool names, room numbers, benefit types, and policy references.
- **Mesh assertions are loosened** — real Haiku agents are non-deterministic, so mesh tests check structural properties (has actions, has checkout, has SNA) rather than exact values. Some mesh checks are informational-only (logged but not asserted).

---

## Test Reference

### `tests/test_scenarios.py` — CSO Pipeline Tests

These tests validate that the CSO's single-pass reasoning pipeline correctly decomposes, reasons about, and executes guest requests. All assertions are strict.

#### Scenario 1: Single Benefit Allocation (`TestScenario1SingleBenefit`)

**Setup:** Guest G-2002 (Gold tier) asks for a complimentary breakfast.

This is the simplest scenario — a single intent with no policy conflicts. It serves as a baseline to confirm the pipeline works end-to-end.

| Test | What it checks |
|---|---|
| `test_status_executable` | Envelope status is `Executable` (no compromises or escalations) |
| `test_single_action` | Exactly 1 action produced |
| `test_action_is_loyalty_allocate` | The action calls `loyalty_allocate_benefit` |
| `test_benefit_type_breakfast` | Benefit type is `ComplimentaryBreakfast` |
| `test_no_escalations` | No escalation notes generated |
| `test_no_compromises` | No actions flagged as compromises |

#### Scenario 2: Tier-Gated Denial + Valid Benefit (`TestScenario2TierGatedDenial`)

**Setup:** Guest G-2002 (Gold tier) asks for a late checkout at 3 PM AND complimentary breakfast.

Tests that the CSO correctly denies the checkout (Gold tier doesn't qualify for late checkout — requires Diamond or Titanium) while still fulfilling the independent breakfast request. This is where the CSO's ability to treat each sub-intent independently matters.

| Test | What it checks |
|---|---|
| `test_status_includes_escalation` | Status is `Human_Escalation_Required` or `Partial_Fulfillment` |
| `test_breakfast_allocated` | Breakfast was allocated despite checkout denial |
| `test_late_checkout_denied_or_escalated` | Checkout denial appears in escalation notes, contextual assertions, or sub-intent tier violations |
| `test_mesh_annotation_present` | Mesh comparison annotation is populated |

#### Scenario 3: Multi-Intent Compromise + Escalation (`TestScenario3MultiIntentCompromise`)

**Setup:** Guest G-1001 (Diamond tier) asks for a 5 PM checkout, Suite Night Award, and a bottle of Chateau Margaux.

The most complex reasoning test: the 5 PM checkout exceeds the Diamond policy ceiling of 4 PM, so the CSO must clamp to 4 PM and issue a compensatory drink voucher. The wine request has no MCP tool, so it must be escalated to staff. The SNA should be allocated normally.

| Test | What it checks |
|---|---|
| `test_status_human_escalation` | Status is `Human_Escalation_Required` (wine needs staff) |
| `test_checkout_clamped_to_4pm` | Checkout time contains `T16:00` (clamped from 17:00) |
| `test_drink_voucher_compromise` | A `ComplimentaryDrinkVoucher` was issued and flagged as a compromise |
| `test_suite_night_award_allocated` | SNA was allocated normally |
| `test_wine_escalated` | Wine/Margaux/Provisions appears in escalation notes or contextual assertions |
| `test_compromise_breadcrumbs` | At least one decision breadcrumb references `COMPROMISE` or `CEILING` |

#### Scenario 4: Proactive Cross-Domain Recovery (`TestScenario4ProactiveRecovery`)

**Setup:** Guest G-3003 (Titanium tier) has 2 Cane Corso dogs (225 lbs total). Flight delayed +3 hours, arriving at 1 AM instead of 10 PM. Current room 1415 on floor 14.

Tests proactive reasoning: the CSO detects that a high-floor room is unsuitable for a 1 AM arrival with large dogs, queries for ground-floor pet-friendly suites near the exit, and reassigns to room 101.

| Test | What it checks |
|---|---|
| `test_status_partial_fulfillment` | Status is `Partial_Fulfillment` or `Executable` |
| `test_room_query_executed` | A `pms_query_rooms` action was executed |
| `test_reassigned_to_room_101` | Guest was reassigned to room `101` |
| `test_old_room_was_1415` | The old room in the reassignment was `1415` |
| `test_core_memory_has_pet_facts` | Core memory block contains pet/dog/Cane Corso facts |
| `test_mesh_annotation_explains_telephone_game` | Mesh annotation mentions "telephone" or "context" degradation |

#### Scenario 5: VIP Concierge Bundle (`TestScenario5VIPConciergeBundle`)

**Setup:** Guest G-1001 (Diamond tier) at LHRW01, room 1412. Requests: extend checkout to 5 PM, move to a ground-floor pet-friendly suite near the exit for a visiting service dog, apply Suite Night Award, and add complimentary breakfast.

The hardest scenario — combines all CSO capabilities in a single request: policy ceiling compromise (checkout), two-phase room handling (query then reassign), and multiple benefit allocations. The CSO must produce 6 actions in one reasoning pass.

| Test | What it checks |
|---|---|
| `test_status_partial_or_escalation` | Status is `Partial_Fulfillment`, `Human_Escalation_Required`, or `Executable` |
| `test_checkout_clamped_to_4pm` | Checkout clamped to `T16:00` |
| `test_drink_voucher_compromise` | `ComplimentaryDrinkVoucher` issued as compensation |
| `test_suite_night_award` | SNA allocated |
| `test_breakfast_allocated` | `ComplimentaryBreakfast` allocated |
| `test_room_query_executed` | `pms_query_rooms` was executed |
| `test_reassigned_to_room_101` | Guest reassigned to room `101` |

### `tests/test_scenarios.py` — CSO vs Mesh Comparison Tests

These tests run both pipelines back-to-back with a DB reset in between, then validate the structural comparison.

#### Comparison 1: Single Benefit (`TestComparison1SingleBenefit`)

Control case. Both CSO and mesh should handle a simple breakfast allocation identically.

| Test | What it checks |
|---|---|
| `test_both_succeed` | Comparison reports `both_succeed: true` |
| `test_same_action_counts` | CSO and mesh have identical action counts |

#### Comparison 2: Tier-Gated Denial (`TestComparison2TierGatedDenial`)

The mesh may cascade the checkout denial to all intents, losing the breakfast. CSO preserves independent intents.

| Test | What it checks |
|---|---|
| `test_cso_outperforms_or_matches_mesh` | CSO action count >= mesh action count |

#### Comparison 3: Multi-Intent Compromise (`TestComparison3MultiIntent`)

The mesh's Coordinator compresses the Reservation Agent's "4 PM confirmed" without mentioning the original 5 PM denial, so the Loyalty Agent never issues the compensatory drink voucher.

| Test | What it checks |
|---|---|
| `test_cso_has_voucher` | CSO issued a `ComplimentaryDrinkVoucher` |

#### Comparison 4: Proactive Recovery (`TestComparison4ProactiveRecovery`)

The mesh's Coordinator compresses pet details (breed, weight, count), so the Rooms Agent may query with incomplete constraints and assign the wrong room.

| Test | What it checks |
|---|---|
| `test_cso_assigns_101` | CSO assigned room `101` |

#### Comparison 5: VIP Concierge Bundle (`TestComparison5VIPConciergeBundle`)

The 7-agent mesh chain has 3 Coordinator compressions. The voucher context is almost always lost, and room constraints may degrade.

| Test | What it checks |
|---|---|
| `test_cso_has_voucher` | CSO issued a `ComplimentaryDrinkVoucher` |
| `test_cso_assigns_101` | CSO assigned room `101` |
| `test_cso_outperforms_or_matches_mesh` | CSO action count >= mesh action count |

---

### `tests/test_mesh.py` — Mesh Agent Pipeline Tests

These tests validate that the real Claude Haiku-powered mesh agents produce structurally sound results. Because agents are non-deterministic, mesh assertions are loosened — they check structural properties rather than exact values. Some checks are informational-only (logged but not asserted).

#### Mesh Scenario 1: Single Benefit (`TestMeshScenario1`)

Control case: the 3-agent chain (Profile → Coordinator → Loyalty) should succeed just like CSO.

| Test | What it checks |
|---|---|
| `test_mesh_has_actions` | Mesh produced at least 1 action |
| `test_mesh_has_breakfast` | Mesh allocated `ComplimentaryBreakfast` |
| `test_mesh_status_executable` | Mesh status is `Executable` |
| `test_both_succeed` | Comparison reports both pipelines succeeded |
| `test_degradation_chain_has_entries` | Agent handoff chain is populated |

#### Mesh Scenario 2: Tier-Gated Denial (`TestMeshScenario2`)

The 5-agent chain (Profile → Coordinator → Reservation → Coordinator → Loyalty) may cascade the checkout denial to all intents.

| Test | What it checks |
|---|---|
| `test_mesh_has_fewer_or_equal_actions` | Mesh successful actions <= CSO action count |
| `test_cso_has_breakfast` | CSO preserved breakfast (strict) |
| `test_cso_status_not_rejected` | CSO was not fully rejected |
| `test_degradation_chain_has_entries` | Agent handoff chain is populated |
| `test_context_loss_documented` | Context loss summary is populated |

#### Mesh Scenario 3: Multi-Intent Compromise (`TestMeshScenario3`)

The 6-agent chain should get checkout and SNA, but likely loses the drink voucher because the Coordinator compresses "4 PM confirmed" without mentioning the original 5 PM denial.

| Test | What it checks |
|---|---|
| `test_mesh_has_checkout` | Mesh has a `pms_update_reservation` action |
| `test_mesh_has_sna` | Mesh allocated a `SuiteNightAward` |
| `test_cso_has_voucher` | CSO has `ComplimentaryDrinkVoucher` (strict) |
| `test_mesh_voucher_informational` | Logs whether mesh also got the voucher (not asserted) |
| `test_comparison_cso_has_voucher` | Comparison reports CSO has voucher |
| `test_degradation_chain_has_entries` | Agent handoff chain is populated |

#### Mesh Scenario 4: Proactive Recovery (`TestMeshScenario4`)

The 3-agent chain (Profile → Coordinator → Rooms) should reassign a room, but the Coordinator's lossy compression of pet details may cause the Rooms Agent to use incomplete query constraints.

| Test | What it checks |
|---|---|
| `test_mesh_assigns_some_room` | Mesh has a `pms_reassign_room` with a room number (any room) |
| `test_cso_assigns_room_101` | CSO assigned room `101` (strict) |
| `test_degradation_chain_has_entries` | Agent handoff chain is populated |
| `test_cso_room_informational` | Logs CSO vs mesh room numbers (not asserted) |

#### Mesh Scenario 5: VIP Concierge Bundle (`TestMeshScenario5`)

The 7-agent chain (Profile → Coordinator → Reservation → Coordinator → Rooms → Coordinator → Loyalty) is the hardest stress test. Three Coordinator compressions means significant context loss. The drink voucher is almost always lost, and room constraints may degrade.

| Test | What it checks |
|---|---|
| `test_mesh_has_actions` | Mesh produced at least 2 actions (checkout + SNA minimum) |
| `test_mesh_has_checkout` | Mesh has a `pms_update_reservation` action |
| `test_mesh_has_sna` | Mesh allocated a `SuiteNightAward` |
| `test_mesh_voucher_informational` | Logs whether mesh got the drink voucher (not asserted) |
| `test_mesh_room_informational` | Logs which room mesh assigned (not asserted) |
| `test_degradation_chain_has_entries` | 7-agent chain has at least 5 degradation entries |
| `test_cso_has_all_six_actions` | CSO produced at least 5 actions (strict) |

#### Comparison Structure (`TestComparisonStructure`)

Validates the comparison response dict has all required fields regardless of scenario content.

| Test | What it checks |
|---|---|
| `test_has_cso_key` | Response contains `cso` key |
| `test_has_mesh_key` | Response contains `mesh` key |
| `test_has_comparison_key` | Response contains `comparison` key |
| `test_comparison_has_required_fields` | Comparison has `both_succeed`, `cso_status`, `mesh_status`, `action_diff`, `context_lost`, `cso_advantage` |
| `test_mesh_has_required_fields` | Mesh result has `scenario_name`, `trace_id`, `steps`, `final_actions`, `final_status`, `degradation_chain` |

## Step 7: Test via API (Optional)

You can also exercise the endpoints directly with `curl`:

### Run a single CSO scenario:

```bash
curl -X POST http://localhost:9001/scenario/single_benefit
```

### Run the mesh simulator alone:

```bash
curl -X POST http://localhost:9001/scenario/proactive_recovery/mesh
```

### Run a side-by-side comparison (CSO vs. mesh):

```bash
curl -X POST http://localhost:9001/scenario/proactive_recovery/compare
```

### Reset state between manual runs:

```bash
curl -X POST http://localhost:9001/reset
```

### Available scenario names:

| Name | Guest | Description |
|---|---|---|
| `single_benefit` | G-2002 (Gold) | Complimentary breakfast |
| `tier_gated_denial` | G-2002 (Gold) | Late checkout denied + breakfast allocated |
| `multi_intent_compromise` | G-1001 (Diamond) | 5PM checkout clamped to 4PM + drink voucher + wine escalated |
| `proactive_recovery` | G-3003 (Titanium) | Flight delay + 2 dogs + room reassignment |
| `vip_concierge_bundle` | G-1001 (Diamond) | 5PM checkout + room change + SNA + breakfast (7-agent mesh stress test) |

## Stopping the Stack

```bash
docker compose down
```

To also remove the Postgres data volume (full reset of seed data):

```bash
docker compose down -v
```

## Rebuilding After Code Changes

Any changes to files in `cso_poc/` require a rebuild:

```bash
docker compose up -d --build
```

## Troubleshooting

**Container exits immediately:**
Check logs for the failing service:

```bash
docker compose logs cso-orchestrator
docker compose logs mcp-hospitality-gateway
```

**Tests fail with connection errors:**
The stack isn't running or hasn't finished starting. Run `docker compose ps` to verify all containers show `Up`, then check `curl http://localhost:9001/health`.

**Tests fail with "duplicate benefit" or stale state:**
The `reset_state` fixture should handle this automatically. If running manually, call `curl -X POST http://localhost:9001/reset` before each test.

**Claude API errors (401/429):**
Verify your `ANTHROPIC_API_KEY` is set correctly in `.env` and the orchestrator container can see it:

```bash
docker compose exec cso-orchestrator printenv ANTHROPIC_API_KEY
```
