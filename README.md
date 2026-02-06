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

The Scenario Runner at port 9001 has four preset scenario buttons and a "Compare with Mesh" button under each one. Click any Compare button to see a side-by-side split view of CSO vs. mesh results.

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

This runs ~25 assertions across 8 test classes:
- `TestScenario1SingleBenefit` — single benefit allocation
- `TestScenario2TierGatedDenial` — tier-gated denial + valid benefit
- `TestScenario3MultiIntentCompromise` — checkout clamp + drink voucher + wine escalation
- `TestScenario4ProactiveRecovery` — room query + reassign to room 101
- `TestComparison1SingleBenefit` — both CSO and mesh succeed (control)
- `TestComparison2TierGatedDenial` — CSO outperforms mesh (breakfast preserved)
- `TestComparison3MultiIntent` — CSO has drink voucher, mesh does not
- `TestComparison4ProactiveRecovery` — CSO assigns room 101, mesh assigns room 201

### Run mesh-only tests:

```bash
pytest tests/test_mesh.py -v
```

This runs ~25 assertions across 5 test classes validating that the mesh simulator produces the expected *wrong* outcomes and the comparison structure is correct.

### Run all tests at once:

```bash
pytest tests/ -v
```

### Important notes about tests:

- Tests require the Docker stack to be running (`docker compose up -d --build`).
- Tests hit the live orchestrator at `http://localhost:9001`.
- Each test automatically resets orchestrator memory and database state via `POST /reset` before running.
- CSO scenario tests call Claude's API (requires a valid `ANTHROPIC_API_KEY`), so they incur API usage.
- Comparison tests run both CSO and mesh sequentially with a DB reset between them, so they take roughly twice as long per scenario.
- Test timeout is 90 seconds per request (120 seconds for comparisons).

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
