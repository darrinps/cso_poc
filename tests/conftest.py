"""
Test fixtures for CSO scenario tests.

Requires the Docker stack to be running:
    docker compose up -d --build

Architectural Decision: Integration tests against live Docker stack
  Tests hit the real orchestrator over HTTP rather than mocking MCP calls.
  This validates the full pipeline (Claude → MCP → Postgres) end-to-end,
  catching integration issues that unit tests would miss (network timeouts,
  policy enforcement at the gateway, database constraint violations).

Architectural Decision: Autouse reset fixture
  Every test gets a clean slate: memory cleared, database reset to seed
  state, allocated_benefits table truncated.  This prevents state leakage
  between tests (e.g., a duplicate-benefit rejection because a previous
  test already allocated the same benefit).

Architectural Decision: High timeouts (180s / 300s)
  Each scenario involves real Claude API calls (Sonnet for CSO, Haiku for
  mesh agents).  Comparison tests run both pipelines sequentially with a
  DB reset between them, so they need roughly 2x the single-scenario timeout.
"""

import os

import pytest
import httpx


@pytest.fixture(scope="session")
def base_url():
    return os.environ.get("CSO_TEST_BASE_URL", "http://localhost:9001")


@pytest.fixture(scope="session")
def client(base_url):
    """Session-scoped client reuses the TCP connection across all tests."""
    with httpx.Client(base_url=base_url, timeout=180.0) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state(client):
    """Reset orchestrator + DB state before each test."""
    resp = client.post("/reset")
    assert resp.status_code == 200, f"Reset failed: {resp.text}"
    yield


def run_scenario(client: httpx.Client, name: str) -> dict:
    """Helper to run a named scenario and return the response dict."""
    resp = client.post(f"/scenario/{name}")
    assert resp.status_code == 200, f"Scenario {name} failed: {resp.text}"
    data = resp.json()
    assert "error" not in data, f"Scenario {name} error: {data}"
    return data


def run_comparison(client: httpx.Client, name: str) -> dict:
    """Helper to run a CSO-vs-mesh comparison and return the response dict."""
    resp = client.post(f"/scenario/{name}/compare", timeout=300.0)
    assert resp.status_code == 200, f"Comparison {name} failed: {resp.text}"
    data = resp.json()
    assert "error" not in data, f"Comparison {name} error: {data}"
    return data
