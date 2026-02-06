"""
Test fixtures for CSO scenario tests.

Requires the Docker stack to be running:
    docker compose up -d --build
"""

import pytest
import httpx


@pytest.fixture(scope="session")
def base_url():
    return "http://localhost:9001"


@pytest.fixture(scope="session")
def client(base_url):
    """Synchronous httpx client for the orchestrator API."""
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
