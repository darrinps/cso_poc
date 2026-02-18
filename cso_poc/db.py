"""
Layer 7 adapter — async Postgres connection pool for the MCP gateway.

The gateway is the *only* service with a route to the database network.

Architectural Decision: Connection pool (not per-query connections)
  MCP tools fire rapidly during envelope execution (up to 6 tool calls
  per scenario).  A pool avoids the overhead of TCP+TLS handshake per
  query and prevents connection exhaustion under concurrent requests.

Architectural Decision: Singleton module-level pool
  The pool is lazily created on first use and shared across all MCP
  tool handlers.  This matches the FastMCP server's single-process
  model and avoids the complexity of dependency injection.
"""

from __future__ import annotations

import os

import asyncpg

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://cso:cso_secret@mock-pms-db:5432/pms",
    )


async def get_pool() -> asyncpg.Pool:
    """Lazy initialization: pool is created on first call, reused thereafter."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(_dsn(), min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
