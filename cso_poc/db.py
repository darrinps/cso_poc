"""
Layer 7 adapter — async Postgres connection pool for the MCP gateway.

The gateway is the *only* service with a route to the database network.
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
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(_dsn(), min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
