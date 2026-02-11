"""
Centralized Model Version Pinning

Defines model configurations for both the CSO reasoning engine and the
agentic mesh pipeline.  All model references in the codebase import from
here so that version bumps happen in one place.

Why Sonnet 4 for CSO:
  Sonnet provides the reasoning depth needed for multi-intent decomposition,
  policy-aware compromise detection, and structured JSON output — all in a
  single pass.  The CSO's value proposition depends on getting the
  decomposition right the first time.

Why Haiku 4.5 for Mesh:
  Haiku simulates realistic small specialist agents — the kind you'd deploy
  in a production agentic mesh.  Faster, cheaper, but individually less
  capable, which is exactly the point: the mesh's weakness is that each
  agent reasons in isolation with lossy context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for an Anthropic model deployment."""
    name: str
    temperature: float
    max_tokens: int
    purpose: str


CSO_MODEL = ModelSpec(
    name="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=2048,
    purpose="CSO reasoning core — single-pass multi-intent decomposition",
)

MESH_MODEL = ModelSpec(
    name="claude-haiku-4-5-20251001",
    temperature=0.2,
    max_tokens=1024,
    purpose="Mesh specialist agents — realistic small-agent simulation",
)

MODEL_CONFIG: dict[str, dict] = {
    "cso": {
        "model": CSO_MODEL.name,
        "temperature": CSO_MODEL.temperature,
        "max_tokens": CSO_MODEL.max_tokens,
        "purpose": CSO_MODEL.purpose,
    },
    "mesh": {
        "model": MESH_MODEL.name,
        "temperature": MESH_MODEL.temperature,
        "max_tokens": MESH_MODEL.max_tokens,
        "purpose": MESH_MODEL.purpose,
    },
}


def get_cso_model() -> ModelSpec:
    """Return the CSO model specification."""
    return CSO_MODEL


def get_mesh_model() -> ModelSpec:
    """Return the mesh model specification."""
    return MESH_MODEL
