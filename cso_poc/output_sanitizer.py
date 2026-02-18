"""
Output Sanitizer — Deterministic Redaction of Sensitive Identifiers

Prevents data exfiltration by redacting guest/reservation identifiers from
untrusted output fields when the CSO pipeline escalates or rejects a request.

The sanitizer operates on provenance: each field in the result dict is tagged
as originating from trusted (policy_config, tool_output) or untrusted
(user_input, rag_context) sources.  Untrusted fields are scanned for
identifier patterns and redacted when force-redaction conditions are met.

This is a post-pipeline filter — it does not change behavioral flow, only
the serialized output text.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Provenance levels
# ---------------------------------------------------------------------------

PROVENANCE_USER_INPUT = "user_input"
PROVENANCE_RAG_CONTEXT = "rag_context"
PROVENANCE_TOOL_OUTPUT = "tool_output"
PROVENANCE_POLICY_CONFIG = "policy_config"

# Map result_dict top-level keys to their provenance
_FIELD_PROVENANCE: dict[str, str] = {
    # Internal pipeline metadata — never redact
    "scenario": PROVENANCE_POLICY_CONFIG,
    "trace_id": PROVENANCE_POLICY_CONFIG,
    "status": PROVENANCE_POLICY_CONFIG,
    "timing": PROVENANCE_POLICY_CONFIG,
    "scorecard": PROVENANCE_POLICY_CONFIG,
    "mesh_annotation": PROVENANCE_POLICY_CONFIG,
    "guest_id": PROVENANCE_POLICY_CONFIG,
    # MCP tool execution results
    "actions": PROVENANCE_TOOL_OUTPUT,
    # Audit trail — may contain echoed adversarial content
    "breadcrumbs": PROVENANCE_TOOL_OUTPUT,
    # Derived from Claude's analysis of untrusted input
    "escalation_notes": PROVENANCE_USER_INPUT,
    "sub_intents": PROVENANCE_USER_INPUT,
    "domain_assertions": PROVENANCE_USER_INPUT,
    "contextual_assertions": PROVENANCE_USER_INPUT,
    # Accumulated state — may reference other guests
    "memory_snapshot": PROVENANCE_RAG_CONTEXT,
}


# ---------------------------------------------------------------------------
# Identifier patterns (compiled regex)
# ---------------------------------------------------------------------------

GUEST_ID_RE = re.compile(r"\bG-\d{4}\b")
RESERVATION_ID_RE = re.compile(r"\bR-\d{4}\b")
ADMIN_TOKEN_RE = re.compile(
    r"(?:admin[_-]?token|api[_-]?key|ANTHROPIC_API_KEY|secret[_-]?key)", re.I
)

_REDACTION_PLACEHOLDER = "[redacted-id]"


# ---------------------------------------------------------------------------
# Provenance context
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceContext:
    """Metadata about the current request used to decide redaction policy."""

    intent_type: str                  # EnvelopeStatus value
    requesting_guest_id: str          # The guest making the request
    injection_detected: bool = False
    detection_confidence: float = 1.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def should_force_redaction(ctx: ProvenanceContext) -> bool:
    """Determine whether output must be redacted."""
    if ctx.intent_type in ("Human_Escalation_Required", "Rejected"):
        return True
    if ctx.injection_detected:
        return True
    if ctx.detection_confidence < 0.5:
        return True
    return False


def redact_identifiers(text: str, permitted_ids: set[str]) -> str:
    """Replace sensitive identifier patterns in *text*, preserving permitted IDs."""

    def _replace(match: re.Match) -> str:
        value = match.group(0)
        if value in permitted_ids:
            return value
        return _REDACTION_PLACEHOLDER

    text = GUEST_ID_RE.sub(_replace, text)
    text = RESERVATION_ID_RE.sub(_replace, text)
    text = ADMIN_TOKEN_RE.sub(lambda _: _REDACTION_PLACEHOLDER, text)
    return text


def _sanitize_value(value: Any, permitted_ids: set[str]) -> Any:
    """Recursively redact identifiers in nested structures."""
    if isinstance(value, str):
        return redact_identifiers(value, permitted_ids)
    if isinstance(value, list):
        return [_sanitize_value(item, permitted_ids) for item in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v, permitted_ids) for k, v in value.items()}
    return value


def sanitize_cso_output(
    result: dict[str, Any],
    ctx: ProvenanceContext,
) -> dict[str, Any]:
    """
    Main entry point — redact sensitive identifiers from CSO output.

    Returns *result* unchanged when redaction is not required.  Otherwise
    returns a shallow copy with untrusted fields sanitized.
    """
    if not should_force_redaction(ctx):
        return result

    # Build the set of identifiers that are permitted in the output
    permitted_ids: set[str] = {ctx.requesting_guest_id}

    # Allow reservation IDs from the requesting guest's successful tool results
    for action in result.get("actions", []):
        action_result = action.get("result")
        if isinstance(action_result, dict):
            res_id = action_result.get("reservation_id") or action_result.get("res_id")
            if res_id:
                permitted_ids.add(res_id)

    out = dict(result)  # shallow copy

    for key, value in result.items():
        provenance = _FIELD_PROVENANCE.get(key, PROVENANCE_USER_INPUT)

        if provenance == PROVENANCE_POLICY_CONFIG:
            # Internal metadata — never redact
            continue

        if provenance == PROVENANCE_TOOL_OUTPUT and key == "actions":
            # Successful tool results contain legitimate data; skip redaction
            # when there are actual results. Redact if empty/error.
            if value:
                continue

        if provenance in (
            PROVENANCE_USER_INPUT,
            PROVENANCE_RAG_CONTEXT,
            PROVENANCE_TOOL_OUTPUT,  # breadcrumbs
        ):
            out[key] = _sanitize_value(copy.deepcopy(value), permitted_ids)

    return out
