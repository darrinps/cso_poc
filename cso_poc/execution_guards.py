"""
Execution Guards — Deterministic Invariant Enforcement

Applied after Claude reasoning but before MCP tool execution.
Prevents behavioral deviations without changing reasoning architecture.

Guard pipeline (order matters):
  1. Deduplicate tool calls by (tool_name, param_hash)   — always active
  2. Deterministic ordering by safety priority            — always active
  3. Write-tool blocking under injection                  — injection-aware
  4. Tool count cap under injection                       — injection-aware
  5. Injection status escalation override                 — injection-aware
  6. Status normalization (Rejected → Escalation)         — always active

These are execution-level constraints, not LLM reasoning modifications.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from cso_poc.injection_detection import InjectionDetectionResult

log = logging.getLogger("cso.execution_guards")


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------

WRITE_TOOLS = frozenset({
    "pms_update_reservation",
    "pms_update_checkin",
    "pms_reassign_room",
    "loyalty_allocate_benefit",
    "_admin_reset_db",
})

# Safety priority for deterministic ordering (lower = higher priority)
# Read-before-write, then lexical fallback
TOOL_SAFETY_PRIORITY: dict[str, int] = {
    "pms_query_rooms": 0,           # read-first
    "pms_update_reservation": 10,
    "pms_update_checkin": 10,
    "loyalty_allocate_benefit": 20,
    "pms_reassign_room": 30,
}

# Max tool calls permitted under active injection
_INJECTION_TOOL_CAP = 1


# ---------------------------------------------------------------------------
# Guard context + result
# ---------------------------------------------------------------------------

@dataclass
class GuardContext:
    """Execution context for guard decisions."""

    injection_detected: bool = False
    injection_confidence: float = 0.0
    intent_type: str = ""  # EnvelopeStatus value


@dataclass
class GuardResult:
    """Output from execution guards."""

    actions: list
    status_override: str | None = None
    breadcrumb_notes: list[str] = field(default_factory=list)
    escalation_note: str | None = None


# ---------------------------------------------------------------------------
# Individual guards
# ---------------------------------------------------------------------------

def _param_hash(action) -> str:
    """Deterministic hash of tool_name + parameters for deduplication."""
    key = json.dumps(
        {"tool": action.tool_name, "params": action.parameters},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def deduplicate_actions(actions: list, notes: list[str]) -> list:
    """Remove duplicate tool calls (same tool_name + normalized params)."""
    seen: set[str] = set()
    out: list = []
    for a in actions:
        h = _param_hash(a)
        if h in seen:
            notes.append(f"duplicate_tool_suppressed: {a.tool_name}")
            log.info("Guard: suppressed duplicate %s (hash=%s)", a.tool_name, h)
            continue
        seen.add(h)
        out.append(a)
    return out


def sort_actions(actions: list) -> list:
    """Deterministic ordering: safety priority → lexical tool name."""
    result = sorted(
        actions,
        key=lambda a: (
            TOOL_SAFETY_PRIORITY.get(a.tool_name, 50),
            a.tool_name,
        ),
    )
    for i, a in enumerate(result):
        a.order = i
    return result


def enforce_injection_guards(
    actions: list,
    ctx: GuardContext,
    notes: list[str],
) -> tuple[list, str | None]:
    """
    Block write tools and cap tool count under active injection.

    Returns (filtered_actions, status_override_or_None).
    """
    if not ctx.injection_detected:
        return actions, None

    # Block all write tools under injection
    safe: list = []
    for a in actions:
        if a.tool_name in WRITE_TOOLS:
            notes.append(f"write_tool_blocked: {a.tool_name}")
            log.info("Guard: blocked write tool %s under injection", a.tool_name)
        else:
            safe.append(a)

    # Cap at _INJECTION_TOOL_CAP
    if len(safe) > _INJECTION_TOOL_CAP:
        for a in safe[_INJECTION_TOOL_CAP:]:
            notes.append(f"tool_cap_truncated: {a.tool_name}")
        safe = safe[:_INJECTION_TOOL_CAP]

    # If we removed anything, force escalation
    status_override = None
    if len(safe) < len(actions):
        status_override = "Human_Escalation_Required"
        notes.append("status_override: injection guard enforcement")

    return safe, status_override


def enforce_injection_status(
    ctx: GuardContext,
    current: str,
    notes: list[str],
) -> str | None:
    """
    Prevent Executable or Partial_Fulfillment under active injection.

    Forces escalation so adversarial inputs never produce a "success" status.
    """
    if not ctx.injection_detected:
        return None

    if current in ("Partial_Fulfillment", "Executable"):
        notes.append(
            f"injection_escalation: {current} -> Human_Escalation_Required"
        )
        return "Human_Escalation_Required"

    return None


def normalize_status(
    current: str,
    action_count: int,
    notes: list[str],
) -> str | None:
    """
    Normalize inconsistent status.

    Rejected with 0 actions is operationally identical to escalation
    (staff must handle it), so normalize to Human_Escalation_Required.
    """
    if current == "Rejected" and action_count == 0:
        notes.append(
            "status_normalized: Rejected -> Human_Escalation_Required"
        )
        return "Human_Escalation_Required"
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_execution_guards(
    actions: list,
    current_status: str,
    detection: InjectionDetectionResult,
) -> GuardResult:
    """
    Apply all deterministic execution guards.

    Called after reason_and_fallback(), before envelope construction.
    Returns a GuardResult with filtered actions and optional status override.
    """
    ctx = GuardContext(
        injection_detected=detection.detected,
        injection_confidence=detection.confidence,
        intent_type=current_status,
    )
    notes: list[str] = []

    # 1. Deduplicate (always active)
    actions = deduplicate_actions(list(actions), notes)

    # 2. Deterministic ordering (always active)
    actions = sort_actions(actions)

    # 3-4. Injection: write-tool blocking + tool cap
    actions, cap_override = enforce_injection_guards(actions, ctx, notes)

    # 5. Injection: status escalation
    effective = current_status
    inj_override = enforce_injection_status(ctx, effective, notes)
    if inj_override:
        effective = inj_override
    if cap_override and effective != cap_override:
        effective = cap_override

    # 6. Status normalization (always active)
    norm = normalize_status(effective, len(actions), notes)
    if norm:
        effective = norm

    status_override = effective if effective != current_status else None

    # Build escalation note if status was forced
    escalation_note = None
    if status_override:
        escalation_note = (
            "Deterministic execution guard: intent escalated "
            "due to invariant enforcement"
        )

    if notes:
        log.info("Execution guards applied: %s", notes)

    return GuardResult(
        actions=actions,
        status_override=status_override,
        breadcrumb_notes=notes,
        escalation_note=escalation_note,
    )
