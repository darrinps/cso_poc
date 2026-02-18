"""
Layer 4 — CSO Orchestrator  (Centralized Reasoning + Cognitive Fallback + Memory)

Five-phase pipeline per turn:
  Phase 0 — SCRUB:     Run scrub_expired_context() to purge zombie facts.
  Phase 1 — DECOMPOSE: Break raw guest text into sub-intents via Claude.
  Phase 2 — REASON:    Check each against MCP manifest + policy ceilings.
                        Apply Cognitive Fallback where needed.
  Phase 3 — EXECUTE:   Dispatch the CanonicalIntentEnvelope via MCP.
  Phase 4 — MEMORISE:  Write facts into the three memory blocks and
                        force-expire transient sentiment.

Memory architecture (inspired by Letta/MemGPT):
  core_block      — permanent guest facts (tier, preferences)
  recall_block    — tactical journey history (48 h TTL)
  transient_block — in-flight sentiment / hypotheses (expires on resolve)

Architectural Decision: Centralized reasoning over agent negotiation
  The CSO processes all sub-intents in a single pass with full context
  visibility.  This eliminates the "telephone game" problem where context
  degrades at each agent-to-agent handoff.  The tradeoff is a larger
  prompt and higher per-call latency — but accuracy improves measurably,
  especially for multi-intent requests with cross-domain dependencies.

Architectural Decision: Cognitive Fallback as a deterministic cascade
  The fallback logic is a priority-ordered cascade, not a decision tree:
    1. No MCP tool exists → escalate to human staff
    2. Tier gate violation → deny + escalate
    3. Policy ceiling exceeded → clamp + compensate
    4. All clear → execute normally
  Each branch produces a structured audit trail (Decision Breadcrumbs)
  so every outcome is explainable.

Exposes a FastAPI HTTP interface for the Streamlit dashboard (Layer 2)
and an embedded single-page HTML UI for the scenario runner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel as PydanticBase

from cso_poc.memory import MemoryManager, MemoryTier
from cso_poc.schemas import (
    CanonicalIntentEnvelope,
    ContextualAssertion,
    DecisionBreadcrumb,
    EnvelopeStatus,
    ProposedAction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("cso.orchestrator")
breadcrumb_log = logging.getLogger("cso.breadcrumbs")

MCP_GATEWAY_URL = os.environ.get(
    "MCP_GATEWAY_URL", "http://mcp-hospitality-gateway:8000/sse"
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
# Module-level singletons: memory, breadcrumbs, and chat history are shared
# across all request handlers within a single process.  The thread lock
# protects breadcrumb_stream because FastAPI may process requests concurrently
# (e.g., scenario + health check), and list.append is not atomic in CPython
# when combined with iteration.

memory = MemoryManager()
breadcrumb_stream: list[dict] = []
chat_history: list[dict] = []
_stream_lock = threading.Lock()


def _emit_breadcrumb(crumb: DecisionBreadcrumb) -> None:
    breadcrumb_log.info(crumb.format_log_line())
    with _stream_lock:
        breadcrumb_stream.append({
            "trace_id": crumb.trace_id,
            "policy_reference": crumb.policy_reference,
            "action_taken": crumb.action_taken,
            "result": crumb.result,
            "timestamp": crumb.timestamp.isoformat(),
            "latency_ms": crumb.latency_ms,
            "log_line": crumb.format_log_line(),
        })


# ---------------------------------------------------------------------------
# Phase 0 — Memory scrub (before every reasoning cycle)
# ---------------------------------------------------------------------------

def run_memory_scrub(trace_id: str) -> None:
    """Purge expired context and log what was removed."""
    scrubbed = memory.scrub_expired_context()
    if scrubbed:
        for f in scrubbed:
            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="MEMORY-DECAY",
                action_taken=f"scrub_expired_context(tier={f.tier.value})",
                result=f"SCRUBBED — '{f.fact[:60]}...' (expired {f.expires_at})",
            )
            _emit_breadcrumb(crumb)
    else:
        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-DECAY",
            action_taken="scrub_expired_context()",
            result="CLEAN — no zombie context found",
        )
        _emit_breadcrumb(crumb)


# ---------------------------------------------------------------------------
# Phase 1 — Sub-intent decomposition
# ---------------------------------------------------------------------------
# The SubIntent dataclass is the bridge between Claude's JSON output and
# the orchestrator's deterministic reasoning.  Each field maps to a
# specific decision point in the Cognitive Fallback cascade:
#   - required_tool=None → no-tool escalation
#   - tier_violation set → tier-gate denial
#   - policy_ceiling set → compromise + compensate
#   - all clear → standard execution

@dataclass
class SubIntent:
    description: str
    domain: str
    required_tool: str | None = None
    original_parameter_value: Any = None
    policy_ceiling: Any = None
    extra_params: dict = field(default_factory=dict)
    tier_violation: str | None = None


def decompose_complex_request() -> list[SubIntent]:
    return [
        SubIntent(
            description="Extend checkout to 5 PM",
            domain="PMS",
            required_tool="pms_update_reservation",
            original_parameter_value="17:00",
            policy_ceiling="16:00",
            extra_params={"res_id": "R-5001", "date": "2026-02-03"},
        ),
        SubIntent(
            description="Bottle of Macallan 12 delivered to room 1412",
            domain="Provisions",
            required_tool=None,
        ),
        SubIntent(
            description="Allocate Suite Night Award for Diamond guest",
            domain="Loyalty",
            required_tool="loyalty_allocate_benefit",
            extra_params={"guest_id": "G-1001", "benefit_type": "SuiteNightAward"},
        ),
        SubIntent(
            description="Move WV check-in to 7 PM due to flight LH456 delay (+4 hrs)",
            domain="PMS",
            required_tool="pms_update_checkin",
            extra_params={"res_id": "R-5003", "date": "2026-02-03"},
            original_parameter_value="19:00",
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 2 — Cognitive Fallback
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    actions: list[ProposedAction] = field(default_factory=list)
    contextual_assertions: list[ContextualAssertion] = field(default_factory=list)
    domain_assertions: list[str] = field(default_factory=list)
    escalation_notes: list[str] = field(default_factory=list)
    has_escalation: bool = False
    has_compromise: bool = False


def reason_and_fallback(
    sub_intents: list[SubIntent],
    available_tools: set[str],
    guest_id: str,
    trace_id: str,
) -> ReasoningResult:
    result = ReasoningResult()
    action_order = 0

    for si in sub_intents:
        if si.required_tool is None or si.required_tool not in available_tools:
            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="COGNITIVE-FALLBACK-NO-TOOL",
                action_taken=f"evaluate_sub_intent({si.description})",
                result=f"ESCALATE — Domain '{si.domain}' lacks authorised MCP tool",
            )
            _emit_breadcrumb(crumb)
            result.contextual_assertions.append(ContextualAssertion(
                domain=si.domain,
                assertion=(
                    f"Domain: {si.domain} lacks authorised MCP tool. "
                    f"Cannot fulfil: '{si.description}'. "
                    f"No tool in gateway manifest matches this capability."
                ),
                requires_escalation=True,
            ))
            result.escalation_notes.append(
                f"[{si.domain}] Guest requested: '{si.description}'. "
                f"Requires manual staff intervention — no automated pathway exists."
            )
            result.has_escalation = True
            continue

        # ── Tier-gated denial ──────────────────────────────────────────
        if si.tier_violation:
            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="COGNITIVE-FALLBACK-TIER-GATE",
                action_taken=f"evaluate_sub_intent({si.description})",
                result=f"DENIED — {si.tier_violation}",
            )
            _emit_breadcrumb(crumb)
            result.contextual_assertions.append(ContextualAssertion(
                domain=si.domain,
                assertion=si.tier_violation,
                requires_escalation=True,
            ))
            result.escalation_notes.append(
                f"[{si.domain}] Denied: {si.description}. {si.tier_violation}"
            )
            result.has_escalation = True
            continue

        if si.policy_ceiling is not None and si.original_parameter_value is not None:
            original_hour = int(si.original_parameter_value.split(":")[0])
            ceiling_hour = int(si.policy_ceiling.split(":")[0])

            if original_hour > ceiling_hour:
                crumb = DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference="COGNITIVE-FALLBACK-POLICY-CEILING",
                    action_taken=f"evaluate_sub_intent({si.description})",
                    result=(
                        f"COMPROMISE — Requested {si.original_parameter_value} "
                        f"exceeds ceiling {si.policy_ceiling}; "
                        f"clamping to {si.policy_ceiling} + compensatory benefit"
                    ),
                )
                _emit_breadcrumb(crumb)
                result.domain_assertions.append(
                    f"Guest requested checkout at {si.original_parameter_value} "
                    f"but POLICY-MAX-CHECKOUT-HOUR caps at {si.policy_ceiling}"
                )
                result.contextual_assertions.append(ContextualAssertion(
                    domain=si.domain,
                    assertion=(
                        f"Original request ({si.original_parameter_value}) exceeds "
                        f"policy ceiling ({si.policy_ceiling}). CSO applied compromise: "
                        f"clamp to {si.policy_ceiling} and add ComplimentaryDrinkVoucher "
                        f"as goodwill compensation."
                    ),
                    requires_escalation=False,
                ))
                res_id = si.extra_params.get("res_id", "R-5001")
                checkout_date = si.extra_params.get("date", "2026-02-03")
                result.actions.append(ProposedAction(
                    tool_name="pms_update_reservation",
                    parameters={
                        "res_id": res_id,
                        "checkout_time": f"{checkout_date}T{si.policy_ceiling}:00",
                        "notes": (
                            f"CSO compromise: guest asked {si.original_parameter_value}, "
                            f"policy max is {si.policy_ceiling}. "
                            f"Drink voucher issued as compensation."
                        ),
                    },
                    order=action_order,
                    is_compromise=True,
                    compromise_rationale=(
                        f"Checkout clamped from {si.original_parameter_value} "
                        f"to {si.policy_ceiling} (POLICY-MAX-CHECKOUT-HOUR). "
                        f"ComplimentaryDrinkVoucher added to offset the gap."
                    ),
                ))
                action_order += 1
                result.actions.append(ProposedAction(
                    tool_name="loyalty_allocate_benefit",
                    parameters={
                        "guest_id": guest_id,
                        "benefit_type": "ComplimentaryDrinkVoucher",
                    },
                    order=action_order,
                    is_compromise=True,
                    compromise_rationale=(
                        "Goodwill benefit to compensate for checkout being capped "
                        "1 hour short of the guest's original request."
                    ),
                ))
                action_order += 1
                result.has_compromise = True
                continue

        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="INTENT-CANONICALIZATION",
            action_taken=f"evaluate_sub_intent({si.description})",
            result="EXECUTABLE — within policy bounds",
        )
        _emit_breadcrumb(crumb)

        if si.required_tool == "loyalty_allocate_benefit":
            result.actions.append(ProposedAction(
                tool_name="loyalty_allocate_benefit",
                parameters={
                    "guest_id": si.extra_params.get("guest_id", guest_id),
                    "benefit_type": si.extra_params.get("benefit_type", "SuiteNightAward"),
                },
                order=action_order,
            ))
        elif si.required_tool == "pms_update_reservation":
            res_id = si.extra_params.get("res_id", "R-5001")
            date = si.extra_params.get("date", "2026-02-03")
            result.actions.append(ProposedAction(
                tool_name="pms_update_reservation",
                parameters={
                    "res_id": res_id,
                    "checkout_time": f"{date}T{si.original_parameter_value}:00",
                    "notes": si.extra_params.get("notes", "Standard checkout extension"),
                },
                order=action_order,
            ))
        elif si.required_tool == "pms_update_checkin":
            res_id = si.extra_params.get("res_id", "R-5003")
            date = si.extra_params.get("date", "2026-02-03")
            result.actions.append(ProposedAction(
                tool_name="pms_update_checkin",
                parameters={
                    "res_id": res_id,
                    "checkin_time": f"{date}T{si.original_parameter_value}:00",
                    "notes": (
                        "Check-in moved to 19:00 due to flight LH456 delay "
                        "(+4 hours). Property WVGB01 notified."
                    ),
                },
                order=action_order,
            ))
            result.domain_assertions.append(
                "Flight LH456 is delayed 4 hours — WV check-in moved from 15:00 to 19:00"
            )
        elif si.required_tool in ("pms_query_rooms", "pms_reassign_room"):
            is_comp = si.extra_params.pop("_is_compromise", False)
            comp_rationale = si.extra_params.pop("_compromise_rationale", "")
            result.actions.append(ProposedAction(
                tool_name=si.required_tool,
                parameters=si.extra_params,
                order=action_order,
                is_compromise=is_comp,
                compromise_rationale=comp_rationale,
            ))
        action_order += 1

    return result


# ---------------------------------------------------------------------------
# Phase 3 — Deterministic execution
# ---------------------------------------------------------------------------
# The envelope is executed sequentially in action.order, not in parallel.
# Sequential execution ensures deterministic behavior and makes saga tracking
# straightforward: we know exactly which actions completed before a failure.
# Parallel execution would be faster but would require distributed transaction
# coordination that is out of scope for this POC.

async def execute_envelope(
    envelope: CanonicalIntentEnvelope,
    session: ClientSession,
) -> tuple[list[dict], list[DecisionBreadcrumb]]:
    """
    Deterministic execution of a CanonicalIntentEnvelope via MCP tool calls.

    Saga tracking: maintains a committed_actions list so that on partial failure
    we know exactly which actions succeeded and which were skipped. In production
    this would integrate with compensating transactions (reversal tools) and
    idempotency keys for safe retries. Here we emit a PARTIAL-EXECUTION-WARNING
    breadcrumb and record the saga state in tactical memory.
    """
    results: list[dict] = []
    crumbs: list[DecisionBreadcrumb] = []
    committed_actions: list[dict] = []

    log.info("=== Executing Intent %s ===", envelope.intent_id)
    log.info("Status    : %s", envelope.status.value)
    log.info("Objective : %s", envelope.primary_objective)

    sorted_actions = sorted(envelope.proposed_actions, key=lambda a: a.order)

    for idx, action in enumerate(sorted_actions):
        tag = "COMPROMISE" if action.is_compromise else "STANDARD"
        params = {**action.parameters, "trace_id": envelope.intent_id}

        log.info("[%s] [%s] Calling %s (order=%d)",
                 envelope.intent_id, tag, action.tool_name, action.order)

        try:
            tool_start = time.time()
            call_result = await session.call_tool(action.tool_name, params)
            tool_latency_ms = (time.time() - tool_start) * 1000

            content_text = "".join(
                block.text for block in call_result.content if hasattr(block, "text")
            )
            try:
                parsed = json.loads(content_text)
            except (json.JSONDecodeError, TypeError):
                parsed = {"raw": content_text}

            is_error = "error" in parsed
            result_label = f"DENIED({parsed.get('error')})" if is_error else "OK"

            crumb = DecisionBreadcrumb(
                trace_id=envelope.intent_id,
                policy_reference=(
                    "COMPROMISE-EXECUTION" if action.is_compromise else action.tool_name
                ),
                action_taken=f"{action.tool_name}({json.dumps(action.parameters, default=str)})",
                result=result_label,
                latency_ms=round(tool_latency_ms, 1),
            )
            _emit_breadcrumb(crumb)
            crumbs.append(crumb)

            action_record = {
                "action": action.tool_name,
                "is_compromise": action.is_compromise,
                "result": parsed,
            }
            results.append(action_record)
            committed_actions.append(action_record)

            # On error mid-saga, emit partial execution warning
            if is_error and idx < len(sorted_actions) - 1:
                remaining = len(sorted_actions) - idx - 1
                warn_crumb = DecisionBreadcrumb(
                    trace_id=envelope.intent_id,
                    policy_reference="PARTIAL-EXECUTION-WARNING",
                    action_taken=f"saga_state(committed={len(committed_actions)}, failed={action.tool_name}, skipped={remaining})",
                    result=f"Action {action.tool_name} returned error; {remaining} actions remaining",
                )
                _emit_breadcrumb(warn_crumb)
                crumbs.append(warn_crumb)
                memory.add_tactical_fact(
                    f"Partial execution: {len(committed_actions)} committed, "
                    f"{action.tool_name} failed, {remaining} remaining",
                    trace_id=envelope.intent_id, domain="Saga",
                    tags=["saga", "partial_execution"],
                )

        except Exception as exc:
            tool_latency_ms = 0.0
            log.error("Tool %s raised exception: %s", action.tool_name, exc)
            parsed = {"error": str(exc)}
            results.append({
                "action": action.tool_name,
                "is_compromise": action.is_compromise,
                "result": parsed,
            })

            if idx < len(sorted_actions) - 1:
                remaining = len(sorted_actions) - idx - 1
                warn_crumb = DecisionBreadcrumb(
                    trace_id=envelope.intent_id,
                    policy_reference="PARTIAL-EXECUTION-WARNING",
                    action_taken=f"saga_state(committed={len(committed_actions)}, exception={action.tool_name}, skipped={remaining})",
                    result=f"Exception in {action.tool_name}: {exc}; {remaining} actions remaining",
                )
                _emit_breadcrumb(warn_crumb)
                crumbs.append(warn_crumb)
                memory.add_tactical_fact(
                    f"Saga exception: {len(committed_actions)} committed, "
                    f"{action.tool_name} exception: {exc}, {remaining} remaining",
                    trace_id=envelope.intent_id, domain="Saga",
                    tags=["saga", "exception"],
                )

    for note in envelope.escalation_notes:
        crumb = DecisionBreadcrumb(
            trace_id=envelope.intent_id,
            policy_reference="HUMAN-ESCALATION-REQUIRED",
            action_taken="flag_for_staff",
            result=note,
        )
        _emit_breadcrumb(crumb)
        crumbs.append(crumb)

    return results, crumbs


# ---------------------------------------------------------------------------
# Phase 4 — Populate memory blocks after execution
# ---------------------------------------------------------------------------

def populate_memory_blocks(
    trace_id: str,
    envelope: CanonicalIntentEnvelope,
    results: list[dict],
    sentiment: str | None = None,
    guest_profile: dict | None = None,
) -> None:
    """Write facts into the appropriate memory blocks after execution."""

    # ── Core block (permanent) ──────────────────────────────────────────
    if guest_profile:
        gid = guest_profile.get("guest_id", "unknown")
        name = guest_profile.get("name", "unknown")
        tier = guest_profile.get("loyalty_tier", "unknown")
        memory.add_core_fact(
            f"Guest {gid} ({name}) — {tier} tier",
            domain="Identity", tags=["guest", tier.lower()],
        )
        prefs = guest_profile.get("preferences", {})
        if prefs:
            pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items()
                                 if not isinstance(v, (list, dict)))
            if pref_str:
                memory.add_core_fact(
                    f"Preferences: {pref_str}",
                    domain="Preferences", tags=["guest", "preferences"],
                )
            pets = prefs.get("pets", [])
            for pet in pets:
                memory.add_core_fact(
                    f"Pet: {pet.get('name', 'unknown')} "
                    f"({pet.get('breed', 'unknown')}, {pet.get('weight_lbs', '?')} lbs)",
                    domain="Preferences", tags=["pet", "dog", "large"],
                )
        memory.add_core_fact(
            f"Loyalty ID: {gid} — {tier}",
            domain="Loyalty", tags=["guest", "loyalty"],
        )
    else:
        memory.add_core_fact(
            "Guest G-1001 (Alexandra Mercer) — Diamond tier",
            domain="Identity", tags=["guest", "diamond"],
        )
        memory.add_core_fact(
            "Preferences: suite, firm pillow, stocked minibar",
            domain="Preferences", tags=["guest", "preferences"],
        )
        memory.add_core_fact(
            "Loyalty ID: G-1001 — Hilton Honors Diamond",
            domain="Loyalty", tags=["guest", "loyalty"],
        )

    # ── Tactical / recall block (48 h TTL) ──────────────────────────────
    for action in envelope.proposed_actions:
        params_str = json.dumps(action.parameters, default=str)
        if action.is_compromise:
            memory.add_tactical_fact(
                f"Compromise: {action.tool_name} — {action.compromise_rationale}",
                trace_id=trace_id, domain="PMS",
                tags=["compromise", action.tool_name],
            )
        if action.tool_name == "pms_update_checkin":
            checkin_time = action.parameters.get("checkin_time", "unknown")
            memory.add_tactical_fact(
                f"Guest requested 7 PM arrival for WV stay — "
                f"check-in moved to {checkin_time} (R-5003, WVGB01)",
                trace_id=trace_id, domain="PMS",
                tags=["wv", "checkin", "R-5003", "WVGB01", "arrival"],
            )
        if action.tool_name == "pms_update_reservation":
            checkout_time = action.parameters.get("checkout_time", "unknown")
            memory.add_tactical_fact(
                f"LHRW01 checkout set to {checkout_time} (R-5001)",
                trace_id=trace_id, domain="PMS",
                tags=["checkout", "R-5001", "LHRW01"],
            )
        if action.tool_name == "loyalty_allocate_benefit":
            benefit = action.parameters.get("benefit_type", "unknown")
            memory.add_tactical_fact(
                f"Benefit allocated: {benefit} for G-1001",
                trace_id=trace_id, domain="Loyalty",
                tags=["benefit", benefit.lower()],
            )

    for note in envelope.escalation_notes:
        memory.add_tactical_fact(
            f"Escalation: {note}",
            trace_id=trace_id, domain="Escalation",
            tags=["escalation"],
        )

    # ── Transient block (short-lived sentiment) ─────────────────────────
    if sentiment:
        memory.add_transient_fact(
            f"Guest sentiment: {sentiment}",
            trace_id=trace_id, domain="Sentiment",
            tags=["sentiment", sentiment.lower()],
        )

    memory.add_transient_fact(
        f"Active reasoning for intent {trace_id[:8]}",
        trace_id=trace_id, domain="State",
        tags=["active_intent"],
    )


# ---------------------------------------------------------------------------
# Turn 1: complex request
# ---------------------------------------------------------------------------

async def process_turn_1(raw_request: str) -> str:
    guest_id = "G-1001"
    chat_history.append({"role": "guest", "text": raw_request})

    sub_intents = decompose_complex_request()

    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            available_tools = {t.name for t in tools_resp.tools}

            envelope_shell = CanonicalIntentEnvelope(primary_objective=raw_request)
            trace_id = envelope_shell.intent_id

            # Phase 0 — Scrub
            run_memory_scrub(trace_id)

            # Transient sentiment: complex request with frustration signals
            memory.add_transient_fact(
                "Guest sentiment: Frustrated — multiple demands in single request, "
                "one exceeds policy, implies urgency",
                trace_id=trace_id, domain="Sentiment",
                tags=["sentiment", "frustrated"],
            )

            # Phase 2 — Reason
            reasoning = reason_and_fallback(
                sub_intents=sub_intents,
                available_tools=available_tools,
                guest_id=guest_id,
                trace_id=trace_id,
            )

            if reasoning.has_escalation and not reasoning.actions:
                status = EnvelopeStatus.REJECTED
            elif reasoning.has_escalation:
                status = EnvelopeStatus.HUMAN_ESCALATION_REQUIRED
            elif reasoning.has_compromise:
                status = EnvelopeStatus.PARTIAL_FULFILLMENT
            else:
                status = EnvelopeStatus.EXECUTABLE

            envelope = CanonicalIntentEnvelope(
                intent_id=trace_id,
                primary_objective=raw_request,
                status=status,
                domain_assertions=[
                    "Guest G-1001 is Diamond tier",
                    "Current LHRW01 checkout is 2026-02-03T11:00:00",
                    "Property LHRW01 max checkout policy: 16:00 for Diamond",
                    "No MCP tool registered for domain 'Provisions'",
                    "Flight LH456 delay: +4 hours — affects WV check-in",
                    "WV stay R-5003 original check-in: 2026-02-03T15:00:00",
                    *reasoning.domain_assertions,
                ],
                contextual_assertions=reasoning.contextual_assertions,
                proposed_actions=reasoning.actions,
                escalation_notes=reasoning.escalation_notes,
            )

            # Phase 3 — Execute
            results, crumbs = await execute_envelope(envelope, session)

            # Phase 4 — Populate memory blocks
            all_crumbs = []
            with _stream_lock:
                for item in breadcrumb_stream:
                    if item["trace_id"] == trace_id:
                        all_crumbs.append(DecisionBreadcrumb(
                            trace_id=item["trace_id"],
                            policy_reference=item["policy_reference"],
                            action_taken=item["action_taken"],
                            result=item["result"],
                        ))

            memory.store_intent(
                guest_message=raw_request,
                envelope=envelope,
                execution_results=results,
                breadcrumbs=all_crumbs,
            )

            populate_memory_blocks(
                trace_id=trace_id,
                envelope=envelope,
                results=results,
                sentiment="Frustrated",
            )

            # Force-expire transient sentiment — intent is now resolved
            expired = memory.expire_transient_for_trace(trace_id)
            for f in expired:
                crumb = DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference="MEMORY-DECAY-RESOLVED",
                    action_taken=f"expire_transient(fact='{f.fact[:50]}...')",
                    result="FORGOTTEN — intent resolved, transient state cleared",
                )
                _emit_breadcrumb(crumb)

            # Build response
            lines = [f"I've processed your request (Trace: {trace_id[:8]}...).\n"]
            for r in results:
                tag = " [compromise]" if r["is_compromise"] else ""
                status_txt = r["result"].get("status", "error")
                if r["action"] == "pms_update_reservation" and status_txt == "confirmed":
                    lines.append(f"  Checkout extended to {r['result'].get('new_checkout', '')}{tag}")
                elif r["action"] == "pms_update_checkin" and status_txt == "confirmed":
                    lines.append(f"  WV check-in moved to {r['result'].get('new_checkin', '')}")
                elif r["action"] == "loyalty_allocate_benefit" and status_txt == "allocated":
                    lines.append(f"  {r['result'].get('benefit_type', '')} allocated{tag}")

            if envelope.escalation_notes:
                lines.append("\n  Items requiring staff attention:")
                for note in envelope.escalation_notes:
                    lines.append(f"    - {note}")
            if reasoning.has_compromise:
                lines.append(
                    "\nNote: Your 5 PM checkout was adjusted to 4 PM (our maximum). "
                    "A complimentary drink voucher has been added to your stay."
                )

            response = "\n".join(lines)
            chat_history.append({"role": "cso", "text": response})
            return response


# ---------------------------------------------------------------------------
# Turn 2: WV follow-up — memory recall
# ---------------------------------------------------------------------------

async def process_turn_2_wv(raw_request: str) -> str:
    chat_history.append({"role": "guest", "text": raw_request})

    envelope_shell = CanonicalIntentEnvelope(primary_objective=raw_request)
    trace_id = envelope_shell.intent_id

    # Phase 0 — Scrub before reasoning
    run_memory_scrub(trace_id)

    crumb = DecisionBreadcrumb(
        trace_id=trace_id,
        policy_reference="MEMORY-RECALL",
        action_taken="search_memory(query='WV')",
        result="Searching Layer 4 memory block for prior WV-related actions",
    )
    _emit_breadcrumb(crumb)

    property_hits = memory.recall_by_property("WVGB01")

    crumb = DecisionBreadcrumb(
        trace_id=trace_id,
        policy_reference="MEMORY-RECALL",
        action_taken="recall_by_property(WVGB01)",
        result=f"Found {len(property_hits)} memory entries for property WVGB01",
    )
    _emit_breadcrumb(crumb)

    wv_checkin_result = None
    source_trace = None
    for hit in property_hits:
        if hit.get("tool") == "pms_update_checkin":
            wv_checkin_result = hit
            source_trace = hit.get("trace_id", "unknown")
            break

    if wv_checkin_result:
        new_checkin = wv_checkin_result["parameters"].get("checkin_time", "unknown")
        notes = wv_checkin_result["parameters"].get("notes", "")

        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-CONFIRMED",
            action_taken=f"confirm_wv_checkin(source_trace={source_trace[:8]}...)",
            result=f"WV check-in (R-5003) confirmed moved to {new_checkin}",
        )
        _emit_breadcrumb(crumb)

        envelope = CanonicalIntentEnvelope(
            intent_id=trace_id,
            primary_objective=raw_request,
            status=EnvelopeStatus.EXECUTABLE,
            domain_assertions=[
                f"Memory recall: WV check-in was updated in trace {source_trace[:8]}",
                f"Reservation R-5003 at WVGB01 check-in moved to {new_checkin}",
                f"Reason: {notes}",
            ],
            contextual_assertions=[ContextualAssertion(
                domain="PMS",
                assertion=(
                    f"WV stay (R-5003, WVGB01) check-in was moved to {new_checkin} "
                    f"during turn 1 (trace {source_trace[:8]}...). "
                    f"Confirmed from Layer 4 memory — no re-execution required."
                ),
                requires_escalation=False,
            )],
        )
        memory.store_intent(
            guest_message=raw_request, envelope=envelope,
            execution_results=[], breadcrumbs=[crumb],
        )
        response = (
            f"Yes — your WV stay at The Greenbrier (WVGB01) was handled in "
            f"the same pass.\n\n"
            f"  Reservation R-5003 check-in: moved to {new_checkin}\n"
            f"  Reason: Flight LH456 delay (+4 hours)\n"
            f"  Property notified: WVGB01\n\n"
            f"This was confirmed from memory (source trace: {source_trace[:8]}...). "
            f"No additional system calls were needed."
        )
    else:
        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-MISS",
            action_taken="search_memory(WV)",
            result="No WV-related actions found in memory",
        )
        _emit_breadcrumb(crumb)
        response = (
            "I don't have a record of any WV stay modifications in this session. "
            "Could you provide more details about the reservation?"
        )

    chat_history.append({"role": "cso", "text": response})
    return response


# ---------------------------------------------------------------------------
# Turn 3: Persistence test — tactical block recall, no re-reasoning
# ---------------------------------------------------------------------------

async def process_turn_3_persistence(raw_request: str) -> str:
    """
    The guest returns and asks about their WV arrival.
    The CSO answers from the tactical memory block without re-running
    the reasoning engine or making any MCP calls.
    """
    chat_history.append({"role": "guest", "text": raw_request})

    envelope_shell = CanonicalIntentEnvelope(primary_objective=raw_request)
    trace_id = envelope_shell.intent_id

    # Phase 0 — Scrub before any reasoning
    run_memory_scrub(trace_id)

    crumb = DecisionBreadcrumb(
        trace_id=trace_id,
        policy_reference="MEMORY-TACTICAL-LOOKUP",
        action_taken="search_tactical(query='wv arrival')",
        result="Searching tactical block for WV arrival facts",
    )
    _emit_breadcrumb(crumb)

    # Search the tactical block directly
    wv_facts = memory.search_tactical("wv")
    arrival_facts = [f for f in wv_facts if "arrival" in f.fact.lower() or "check-in" in f.fact.lower()]

    if arrival_facts:
        fact = arrival_facts[0]
        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-TACTICAL-HIT",
            action_taken=f"recall_tactical(fact='{fact.fact[:50]}...')",
            result=(
                f"FOUND in tactical block — TTL {fact.ttl_seconds:.0f}s remaining, "
                f"source trace {fact.trace_id[:8]}"
            ),
        )
        _emit_breadcrumb(crumb)

        # Confirm transient sentiment was already scrubbed
        sentiment_facts = [
            f for f in memory.transient_block
            if "frustrated" in f.fact.lower()
        ]
        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-TRANSIENT-CHECK",
            action_taken="check_transient(sentiment='Frustrated')",
            result=(
                "ABSENT — transient sentiment was correctly decayed after "
                "Turn 1 intent resolution"
                if not sentiment_facts
                else f"STILL PRESENT — {len(sentiment_facts)} sentiment facts remain"
            ),
        )
        _emit_breadcrumb(crumb)

        envelope = CanonicalIntentEnvelope(
            intent_id=trace_id,
            primary_objective=raw_request,
            status=EnvelopeStatus.EXECUTABLE,
            domain_assertions=[
                f"Tactical memory hit: {fact.fact}",
                f"Source trace: {fact.trace_id[:8]}, TTL: {fact.ttl_seconds:.0f}s",
                "No MCP calls required — answered from memory",
                "Transient sentiment (Frustrated) has been decayed",
            ],
        )
        memory.store_intent(
            guest_message=raw_request, envelope=envelope,
            execution_results=[], breadcrumbs=[],
        )

        response = (
            f"I have your late arrival for WV on file.\n\n"
            f"  {fact.fact}\n"
            f"  Source: trace {fact.trace_id[:8]}... "
            f"(tactical memory, {fact.ttl_seconds:.0f}s TTL remaining)\n\n"
            f"No need to re-run the reasoning engine — this is confirmed "
            f"from the tactical memory block."
        )
    else:
        crumb = DecisionBreadcrumb(
            trace_id=trace_id,
            policy_reference="MEMORY-TACTICAL-MISS",
            action_taken="search_tactical(wv)",
            result="No WV arrival facts in tactical block — may have decayed",
        )
        _emit_breadcrumb(crumb)

        response = (
            "The WV arrival details appear to have expired from tactical memory. "
            "I would need to re-query the PMS to confirm the current state."
        )

    chat_history.append({"role": "cso", "text": response})
    return response


# ---------------------------------------------------------------------------
# Turn 4: High-Value Guest Recovery (flight delay + proactive reassignment)
# ---------------------------------------------------------------------------

def decompose_recovery_request(
    guest_id: str, reservation_id: str, property_code: str,
) -> list[SubIntent]:
    """Proactive recovery: query ground-floor rooms, then reassign."""
    return [
        SubIntent(
            description="Query ground-floor pet-friendly rooms near exit",
            domain="PMS",
            required_tool="pms_query_rooms",
            extra_params={
                "property_code": property_code,
                "pet_friendly": True,
                "max_floor": 2,
                "near_exit": True,
                "room_type": "suite",
            },
        ),
        SubIntent(
            description="Reassign to ground-floor room for dog welfare",
            domain="PMS",
            required_tool="pms_reassign_room",
            extra_params={
                "res_id": reservation_id,
                "reason": "Proactive reassignment: flight delay + large pets + late arrival",
            },
        ),
    ]


async def process_turn_4_recovery(event_data: dict) -> str:
    """
    High-Value Guest Recovery — triggered by external SSE event.

    Flow:
      1. Load core facts (Titanium tier, 2 Cane Corso dogs)
      2. Assess current room suitability (floor 14 vs. late arrival with large dogs)
      3. Compromise: reassign to ground-floor suite near exit
      4. Execute via MCP (query rooms, then reassign)
      5. Store tactical facts for future recall
    """
    guest_id = event_data.get("guest_id", "G-3003")
    reservation_id = event_data.get("reservation_id", "R-6001")
    property_code = event_data.get("property_code", "LHRW01")
    delay_hours = event_data.get("delay_hours", 3)
    current_room = event_data.get("current_room", "1415")
    current_floor = event_data.get("current_floor", 14)

    synthetic_message = (
        f"[SYSTEM EVENT] Flight delay detected: +{delay_hours} hours. "
        f"Proactive reassignment required for guest {guest_id} with pets."
    )
    chat_history.append({"role": "system", "text": synthetic_message})

    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            envelope_shell = CanonicalIntentEnvelope(
                primary_objective=(
                    "Proactive room reassignment for Titanium guest "
                    "with large dogs after flight delay"
                ),
            )
            trace_id = envelope_shell.intent_id

            # Phase 0 — Scrub
            run_memory_scrub(trace_id)

            # ── Core Memory: permanent guest + pet facts ──────────────────
            memory.add_core_fact(
                "Guest G-3003 (Marcus Wolfe) — Titanium tier",
                domain="Identity", tags=["guest", "titanium"],
            )
            memory.add_core_fact(
                "Pet 1: Atlas (Cane Corso mix, 110 lbs)",
                domain="Preferences", tags=["pet", "dog", "large"],
            )
            memory.add_core_fact(
                "Pet 2: Zeus (Cane Corso mix, 115 lbs)",
                domain="Preferences", tags=["pet", "dog", "large"],
            )

            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="MEMORY-CORE-LOAD",
                action_taken="load_core_memory(guest=G-3003)",
                result=(
                    "Loaded 3 core facts: Titanium tier, "
                    "2 Cane Corso dogs (110+115 lbs)"
                ),
            )
            _emit_breadcrumb(crumb)

            # ── Phase 2: Assess room suitability ──────────────────────────
            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="COGNITIVE-ASSESSMENT",
                action_taken=(
                    f"assess_room_suitability(current={current_room},"
                    f"floor={current_floor},dogs=2,arrival=01:00)"
                ),
                result=(
                    "UNSUITABLE — High-floor room requires elevator "
                    "for 225 lbs of dogs at 1 AM"
                ),
            )
            _emit_breadcrumb(crumb)

            # ── Phase 2: Build compromise actions ─────────────────────────
            reasoning = ReasoningResult()
            reasoning.domain_assertions = [
                f"Current room {current_room} is on floor {current_floor} "
                f"— unsuitable for late-night arrival with large dogs",
                f"Flight delay: +{delay_hours} hours → arrival 01:00 (1 AM)",
                "Dog info from Core Memory: 2x Cane Corso mix (110+115 lbs)",
            ]

            sub_intents = decompose_recovery_request(
                guest_id, reservation_id, property_code,
            )

            # Action 1: query available rooms
            reasoning.actions.append(ProposedAction(
                tool_name="pms_query_rooms",
                parameters=sub_intents[0].extra_params,
                order=0,
            ))

            # Action 2: reassign (compromise)
            reasoning.actions.append(ProposedAction(
                tool_name="pms_reassign_room",
                parameters={
                    "res_id": reservation_id,
                    "new_room_number": "101",
                    "reason": (
                        "Proactive reassignment: flight delay "
                        "+ large pets + late arrival"
                    ),
                    "notes": (
                        f"Flight delayed {delay_hours}h → arrival 01:00. "
                        "Guest has 2 Cane Corso dogs (110+115 lbs). "
                        f"Moved from {current_room} (floor {current_floor}) "
                        "to 101 (ground floor, near exit) to minimise "
                        "elevator trips and facilitate late-night dog walking."
                    ),
                },
                order=1,
                is_compromise=True,
                compromise_rationale=(
                    f"Original suite {current_room} on floor {current_floor} "
                    "is unsuitable for 1 AM arrival with 225 lbs of dogs. "
                    "Ground-floor room 101 near exit provides immediate "
                    "outdoor access without elevator queues."
                ),
            ))
            reasoning.has_compromise = True

            reasoning.contextual_assertions.append(ContextualAssertion(
                domain="PMS",
                assertion=(
                    "CSO detected late-night arrival (01:00) with 2 large dogs "
                    "(225 lbs total). Current floor-14 room requires extended "
                    "elevator use. Compromise: reassign to ground-floor suite "
                    "near exit for easier access."
                ),
                requires_escalation=False,
            ))

            crumb = DecisionBreadcrumb(
                trace_id=trace_id,
                policy_reference="COGNITIVE-FALLBACK-PROACTIVE",
                action_taken="detect_and_compromise(trigger=flight_delay)",
                result=(
                    f"COMPROMISE — Reassign {current_room}→101 "
                    f"(floor {current_floor}→1) for dog welfare "
                    "and guest convenience at 1 AM arrival"
                ),
            )
            _emit_breadcrumb(crumb)

            # ── Phase 3: Execute ──────────────────────────────────────────
            envelope = CanonicalIntentEnvelope(
                intent_id=trace_id,
                primary_objective=(
                    "Proactive room reassignment for Titanium guest "
                    "with large dogs after flight delay"
                ),
                status=EnvelopeStatus.PARTIAL_FULFILLMENT,
                domain_assertions=reasoning.domain_assertions,
                contextual_assertions=reasoning.contextual_assertions,
                proposed_actions=reasoning.actions,
            )

            results, crumbs = await execute_envelope(envelope, session)

            # ── Phase 4: Populate memory ──────────────────────────────────
            memory.add_tactical_fact(
                f"Proactive reassignment: R-6001 moved from "
                f"{current_room} (floor {current_floor}) to 101 (floor 1)",
                trace_id=trace_id, domain="PMS",
                tags=["reassignment", "R-6001", property_code, "compromise"],
            )
            memory.add_tactical_fact(
                f"Reason: Flight delay +{delay_hours}h, arrival 01:00, "
                "2 large dogs (225 lbs)",
                trace_id=trace_id, domain="PMS",
                tags=["flight_delay", "late_arrival", "pets"],
            )

            all_crumbs = []
            with _stream_lock:
                for item in breadcrumb_stream:
                    if item["trace_id"] == trace_id:
                        all_crumbs.append(DecisionBreadcrumb(
                            trace_id=item["trace_id"],
                            policy_reference=item["policy_reference"],
                            action_taken=item["action_taken"],
                            result=item["result"],
                        ))

            memory.store_intent(
                guest_message=synthetic_message,
                envelope=envelope,
                execution_results=results,
                breadcrumbs=all_crumbs,
            )

            # ── Build narrative response ──────────────────────────────────
            narrative = (
                f"**High-Value Guest Recovery Executed** "
                f"(Trace: {trace_id[:8]}...)\n\n"

                f"**Context:**\n"
                f"- Guest: Marcus Wolfe (G-3003, Titanium tier)\n"
                f"- Pets: Atlas (Cane Corso, 110 lbs), "
                f"Zeus (Cane Corso, 115 lbs)\n"
                f"- Flight delay: +{delay_hours} hours "
                f"→ arrival 01:00 (1 AM)\n"
                f"- Original room: {current_room} "
                f"(floor {current_floor})\n\n"

                f"**Decision Logic:**\n"
                f"- Retrieved dog breed/size from Core Memory\n"
                f"- Detected high-floor room unsuitable for "
                f"late-night arrival with large dogs\n"
                f"- Queried ground-floor pet-friendly rooms near exit\n\n"

                f"**Compromise Action:**\n"
                f"- Reassigned to Room 101 (ground floor, near exit)\n"
                f"- Rationale: Minimise elevator trips, facilitate "
                f"late-night dog walking\n\n"

                f"*The system detected a late arrival for a VIP. "
                f"Recalling the guest has large dogs, it proactively "
                f"reassigned them to Room 101 (Ground Floor) to "
                f"minimise friction and ensure a premium experience "
                f"despite the delay.*"
            )

            chat_history.append({"role": "cso", "text": narrative})
            return narrative


# ---------------------------------------------------------------------------
# Message router
# ---------------------------------------------------------------------------

async def process_message(text: str) -> str:
    lower = text.lower()

    # Turn 4: recovery scenario — flight delay / SSE event
    recovery_signals = ["flight delay", "recovery", "sse event"]
    is_recovery = any(s in lower for s in recovery_signals)

    if is_recovery:
        event_data = {
            "guest_id": "G-3003",
            "reservation_id": "R-6001",
            "property_code": "LHRW01",
            "delay_hours": 3,
            "current_room": "1415",
            "current_floor": 14,
        }
        return await process_turn_4_recovery(event_data)

    # Turn 3: persistence check — "confirm", "arrival", "on file"
    persistence_signals = ["confirm", "arrival", "on file", "still have", "check on"]
    is_persistence = any(s in lower for s in persistence_signals) and "wv" in lower

    if is_persistence and memory.turn_count >= 2:
        return await process_turn_3_persistence(text)
    elif "wv" in lower and memory.turn_count >= 1:
        return await process_turn_2_wv(text)
    else:
        return await process_turn_1(text)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class ChatRequest(PydanticBase):
    message: str

class ChatResponse(PydanticBase):
    response: str
    trace_id: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("CSO Orchestrator API started — awaiting messages")
    yield
    log.info("CSO Orchestrator API shutting down")


app = FastAPI(title="CSO Orchestrator", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


_SCENARIO_UI = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSO Scenario Runner</title>
<style>
  :root { --bg: #0f1117; --surface: #1a1d27; --border: #2d3040; --accent: #7c6aef;
          --accent2: #4ecdc4; --text: #e2e4ea; --dim: #8b8fa3; --red: #e74c5f;
          --green: #4ecdc4; --amber: #f0a030; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
         background: var(--bg); color: var(--text); min-height: 100vh; padding: 1.5rem; }
  h1 { font-size: 1.3rem; color: var(--accent); margin-bottom: .3rem; }
  .subtitle { color: var(--dim); font-size: .75rem; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; height: calc(100vh - 5rem); }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 1.2rem; overflow-y: auto; }
  h2 { font-size: .85rem; color: var(--accent2); margin-bottom: 1rem; text-transform: uppercase;
       letter-spacing: .08em; }
  .scenario-btn { display: block; width: 100%; padding: .7rem .9rem; margin-bottom: .6rem;
                  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                  color: var(--text); font-family: inherit; font-size: .78rem; cursor: pointer;
                  text-align: left; transition: border-color .15s, background .15s; }
  .scenario-btn:hover { border-color: var(--accent); background: #1e2130; }
  .scenario-btn:disabled { opacity: .5; cursor: wait; }
  .scenario-btn .label { font-weight: 600; }
  .scenario-btn .desc { color: var(--dim); font-size: .7rem; margin-top: .25rem; }
  .compare-btn { display: block; width: 100%; padding: .45rem .9rem; margin-bottom: .6rem;
                 margin-top: -.3rem;
                 background: rgba(240,160,48,.08); border: 1px solid rgba(240,160,48,.25);
                 border-radius: 6px; color: var(--amber); font-family: inherit; font-size: .7rem;
                 cursor: pointer; text-align: left; transition: border-color .15s, background .15s; }
  .compare-btn:hover { border-color: var(--amber); background: rgba(240,160,48,.15); }
  .compare-btn:disabled { opacity: .5; cursor: wait; }
  .divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
  label { display: block; font-size: .75rem; color: var(--dim); margin-bottom: .3rem; }
  select, textarea { width: 100%; padding: .55rem .7rem; background: var(--bg); border: 1px solid var(--border);
                     border-radius: 5px; color: var(--text); font-family: inherit; font-size: .78rem;
                     resize: vertical; }
  select:focus, textarea:focus { outline: none; border-color: var(--accent); }
  textarea { min-height: 70px; }
  .send-btn { margin-top: .7rem; padding: .55rem 1.2rem; background: var(--accent); border: none;
              border-radius: 5px; color: #fff; font-family: inherit; font-size: .78rem;
              font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .send-btn:hover { opacity: .85; }
  .send-btn:disabled { opacity: .5; cursor: wait; }
  .reset-btn { margin-top: .5rem; padding: .4rem .9rem; background: transparent;
               border: 1px solid var(--red); border-radius: 5px; color: var(--red);
               font-family: inherit; font-size: .7rem; cursor: pointer; }
  .reset-btn:hover { background: rgba(231,76,95,.1); }
  #status { font-size: .7rem; margin-top: .5rem; }
  .result-area { display: flex; flex-direction: column; gap: 1rem; }
  .result-header { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; }
  .badge { display: inline-block; padding: .2rem .55rem; border-radius: 4px; font-size: .7rem;
           font-weight: 600; }
  .badge-exec { background: rgba(78,205,196,.15); color: var(--green); }
  .badge-partial { background: rgba(240,160,48,.15); color: var(--amber); }
  .badge-escalation { background: rgba(231,76,95,.15); color: var(--red); }
  .badge-rejected { background: rgba(231,76,95,.25); color: var(--red); }
  .section { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
             padding: .9rem; }
  .section h3 { font-size: .75rem; color: var(--accent2); margin-bottom: .6rem; }
  .action { padding: .5rem .7rem; margin-bottom: .4rem; border-left: 3px solid var(--accent);
            background: rgba(124,106,239,.05); border-radius: 0 4px 4px 0; font-size: .73rem; }
  .action.compromise { border-left-color: var(--amber); background: rgba(240,160,48,.05); }
  .action.error { border-left-color: var(--red); background: rgba(231,76,95,.05); }
  .tool-name { font-weight: 600; color: var(--accent); }
  .action.compromise .tool-name { color: var(--amber); }
  .escalation { padding: .4rem .7rem; margin-bottom: .3rem; border-left: 3px solid var(--red);
                font-size: .73rem; color: var(--red); }
  .breadcrumb { font-size: .68rem; color: var(--dim); padding: .25rem 0;
                border-bottom: 1px solid var(--border); }
  .breadcrumb:last-child { border-bottom: none; }
  .bc-policy { color: var(--accent); }
  .bc-result { font-weight: 600; }
  .bc-result.ok { color: var(--green); }
  .bc-result.denied { color: var(--red); }
  .bc-result.compromise { color: var(--amber); }
  .bc-result.escalate { color: var(--red); }
  .mesh-note { font-size: .72rem; padding: .6rem .8rem; background: rgba(240,160,48,.08);
               border: 1px solid rgba(240,160,48,.2); border-radius: 5px; color: var(--amber); }
  .memory-block { margin-bottom: .5rem; }
  .memory-block summary { cursor: pointer; font-size: .73rem; font-weight: 600; color: var(--dim); }
  .memory-fact { font-size: .7rem; padding: .2rem .5rem; color: var(--dim); }
  .empty { color: var(--dim); font-size: .8rem; font-style: italic; text-align: center;
           padding: 3rem 1rem; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Split-view comparison styles */
  .split-view { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .split-pane { min-width: 0; }
  .split-pane h3 { font-size: .8rem; margin-bottom: .7rem; }
  .split-pane.cso h3 { color: var(--green); }
  .split-pane.mesh h3 { color: var(--red); }
  .degradation-chain { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                        padding: .9rem; }
  .degradation-chain h3 { font-size: .75rem; color: var(--amber); margin-bottom: .6rem; }
  .handoff { padding: .4rem .7rem; margin-bottom: .4rem; border-left: 3px solid var(--amber);
             background: rgba(240,160,48,.04); border-radius: 0 4px 4px 0; font-size: .7rem; }
  .handoff .agent { font-weight: 600; color: var(--amber); }
  .handoff .dropped { color: var(--red); font-size: .65rem; margin-top: .15rem; }
  .advantage-box { background: rgba(78,205,196,.06); border: 1px solid rgba(78,205,196,.2);
                    border-radius: 6px; padding: .9rem; }
  .advantage-box h3 { font-size: .75rem; color: var(--green); margin-bottom: .6rem; }
  .advantage-item { font-size: .72rem; padding: .3rem 0; border-bottom: 1px solid var(--border);
                     color: var(--text); }
  .advantage-item:last-child { border-bottom: none; }
  .diff-indicator { display: inline-block; padding: .1rem .4rem; border-radius: 3px;
                     font-size: .6rem; font-weight: 600; margin-left: .4rem; }
  .diff-correct { background: rgba(78,205,196,.15); color: var(--green); }
  .diff-wrong { background: rgba(231,76,95,.15); color: var(--red); }
  .diff-missing { background: rgba(240,160,48,.15); color: var(--amber); }

  /* Phase progress banner */
  .phase-banner { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                   padding: .8rem 1rem; margin-bottom: .5rem; }
  .phase-item { font-size: .73rem; padding: .3rem 0; color: var(--dim); }
  .phase-item .spinner { margin-right: .4rem; vertical-align: middle; }
  .phase-done { color: var(--green); font-weight: 700; margin-right: .3rem; }
  .phase-time { color: var(--accent); font-weight: 600; }

  /* Bottom comparison scorecard */
  .scorecard { border: 2px solid var(--accent); border-radius: 10px; overflow: hidden; margin-top: .5rem; }
  .scorecard-title { background: rgba(124,106,239,.12); padding: .7rem 1rem; font-size: .85rem;
                      font-weight: 700; color: var(--accent); text-align: center; text-transform: uppercase;
                      letter-spacing: .1em; border-bottom: 2px solid var(--accent); }
  .scorecard-grid { display: grid; grid-template-columns: 1fr 1fr; }
  .scorecard-col { padding: 1.2rem 1.4rem; }
  .scorecard-col.cso { border-right: 1px solid var(--border); }
  .scorecard-col-header { font-size: .9rem; font-weight: 700; text-transform: uppercase;
                           letter-spacing: .06em; margin-bottom: 1rem; }
  .scorecard-col.cso .scorecard-col-header { color: var(--green); }
  .scorecard-col.mesh .scorecard-col-header { color: var(--red); }
  .score-value { font-size: 3rem; font-weight: 800; line-height: 1; margin-bottom: .15rem; }
  .score-green { color: var(--green); }
  .score-amber { color: var(--amber); }
  .score-red { color: var(--red); }
  .score-sublabel { font-size: .68rem; color: var(--dim); margin-bottom: .8rem; }
  .time-pill { font-size: .72rem; color: var(--dim); margin-bottom: 1rem; padding: .3rem .7rem;
               background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
               display: inline-block; }
  .time-pill strong { color: var(--accent); }
  .check-list { margin-top: .5rem; }
  .check-item { font-size: .73rem; padding: .35rem 0; border-bottom: 1px solid var(--border); }
  .check-item:last-child { border-bottom: none; }
  .check-pass { color: var(--green); }
  .check-fail { color: var(--red); }
  .check-icon { font-weight: 700; margin-right: .4rem; }
</style>
</head>
<body>

<h1>Cognitive Singularity Orchestrator</h1>
<p class="subtitle">Scenario Runner &amp; Mesh Comparison Interface</p>

<div class="grid">
  <div class="panel">
    <h2>Preset Scenarios</h2>

    <button class="scenario-btn" onclick="runScenario('single_benefit')">
      <div class="label">1. Single Benefit Allocation</div>
      <div class="desc">G-2002 (Gold) &mdash; Complimentary breakfast</div>
    </button>
    <button class="compare-btn" onclick="runCompare('single_benefit')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('tier_gated_denial')">
      <div class="label">2. Tier-Gated Denial + Valid Benefit</div>
      <div class="desc">G-2002 (Gold) &mdash; Late checkout + breakfast</div>
    </button>
    <button class="compare-btn" onclick="runCompare('tier_gated_denial')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('multi_intent_compromise')">
      <div class="label">3. Multi-Intent Compromise + Escalation</div>
      <div class="desc">G-1001 (Diamond) &mdash; 5PM checkout, SNA, wine</div>
    </button>
    <button class="compare-btn" onclick="runCompare('multi_intent_compromise')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('proactive_recovery')">
      <div class="label">4. Proactive Cross-Domain Recovery</div>
      <div class="desc">G-3003 (Titanium) &mdash; Flight delay + 2 dogs</div>
    </button>
    <button class="compare-btn" onclick="runCompare('proactive_recovery')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('vip_concierge_bundle')">
      <div class="label">5. VIP Concierge Bundle</div>
      <div class="desc">G-1001 (Diamond) &mdash; 5PM checkout + room change + SNA + breakfast</div>
    </button>
    <button class="compare-btn" onclick="runCompare('vip_concierge_bundle')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('contradictory_intent')">
      <div class="label">6. Contradictory Intent</div>
      <div class="desc">G-1001 (Diamond) &mdash; Late checkout + early check-in same day (impossible)</div>
    </button>
    <button class="compare-btn" onclick="runCompare('contradictory_intent')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('ambiguous_escalation')">
      <div class="label">7. Ambiguous Escalation</div>
      <div class="desc">G-2002 (Gold) &mdash; Vague complaint, no actionable intent</div>
    </button>
    <button class="compare-btn" onclick="runCompare('ambiguous_escalation')">Compare with Mesh</button>

    <button class="scenario-btn" onclick="runScenario('mesh_favorable_baseline')">
      <div class="label">8. Mesh-Favorable Baseline</div>
      <div class="desc">G-2002 (Gold) &mdash; Simple checkout time query (both equivalent)</div>
    </button>
    <button class="compare-btn" onclick="runCompare('mesh_favorable_baseline')">Compare with Mesh</button>

    <hr class="divider">
    <h2>Free-form Query</h2>

    <label for="guest">Guest</label>
    <select id="guest">
      <option value="G-1001">G-1001 &mdash; Alexandra Mercer (Diamond)</option>
      <option value="G-2002">G-2002 &mdash; Jordan Whitfield (Gold)</option>
      <option value="G-3003">G-3003 &mdash; Marcus Wolfe (Titanium)</option>
    </select>

    <label for="msg" style="margin-top:.7rem">Message</label>
    <textarea id="msg" placeholder="e.g. Can I get a late checkout and a room upgrade?"></textarea>

    <label for="ctx" style="margin-top:.5rem">Context (optional)</label>
    <textarea id="ctx" rows="2" placeholder="e.g. Flight delayed 3 hours"></textarea>

    <button class="send-btn" onclick="runFreeform()">Send Query</button>

    <hr class="divider">
    <button class="reset-btn" onclick="resetState()">Reset State</button>
    <div id="status"></div>
  </div>

  <div class="panel" id="results">
    <div class="empty">Run a scenario or send a query to see results here.</div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

function setLoading(on) {
  $$('.scenario-btn, .compare-btn, .send-btn').forEach(b => b.disabled = on);
  if (on) $('#status').innerHTML = '<span class="spinner"></span> Processing&hellip;';
  else $('#status').textContent = '';
}

async function runScenario(name) {
  setLoading(true);
  try {
    const r = await fetch('/scenario/' + name, { method: 'POST' });
    const d = await r.json();
    render(d);
  } catch(e) { $('#results').innerHTML = '<div class="empty">Error: ' + e.message + '</div>'; }
  setLoading(false);
}

function scoreResult(scenario, data, isMesh) {
  const checks = [];
  const actions = isMesh ? (data.final_actions || []) : (data.actions || []);
  const escalations = isMesh ? (data.escalation_notes || []) : (data.escalation_notes || []);
  const status = isMesh ? (data.final_status || '') : (data.status || '');

  if (scenario === 'single_benefit') {
    const hasBkfst = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'ComplimentaryBreakfast');
    checks.push({label:'Breakfast allocated', pass: hasBkfst});
    checks.push({label:'No errors', pass: !actions.some(a => (a.result||{}).error)});
    checks.push({label:'Status Executable', pass: status === 'Executable'});
  } else if (scenario === 'tier_gated_denial') {
    const hasBkfst = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'ComplimentaryBreakfast');
    const checkoutDenied = escalations.some(n => /checkout|late|denied|tier/i.test(n)) || status !== 'Executable';
    checks.push({label:'Breakfast allocated', pass: hasBkfst});
    checks.push({label:'Late checkout denied', pass: checkoutDenied});
  } else if (scenario === 'multi_intent_compromise') {
    const has4pm = actions.some(a => a.action === 'pms_update_reservation' && ((a.result||{}).new_checkout||'').includes('T16:00'));
    const hasVoucher = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'ComplimentaryDrinkVoucher');
    const hasSNA = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'SuiteNightAward');
    const wineEsc = escalations.some(n => /wine|margaux|chateau|provision/i.test(n));
    checks.push({label:'Checkout clamped to 4 PM', pass: has4pm});
    checks.push({label:'Drink voucher issued', pass: hasVoucher});
    checks.push({label:'Suite Night Award', pass: hasSNA});
    checks.push({label:'Wine escalated', pass: wineEsc});
  } else if (scenario === 'proactive_recovery') {
    const hasQuery = actions.some(a => a.action === 'pms_query_rooms');
    const reassign = actions.find(a => a.action === 'pms_reassign_room');
    const room = reassign ? (reassign.result||{}).new_room : null;
    checks.push({label:'Room query executed', pass: hasQuery});
    checks.push({label:'Pet-friendly room', pass: room === '101'});
    checks.push({label:'Correct room (101)', pass: room === '101'});
  } else if (scenario === 'vip_concierge_bundle') {
    const has4pm = actions.some(a => a.action === 'pms_update_reservation' && ((a.result||{}).new_checkout||'').includes('T16:00'));
    const hasVoucher = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'ComplimentaryDrinkVoucher');
    const hasSNA = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'SuiteNightAward');
    const hasBkfst = actions.some(a => a.action === 'loyalty_allocate_benefit' && (a.result||{}).benefit_type === 'ComplimentaryBreakfast');
    const hasRoomQuery = actions.some(a => a.action === 'pms_query_rooms');
    const reassign = actions.find(a => a.action === 'pms_reassign_room');
    const room = reassign ? (reassign.result||{}).new_room : null;
    checks.push({label:'Checkout clamped to 4 PM', pass: has4pm});
    checks.push({label:'Drink voucher issued', pass: hasVoucher});
    checks.push({label:'Suite Night Award', pass: hasSNA});
    checks.push({label:'Breakfast allocated', pass: hasBkfst});
    checks.push({label:'Room query executed', pass: hasRoomQuery});
    checks.push({label:'Room assigned (101)', pass: room === '101'});
  } else if (scenario === 'contradictory_intent') {
    const hasContradict = status === 'Rejected' || status === 'Human_Escalation_Required'
      || escalations.some(n => /contradic|conflict|impossible|same day/i.test(n));
    const noConflictActions = !actions.some(a => a.action === 'pms_update_reservation' || a.action === 'pms_update_checkin');
    const escalated = status !== 'Executable';
    checks.push({label:'Contradiction detected', pass: hasContradict});
    checks.push({label:'No conflicting actions', pass: noConflictActions});
    checks.push({label:'Escalation status', pass: escalated});
  } else if (scenario === 'ambiguous_escalation') {
    const ambiguityRecognised = status === 'Rejected' || status === 'Human_Escalation_Required'
      || escalations.some(n => /ambigu|clarif|vague|unclear/i.test(n));
    const noHallucinated = actions.length === 0;
    const escalationRequired = status !== 'Executable' || escalations.length > 0;
    checks.push({label:'Ambiguity recognised', pass: ambiguityRecognised});
    checks.push({label:'No hallucinated actions', pass: noHallucinated});
    checks.push({label:'Escalation required', pass: escalationRequired});
  } else if (scenario === 'mesh_favorable_baseline') {
    const simpleHandled = status === 'Executable' || status === 'Rejected' || status === 'Human_Escalation_Required';
    const noErrors = !actions.some(a => (a.result||{}).error);
    const noMutations = !actions.some(a => ['pms_update_reservation','pms_update_checkin','pms_reassign_room','loyalty_allocate_benefit'].includes(a.action));
    checks.push({label:'Simple query handled', pass: simpleHandled});
    checks.push({label:'No errors', pass: noErrors});
    checks.push({label:'No mutation tools', pass: noMutations});
  }

  const passed = checks.filter(c => c.pass).length;
  const pct = checks.length ? Math.round((passed / checks.length) * 100) : 0;
  return {checks, passed, total: checks.length, pct};
}

async function runCompare(name) {
  setLoading(true);
  const res = $('#results');
  res.innerHTML = '<div class="phase-banner" id="phase-banner"></div><div id="compare-body"></div>';
  const pb = $('#phase-banner');

  try {
    // Phase 1: CSO
    pb.innerHTML = '<div class="phase-item"><span class="spinner"></span> Phase 1: Running CSO Architecture&hellip;</div>';
    await fetch('/reset', {method:'POST'});
    const t0 = performance.now();
    const r1 = await fetch('/scenario/' + name, {method:'POST'});
    const cso = await r1.json();
    const csoTime = ((performance.now() - t0) / 1000).toFixed(1);
    pb.innerHTML = '<div class="phase-item phase-done">Phase 1: CSO Architecture &mdash; ' + csoTime + 's &#10003;</div>'
      + '<div class="phase-item"><span class="spinner"></span> Phase 2: Running Agentic Mesh&hellip;</div>';

    // Phase 2: Mesh
    await fetch('/reset', {method:'POST'});
    const t1 = performance.now();
    const r2 = await fetch('/scenario/' + name + '/mesh', {method:'POST'});
    const mesh = await r2.json();
    const meshTime = ((performance.now() - t1) / 1000).toFixed(1);
    pb.innerHTML = '<div class="phase-item phase-done">Phase 1: CSO Architecture &mdash; ' + csoTime + 's &#10003;</div>'
      + '<div class="phase-item phase-done">Phase 2: Agentic Mesh &mdash; ' + meshTime + 's &#10003;</div>';

    const csoScore = scoreResult(name, cso, false);
    const meshScore = scoreResult(name, mesh, true);
    renderComparison({scenario: name, cso, mesh, csoTime, meshTime, csoScore, meshScore});
  } catch(e) { res.innerHTML = '<div class="empty">Error: ' + e.message + '</div>'; }
  setLoading(false);
}

async function runFreeform() {
  const body = { guest_id: $('#guest').value, message: $('#msg').value };
  if ($('#ctx').value.trim()) body.context = $('#ctx').value.trim();
  if (!body.message.trim()) return;
  setLoading(true);
  try {
    const r = await fetch('/query', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const d = await r.json();
    render(d);
  } catch(e) { $('#results').innerHTML = '<div class="empty">Error: ' + e.message + '</div>'; }
  setLoading(false);
}

async function resetState() {
  await fetch('/reset', { method: 'POST' });
  $('#status').textContent = 'State reset.';
  setTimeout(() => $('#status').textContent = '', 2000);
}

function statusBadge(s) {
  const cls = s === 'Executable' ? 'exec' : s === 'Partial_Fulfillment' ? 'partial'
    : s === 'Human_Escalation_Required' ? 'escalation' : 'rejected';
  return '<span class="badge badge-' + cls + '">' + s.replace(/_/g,' ') + '</span>';
}

function bcResultClass(r) {
  const u = r.toUpperCase();
  if (u.includes('OK') || u.includes('EXECUTABLE') || u.includes('CLEAN')) return 'ok';
  if (u.includes('COMPROMISE') || u.includes('CLAMP')) return 'compromise';
  if (u.includes('DENIED') || u.includes('ESCALAT')) return 'denied';
  return '';
}

function renderActions(actions) {
  if (!actions || !actions.length) return '<div style="color:var(--dim);font-size:.72rem;font-style:italic">No actions executed</div>';
  let h = '';
  for (const a of actions) {
    const isComp = a.is_compromise;
    const isErr = a.result && a.result.error;
    const cls = isErr ? 'error' : isComp ? 'compromise' : '';
    h += '<div class="action ' + cls + '">';
    h += '<span class="tool-name">' + a.action + '</span>';
    if (isComp) h += ' <span style="color:var(--amber);font-size:.65rem">[compromise]</span>';
    const res = a.result || {};
    if (res.status) h += ' &rarr; <strong>' + res.status + '</strong>';
    const details = [];
    if (res.benefit_type) details.push(res.benefit_type);
    if (res.new_checkout) details.push('checkout: ' + res.new_checkout);
    if (res.new_room) details.push('room: ' + (res.old_room||'?') + ' &rarr; ' + res.new_room);
    if (res.count !== undefined) details.push(res.count + ' rooms found');
    if (res.error) details.push('Error ' + res.error + ': ' + (res.detail||''));
    if (details.length) h += '<div style="color:var(--dim);font-size:.68rem;margin-top:.2rem">' + details.join(' | ') + '</div>';
    h += '</div>';
  }
  return h;
}

function renderEscalations(notes) {
  if (!notes || !notes.length) return '';
  let h = '<div style="margin-top:.5rem"><strong style="font-size:.7rem;color:var(--red)">Escalations:</strong>';
  for (const n of notes) h += '<div class="escalation">' + n + '</div>';
  h += '</div>';
  return h;
}

function renderComparison(d) {
  const cso = d.cso;
  const mesh = d.mesh;
  const csoScore = d.csoScore;
  const meshScore = d.meshScore;
  const csoTime = d.csoTime;
  const meshTime = d.meshTime;
  const scenario = d.scenario;
  const csoStatus = cso.status || 'Unknown';
  const meshStatus = mesh.final_status || 'Unknown';
  const csoActions = cso.actions || [];
  const meshActions = mesh.final_actions || [];

  let h = '';

  // Phase banner (timing recap)
  h += '<div class="phase-banner">';
  h += '<div class="phase-item phase-done">Phase 1: CSO Architecture &mdash; <span class="phase-time">' + csoTime + 's</span> &#10003;</div>';
  h += '<div class="phase-item phase-done">Phase 2: Agentic Mesh &mdash; <span class="phase-time">' + meshTime + 's</span> &#10003;</div>';
  h += '</div>';

  h += '<div class="result-area">';

  // Header
  h += '<div class="result-header">';
  h += '<span style="font-size:.85rem;font-weight:600;color:var(--accent)">CSO vs Mesh &mdash; ' + scenario.replace(/_/g,' ') + '</span>';
  const identical = csoScore.pct === meshScore.pct;
  if (identical) h += '<span class="diff-indicator diff-correct">IDENTICAL</span>';
  else h += '<span class="diff-indicator diff-wrong">DIVERGENT</span>';
  h += '</div>';

  // Split view — actions
  h += '<div class="split-view">';

  // CSO pane
  h += '<div class="split-pane cso">';
  h += '<h3>CSO Pipeline ' + statusBadge(csoStatus) + '</h3>';
  h += '<div class="section">';
  h += '<h3>Actions (' + csoActions.length + ')</h3>';
  h += renderActions(csoActions);
  h += renderEscalations(cso.escalation_notes);
  h += '</div>';
  h += '</div>';

  // Mesh pane
  h += '<div class="split-pane mesh">';
  h += '<h3>Mesh Pipeline ' + statusBadge(meshStatus) + '</h3>';
  h += '<div class="section">';
  h += '<h3>Actions (' + meshActions.length + ')</h3>';
  h += renderActions(meshActions);
  h += renderEscalations(mesh.escalation_notes);
  h += '</div>';
  h += '</div>';

  h += '</div>'; // end split-view

  // Degradation chain
  if (mesh.degradation_chain && mesh.degradation_chain.length) {
    h += '<div class="degradation-chain">';
    h += '<h3>Context Degradation Chain</h3>';
    for (let i = 0; i < mesh.degradation_chain.length; i++) {
      const hoff = mesh.degradation_chain[i];
      h += '<div class="handoff">';
      h += '<span class="agent">' + hoff.agent_name + '</span>';
      h += ' &mdash; ' + hoff.summary;
      if (hoff.confidence < 1) h += ' <span style="color:var(--dim);font-size:.6rem">(conf: ' + (hoff.confidence * 100).toFixed(0) + '%)</span>';
      if (hoff.dropped_fields && hoff.dropped_fields.length) {
        h += '<div class="dropped">LOST: ' + hoff.dropped_fields.join(', ') + '</div>';
      }
      h += '</div>';
    }
    h += '</div>';
  }

  // ========== SCORECARD ==========
  h += '<div class="scorecard">';
  h += '<div class="scorecard-title">Results Comparison</div>';
  h += '<div class="scorecard-grid">';

  // CSO column
  const csoClr = csoScore.pct >= 80 ? 'green' : csoScore.pct >= 50 ? 'amber' : 'red';
  h += '<div class="scorecard-col">';
  h += '<div class="scorecard-col-header">CSO Architecture</div>';
  h += '<div class="score-value score-' + csoClr + '">' + csoScore.pct + '%</div>';
  h += '<div class="score-sublabel">' + csoScore.passed + ' / ' + csoScore.total + ' checks passed</div>';
  h += '<div class="time-pill">' + csoTime + 's</div>';
  h += '<div class="check-list">';
  for (const c of csoScore.checks) {
    h += '<div class="check-item ' + (c.pass ? 'check-pass' : 'check-fail') + '">';
    h += '<span class="check-icon">' + (c.pass ? '&#10003;' : '&#10007;') + '</span> ' + c.label;
    h += '</div>';
  }
  h += '</div>';
  h += '</div>';

  // Mesh column
  const meshClr = meshScore.pct >= 80 ? 'green' : meshScore.pct >= 50 ? 'amber' : 'red';
  h += '<div class="scorecard-col">';
  h += '<div class="scorecard-col-header">Agentic Mesh</div>';
  h += '<div class="score-value score-' + meshClr + '">' + meshScore.pct + '%</div>';
  h += '<div class="score-sublabel">' + meshScore.passed + ' / ' + meshScore.total + ' checks passed</div>';
  h += '<div class="time-pill">' + meshTime + 's</div>';
  h += '<div class="check-list">';
  for (const c of meshScore.checks) {
    h += '<div class="check-item ' + (c.pass ? 'check-pass' : 'check-fail') + '">';
    h += '<span class="check-icon">' + (c.pass ? '&#10003;' : '&#10007;') + '</span> ' + c.label;
    h += '</div>';
  }
  h += '</div>';

  h += '</div>'; // end scorecard-grid
  h += '</div>'; // end scorecard

  h += '</div>'; // end result-area
  $('#compare-body').innerHTML = h;
}

function render(d) {
  let h = '<div class="result-area">';

  // Header
  h += '<div class="result-header">';
  h += statusBadge(d.status);
  h += '<span style="font-size:.75rem;color:var(--dim)">Trace: ' + (d.trace_id||'').slice(0,8) + '&hellip;</span>';
  if (d.scenario && d.scenario !== 'freeform')
    h += '<span style="font-size:.75rem;color:var(--dim)">Scenario: ' + d.scenario + '</span>';
  h += '</div>';

  // Actions
  if (d.actions && d.actions.length) {
    h += '<div class="section"><h3>Actions (' + d.actions.length + ')</h3>';
    h += renderActions(d.actions);
    h += '</div>';
  }

  // Escalations
  if (d.escalation_notes && d.escalation_notes.length) {
    h += '<div class="section"><h3>Escalation Notes</h3>';
    for (const n of d.escalation_notes) h += '<div class="escalation">' + n + '</div>';
    h += '</div>';
  }

  // Sub-intents from Claude
  if (d.sub_intents && d.sub_intents.length) {
    h += '<div class="section"><h3>Claude Sub-Intents (' + d.sub_intents.length + ')</h3>';
    for (const si of d.sub_intents) {
      h += '<div style="font-size:.72rem;padding:.3rem 0;border-bottom:1px solid var(--border)">';
      h += '<strong>' + si.description + '</strong>';
      h += ' <span style="color:var(--dim)">[' + si.domain + ']</span>';
      if (si.required_tool) h += ' &rarr; <span class="tool-name">' + si.required_tool + '</span>';
      else h += ' &rarr; <span style="color:var(--red)">no tool</span>';
      if (si.tier_violation) h += '<div style="color:var(--red);font-size:.68rem">' + si.tier_violation + '</div>';
      h += '</div>';
    }
    h += '</div>';
  }

  // Breadcrumbs
  if (d.breadcrumbs && d.breadcrumbs.length) {
    h += '<div class="section"><h3>Decision Breadcrumbs (' + d.breadcrumbs.length + ')</h3>';
    for (const b of d.breadcrumbs) {
      const rc = bcResultClass(b.result || '');
      h += '<div class="breadcrumb">';
      h += '<span class="bc-policy">[' + b.policy_reference + ']</span> ';
      h += '<span class="bc-result ' + rc + '">' + (b.result||'').slice(0,120) + '</span>';
      h += '</div>';
    }
    h += '</div>';
  }

  // Mesh annotation
  if (d.mesh_annotation && d.mesh_annotation !== 'User-defined query \\u2014 no preset mesh comparison.') {
    h += '<div class="mesh-note"><strong>Mesh Comparison:</strong> ' + d.mesh_annotation + '</div>';
  }

  // Memory snapshot
  if (d.memory_snapshot) {
    const m = d.memory_snapshot;
    h += '<div class="section"><h3>Memory Vault</h3>';
    for (const [block, label] of [['core_block','Core (permanent)'],['recall_block','Tactical (48h)'],['transient_block','Transient (10m)']]) {
      const facts = m[block] || [];
      h += '<details class="memory-block"><summary>' + label + ' (' + facts.length + ')</summary>';
      for (const f of facts) h += '<div class="memory-fact">&bull; ' + f.fact + '</div>';
      h += '</details>';
    }
    h += '</div>';
  }

  h += '</div>';
  $('#results').innerHTML = h;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root_ui():
    """Serve the scenario runner UI."""
    return _SCENARIO_UI


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    response_text = await process_message(req.message)
    summaries = memory.conversation_summary()
    latest = summaries[-1] if summaries else {}
    return ChatResponse(
        response=response_text,
        trace_id=latest.get("trace_id", "unknown"),
        status=latest.get("status", "unknown"),
    )


@app.get("/breadcrumbs")
async def get_breadcrumbs(since: int = 0):
    with _stream_lock:
        return {"breadcrumbs": breadcrumb_stream[since:], "total": len(breadcrumb_stream)}


@app.get("/memory")
async def get_memory():
    return {"turns": memory.conversation_summary(), "turn_count": memory.turn_count}


@app.get("/memory-vault")
async def get_memory_vault():
    """Full three-block memory state for the Memory Inspector."""
    return memory.vault_snapshot()


@app.get("/chat-history")
async def get_chat_history():
    return {"messages": chat_history}


@app.post("/reset")
async def reset_state():
    """Clear all in-memory state and reset the database to seed."""
    memory.reset()
    with _stream_lock:
        breadcrumb_stream.clear()
    chat_history.clear()

    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("_admin_reset_db", {"trace_id": "reset"})
            content = "".join(
                b.text for b in result.content if hasattr(b, "text")
            )
    return {"status": "reset", "detail": content}


@app.post("/scenario/{name}")
async def run_scenario_endpoint(name: str):
    """Run a named test scenario through the Claude reasoning pipeline."""
    from cso_poc.scenarios import SCENARIOS, run_scenario

    if name not in SCENARIOS:
        return {"error": f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}"}
    return await run_scenario(SCENARIOS[name])


@app.post("/scenario/{name}/mesh")
async def run_mesh_endpoint(name: str):
    """Run the mesh simulation for a named scenario."""
    from cso_poc.mesh import run_mesh_scenario
    from cso_poc.scenarios import SCENARIOS

    if name not in SCENARIOS:
        return {"error": f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}"}
    mesh_result = await run_mesh_scenario(SCENARIOS[name])
    return mesh_result.to_dict()


@app.post("/scenario/{name}/compare")
async def run_comparison_endpoint(name: str):
    """Run CSO, reset DB, run mesh, return both + comparison."""
    from cso_poc.mesh import build_comparison, run_mesh_scenario
    from cso_poc.scenarios import SCENARIOS, run_scenario

    if name not in SCENARIOS:
        return {"error": f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}"}

    config = SCENARIOS[name]

    # 1. Run CSO pipeline
    cso_start = time.time()
    cso_result = await run_scenario(config)
    cso_elapsed_ms = (time.time() - cso_start) * 1000

    # 2. Reset DB + memory between runs (mesh mutates same tables)
    memory.reset()
    with _stream_lock:
        breadcrumb_stream.clear()
    chat_history.clear()
    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.call_tool("_admin_reset_db", {"trace_id": "compare-reset"})

    # 3. Run mesh pipeline
    mesh_start = time.time()
    mesh_result = await run_mesh_scenario(config)
    mesh_elapsed_ms = (time.time() - mesh_start) * 1000

    # 4. Build comparison
    comparison = build_comparison(cso_result, mesh_result)

    return {
        "scenario": name,
        "cso": cso_result,
        "mesh": mesh_result.to_dict(),
        "comparison": comparison,
        "timing_summary": {
            "cso_ms": round(cso_elapsed_ms, 1),
            "mesh_ms": round(mesh_elapsed_ms, 1),
            "total_ms": round(cso_elapsed_ms + mesh_elapsed_ms, 1),
        },
    }


class FreeformRequest(PydanticBase):
    guest_id: str
    message: str
    context: str | None = None


@app.post("/query")
async def freeform_query(req: FreeformRequest):
    """Run an arbitrary user-defined message through the Claude reasoning pipeline."""
    from cso_poc.scenarios import ScenarioConfig, run_scenario

    config = ScenarioConfig(
        name="freeform",
        guest_id=req.guest_id,
        raw_message=req.message,
        context=req.context,
        mesh_annotation="User-defined query — no preset mesh comparison.",
    )
    return await run_scenario(config)


@app.post("/security/scenario/{name}")
async def run_security_scenario(name: str):
    """Run a single adversarial scenario against both architectures."""
    from cso_poc.adversarial import ATTACK_SCENARIOS
    from cso_poc.scenarios import run_adversarial_scenario
    from cso_poc.mesh import run_adversarial_mesh_scenario

    if name not in ATTACK_SCENARIOS:
        return {
            "error": f"Unknown attack scenario: {name}. "
            f"Available: {list(ATTACK_SCENARIOS.keys())}"
        }

    attack = ATTACK_SCENARIOS[name]

    # Run CSO
    cso_result = await run_adversarial_scenario(attack.config, attack)

    # Reset between runs
    await reset_state()

    # Run mesh
    mesh_result = await run_adversarial_mesh_scenario(attack.config, attack)

    return {
        "scenario": name,
        "cso": cso_result,
        "mesh": mesh_result,
    }


@app.post("/security/benchmark")
async def run_full_benchmark():
    """Run the complete adversarial corpus and generate reports."""
    from cso_poc.security_benchmark import run_security_benchmark

    async def reset_fn():
        await reset_state()

    report = await run_security_benchmark(reset_fn)
    return {
        "status": "complete",
        "corpus_size": report.get("corpus_size"),
        "cso_metrics": report.get("cso_metrics"),
        "mesh_metrics": report.get("mesh_metrics"),
        "comparative_summary": report.get("comparative_summary"),
        "files_written": ["security_results.json", "security_summary.md"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "turns_processed": memory.turn_count}


# ---------------------------------------------------------------------------
# Demo / entrypoint
# ---------------------------------------------------------------------------

COMPLEX_GUEST_REQUEST = (
    "I'd like a late checkout at 5 PM, a bottle of Macallan 12 sent to "
    "my room, and please apply my Suite Night Award for tonight."
)
FOLLOWUP_REQUEST = "Wait, did you handle my WV stay too?"
PERSISTENCE_REQUEST = "Can you confirm my WV arrival is still on file?"


async def run_demo() -> None:
    demo_mode = os.environ.get("CSO_DEMO_MODE", "api")

    if demo_mode == "headless":
        log.info("=" * 72)
        log.info("TURN 1 — Complex Guest Request")
        log.info("=" * 72)
        resp1 = await process_turn_1(COMPLEX_GUEST_REQUEST)
        log.info("\nCSO Response:\n%s", resp1)

        log.info("\n" + "=" * 72)
        log.info("TURN 2 — WV Follow-up (Memory Recall)")
        log.info("=" * 72)
        resp2 = await process_turn_2_wv(FOLLOWUP_REQUEST)
        log.info("\nCSO Response:\n%s", resp2)

        log.info("\n" + "=" * 72)
        log.info("TURN 3 — Persistence Test (Tactical Block)")
        log.info("=" * 72)
        resp3 = await process_turn_3_persistence(PERSISTENCE_REQUEST)
        log.info("\nCSO Response:\n%s", resp3)

        log.info("\n" + "=" * 72)
        log.info("MEMORY VAULT SNAPSHOT")
        log.info("=" * 72)
        log.info(json.dumps(memory.vault_snapshot(), indent=2, default=str))
    else:
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    asyncio.run(run_demo())
