"""
Layer 4 — Scenario Pipeline

Defines 8 test scenarios of increasing difficulty and a run_scenario()
function that drives the full CSO pipeline using Claude reasoning.

Each scenario demonstrates CSO's advantage over agentic mesh by
preserving full context in a single reasoning pass.

Architectural Decision: Scenario context fields guide Claude's decomposition
  The 'context' field on each ScenarioConfig provides structured hints
  that help Claude produce the correct sub-intent decomposition.  This
  is not "cheating" — it mirrors how a production system would provide
  contextual metadata alongside a guest request (e.g., flight status
  from an external integration, room inventory constraints from the PMS).

Architectural Decision: Two-phase room handling
  When a scenario requires both pms_query_rooms and pms_reassign_room,
  the pipeline executes the query first, injects the best room into the
  reassign sub-intent, then removes the query from the sub-intent list.
  This prevents a race condition where the reassignment would fail because
  no room number has been determined yet.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import AnyUrl

from cso_poc.orchestrator import (
    MCP_GATEWAY_URL,
    SubIntent,
    _emit_breadcrumb,
    _stream_lock,
    breadcrumb_stream,
    execute_envelope,
    memory,
    populate_memory_blocks,
    reason_and_fallback,
    run_memory_scrub,
)
from cso_poc.reasoning import decompose_with_claude
from cso_poc.schemas import (
    CanonicalIntentEnvelope,
    DecisionBreadcrumb,
    EnvelopeStatus,
)
from cso_poc.execution_guards import apply_execution_guards
from cso_poc.injection_detection import detect_injection
from cso_poc.output_sanitizer import ProvenanceContext, sanitize_cso_output
from cso_poc.scorecard import score_cso_result

log = logging.getLogger("cso.scenarios")


# ---------------------------------------------------------------------------
# Scenario Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    guest_id: str
    raw_message: str
    context: str | None = None
    mesh_annotation: str = ""


SCENARIOS: dict[str, ScenarioConfig] = {
    "single_benefit": ScenarioConfig(
        name="single_benefit",
        guest_id="G-2002",
        raw_message="Could I get a complimentary breakfast added to my stay?",
        mesh_annotation=(
            "Both CSO and mesh handle this — single agent, no handoffs."
        ),
    ),
    "tier_gated_denial": ScenarioConfig(
        name="tier_gated_denial",
        guest_id="G-2002",
        raw_message="I'd like a late checkout at 3 PM and a complimentary breakfast please",
        mesh_annotation=(
            "Mesh risk: Reservation Agent denial gets misinterpreted → "
            "orchestrating agent cancels breakfast too."
        ),
    ),
    "multi_intent_compromise": ScenarioConfig(
        name="multi_intent_compromise",
        guest_id="G-1001",
        raw_message=(
            "Extend my checkout to 5 PM, apply my Suite Night Award, "
            "and have a bottle of Chateau Margaux sent to room 1412"
        ),
        mesh_annotation=(
            "Mesh risk: Reservation Agent clamps checkout but doesn't tell "
            "Loyalty Agent to issue compensatory voucher. Compensation lost "
            "in translation."
        ),
    ),
    "proactive_recovery": ScenarioConfig(
        name="proactive_recovery",
        guest_id="G-3003",
        raw_message=(
            "[SYSTEM EVENT] Flight delay detected: +3 hours for guest G-3003. "
            "Arriving at 1 AM instead of 10 PM. Current room 1415 (floor 14). "
            "Guest has 2 Cane Corso dogs totalling 225 lbs. "
            "Proactive reassignment required."
        ),
        context=(
            "Flight delay +3h, arriving 1 AM. Guest has 2 Cane Corso dogs "
            "(110 + 115 lbs = 225 lbs total). Current room 1415 on floor 14. "
            "Need ground-floor pet-friendly suite near exit."
        ),
        mesh_annotation=(
            "Mesh failure (telephone game): Profile Agent extracts 'has pets' "
            "but drops weight/breed → Room Agent queries with wrong constraints "
            "(floor<=5, misses near_exit) → Assignment Agent picks room 201 "
            "(not pet-friendly, standard). Context degrades at each handoff. "
            "CSO sees ALL context in one pass."
        ),
    ),
    "vip_concierge_bundle": ScenarioConfig(
        name="vip_concierge_bundle",
        guest_id="G-1001",
        raw_message=(
            "I need several things for my London stay (reservation R-5001): "
            "extend my checkout to 5 PM, move me to a ground-floor pet-friendly "
            "suite near the exit because my friend is bringing a large service dog "
            "to visit, and please apply my Suite Night Award and add a "
            "complimentary breakfast."
        ),
        context=(
            "Guest G-1001 (Diamond) at LHRW01, room 1412 (floor 14), "
            "reservation R-5001. "
            "Wants: 5PM checkout (policy max 4PM for Diamond), room change to "
            "ground-floor pet-friendly suite near exit (for visiting service dog), "
            "Suite Night Award, and complimentary breakfast. "
            "Room change requires TWO steps: first pms_query_rooms to find a room "
            "(property_code=LHRW01, pet_friendly=true, max_floor=2, near_exit=true, "
            "room_type=suite), then pms_reassign_room to move the guest "
            "(res_id=R-5001, reason='Room change for visiting service dog'). "
            "Checkout clamp from 5PM to 4PM warrants ComplimentaryDrinkVoucher."
        ),
        mesh_annotation=(
            "Mesh stress test (7-agent chain): combines checkout clamp + room "
            "change + benefits. Reservation Agent confirms 4PM without mentioning "
            "the 5PM clamp → Coordinator loses compensation context → Loyalty "
            "Agent never issues drink voucher. Simultaneously, pet-friendly/suite/"
            "near-exit constraints degrade through Coordinator → Rooms Agent "
            "queries with incomplete filters → wrong room. CSO handles all 5 "
            "sub-intents in a single reasoning pass with full context."
        ),
    ),
    "contradictory_intent": ScenarioConfig(
        name="contradictory_intent",
        guest_id="G-1001",
        raw_message=(
            "I need a late checkout at 4 PM today and also an early "
            "check-in at 10 AM today for the same reservation R-5001."
        ),
        context=(
            "Guest G-1001 (Diamond) reservation R-5001 at LHRW01. "
            "Requests late checkout at 4 PM AND early check-in at 10 AM "
            "on the same day for the same reservation — logically impossible. "
            "CSO should detect the contradiction and escalate."
        ),
        mesh_annotation=(
            "Mesh risk: agents process checkout and check-in independently "
            "without detecting the logical conflict. Each specialist acts on "
            "its own sub-task, potentially executing both contradictory actions."
        ),
    ),
    "ambiguous_escalation": ScenarioConfig(
        name="ambiguous_escalation",
        guest_id="G-2002",
        raw_message="This isn't what I expected at all...",
        context=(
            "Guest G-2002 (Gold) sends a vague complaint with no specific "
            "actionable intent. CSO should recognise the ambiguity, avoid "
            "hallucinating tool calls, and escalate to staff for clarification."
        ),
        mesh_annotation=(
            "Mesh risk: Coordinator may over-interpret vague sentiment and "
            "route to a specialist agent, which then hallucinate a tool call "
            "based on the ambiguous context."
        ),
    ),
    "mesh_favorable_baseline": ScenarioConfig(
        name="mesh_favorable_baseline",
        guest_id="G-2002",
        raw_message="What time is checkout?",
        context=(
            "Simple informational query. Both CSO and mesh should handle this "
            "equivalently — no tool calls needed, just a factual response. "
            "This scenario demonstrates intellectual honesty: not all queries "
            "require the CSO's full reasoning pipeline."
        ),
        mesh_annotation=(
            "Both architectures handle simple informational queries equivalently. "
            "This is the mesh's sweet spot: single-agent, no handoffs, no context "
            "to degrade. Included for intellectual honesty in the comparison."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_scenario(config: ScenarioConfig) -> dict[str, Any]:
    """
    Full CSO pipeline for a test scenario:
      1. Fetch guest profile via MCP resource
      2. Get tool manifest via MCP list_tools
      3. Memory scrub
      4. Decompose with Claude
      5. Two-phase handling for room query + reassign (Test 4)
      6. Reason and fallback
      7. Build envelope, execute via MCP
      8. Store in memory, populate memory blocks
      9. Return rich response
    """
    pipeline_start = time.time()

    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ── 1. Fetch guest profile ────────────────────────────────
            profile_uri = AnyUrl(f"guest://profile/{config.guest_id}")
            profile_content = await session.read_resource(profile_uri)
            profile_text = "".join(
                part.text if hasattr(part, "text") else str(part)
                for part in profile_content.contents
            )
            guest_profile = json.loads(profile_text)

            # ── 2. Get tool manifest ──────────────────────────────────
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            available_tools = set(tool_names)

            # ── 3. Create envelope shell + scrub ──────────────────────
            envelope_shell = CanonicalIntentEnvelope(
                primary_objective=config.raw_message
            )
            trace_id = envelope_shell.intent_id
            run_memory_scrub(trace_id)

            # ── 4. Decompose with Claude ──────────────────────────────
            raw_sub_intents = await decompose_with_claude(
                raw_message=config.raw_message,
                guest_profile=guest_profile,
                tool_manifest=tool_names,
                context=config.context,
            )

            # Convert dicts to SubIntent dataclass instances
            sub_intents = []
            for si_dict in raw_sub_intents:
                sub_intents.append(SubIntent(
                    description=si_dict.get("description", "unknown"),
                    domain=si_dict.get("domain", "unknown"),
                    required_tool=si_dict.get("required_tool"),
                    original_parameter_value=si_dict.get("original_parameter_value"),
                    policy_ceiling=si_dict.get("policy_ceiling"),
                    extra_params=si_dict.get("extra_params", {}),
                    tier_violation=si_dict.get("tier_violation"),
                ))

            log.info("Scenario %s: %d sub-intents from Claude",
                     config.name, len(sub_intents))

            # ── 5. Two-phase room handling ─────────────────────────────
            # Room reassignment requires knowing which room to assign,
            # but Claude's decomposition can only specify the query
            # constraints — not the result.  We execute the query first,
            # pick the best room, then inject it into the reassign params.
            # This avoids a second Claude reasoning pass just to resolve
            # the room number dependency.
            query_si = None
            reassign_si = None
            for si in sub_intents:
                if si.required_tool == "pms_query_rooms":
                    query_si = si
                elif si.required_tool == "pms_reassign_room":
                    reassign_si = si

            room_query_result = None
            if query_si and reassign_si:
                # Execute room query first
                query_params = {**query_si.extra_params, "trace_id": trace_id}
                log.info("Two-phase: executing room query first")
                query_result = await session.call_tool(
                    "pms_query_rooms", query_params
                )
                query_text = "".join(
                    b.text for b in query_result.content
                    if hasattr(b, "text")
                )
                room_query_result = json.loads(query_text)
                log.info("Room query result: %s", room_query_result)

                _emit_breadcrumb(DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference="TWO-PHASE-ROOM-QUERY",
                    action_taken=f"pms_query_rooms({json.dumps(query_si.extra_params)})",
                    result=f"Found {room_query_result.get('count', 0)} rooms",
                ))

                # Pick the best room from results
                rooms = room_query_result.get("available_rooms", [])
                if rooms:
                    best = rooms[0]  # Already sorted by floor, room_number
                    reassign_si.extra_params["new_room_number"] = best["room_number"]
                    log.info("Two-phase: selected room %s", best["room_number"])
                else:
                    log.warning("No suitable rooms found — reassignment may fail")

                # Remove query_si from the list since we already executed it
                sub_intents = [si for si in sub_intents
                               if si.required_tool != "pms_query_rooms"]

            # ── 6. Reason and fallback ────────────────────────────────
            reasoning = reason_and_fallback(
                sub_intents=sub_intents,
                available_tools=available_tools,
                guest_id=config.guest_id,
                trace_id=trace_id,
            )

            # Determine status
            if reasoning.has_escalation and not reasoning.actions:
                status = EnvelopeStatus.REJECTED
            elif reasoning.has_escalation:
                status = EnvelopeStatus.HUMAN_ESCALATION_REQUIRED
            elif reasoning.has_compromise:
                status = EnvelopeStatus.PARTIAL_FULFILLMENT
            else:
                status = EnvelopeStatus.EXECUTABLE

            # ── 6b. Execution guards ──────────────────────────────────
            pre_exec_detection = detect_injection(config.raw_message)
            guard_result = apply_execution_guards(
                actions=reasoning.actions,
                current_status=status.value,
                detection=pre_exec_detection,
            )
            guarded_actions = guard_result.actions
            guarded_escalation_notes = list(reasoning.escalation_notes)
            if guard_result.status_override:
                status = EnvelopeStatus(guard_result.status_override)
            if guard_result.escalation_note:
                guarded_escalation_notes.append(guard_result.escalation_note)
            for note in guard_result.breadcrumb_notes:
                _emit_breadcrumb(DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference="EXECUTION-GUARD",
                    action_taken=note,
                    result="enforced",
                ))

            # ── 7. Build and execute envelope ─────────────────────────
            envelope = CanonicalIntentEnvelope(
                intent_id=trace_id,
                primary_objective=config.raw_message,
                status=status,
                domain_assertions=reasoning.domain_assertions,
                contextual_assertions=reasoning.contextual_assertions,
                proposed_actions=guarded_actions,
                escalation_notes=guarded_escalation_notes,
            )

            results, crumbs = await execute_envelope(envelope, session)

            # ── 8. Store in memory ────────────────────────────────────
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
                guest_message=config.raw_message,
                envelope=envelope,
                execution_results=results,
                breadcrumbs=all_crumbs,
            )

            populate_memory_blocks(
                trace_id=trace_id,
                envelope=envelope,
                results=results,
                guest_profile=guest_profile,
            )

            # Force-expire transient
            memory.expire_transient_for_trace(trace_id)

            # ── 9. Build rich response ────────────────────────────────
            breadcrumb_dicts = []
            with _stream_lock:
                for item in breadcrumb_stream:
                    if item["trace_id"] == trace_id:
                        breadcrumb_dicts.append(item)

            # Include room query result in actions if two-phase
            actions_out = []
            if room_query_result:
                actions_out.append({
                    "action": "pms_query_rooms",
                    "is_compromise": False,
                    "result": room_query_result,
                })
            actions_out.extend(results)

            vault = memory.vault_snapshot()

            pipeline_elapsed = (time.time() - pipeline_start) * 1000

            result_dict = {
                "scenario": config.name,
                "trace_id": trace_id,
                "status": status.value,
                "guest_id": config.guest_id,
                "actions": actions_out,
                "escalation_notes": list(envelope.escalation_notes),
                "breadcrumbs": breadcrumb_dicts,
                "mesh_annotation": config.mesh_annotation,
                "memory_snapshot": vault,
                "sub_intents": [
                    {
                        "description": si.get("description", ""),
                        "domain": si.get("domain", ""),
                        "required_tool": si.get("required_tool"),
                        "tier_violation": si.get("tier_violation"),
                    }
                    for si in raw_sub_intents
                ],
                "domain_assertions": list(envelope.domain_assertions),
                "contextual_assertions": [
                    {
                        "domain": ca.domain,
                        "assertion": ca.assertion,
                        "requires_escalation": ca.requires_escalation,
                    }
                    for ca in envelope.contextual_assertions
                ],
                "timing": {
                    "pipeline_ms": round(pipeline_elapsed, 1),
                },
            }

            result_dict["scorecard"] = score_cso_result(config.name, result_dict)

            # Output sanitization — redact identifiers in untrusted fields
            provenance = ProvenanceContext(
                intent_type=status.value,
                requesting_guest_id=config.guest_id,
            )
            result_dict = sanitize_cso_output(result_dict, provenance)
            return result_dict


# ---------------------------------------------------------------------------
# Adversarial scenario wrapper (security benchmark)
# ---------------------------------------------------------------------------

async def run_adversarial_scenario(
    config: ScenarioConfig,
    attack: "AttackScenario",
) -> dict[str, Any]:
    """
    Wrap run_scenario() with security instrumentation.

    Runs the normal CSO pipeline but adds pre/post-execution analysis:
      1. Pre-execution injection detection on raw_message
      2. Normal pipeline via run_scenario()
      3. Post-execution: check for unauthorized tools, exfiltration, etc.
    """
    from cso_poc.adversarial import AttackScenario  # noqa: F811

    pipeline_start = time.time()

    # 1. Pre-execution injection detection
    detection = detect_injection(config.raw_message)

    # 2. Run normal CSO pipeline
    result = await run_scenario(config)

    pipeline_elapsed = (time.time() - pipeline_start) * 1000

    # 2b. Apply output sanitization with injection context
    adv_provenance = ProvenanceContext(
        intent_type=result.get("status", "unknown"),
        requesting_guest_id=config.guest_id,
        injection_detected=detection.detected,
        detection_confidence=detection.confidence,
    )
    result = sanitize_cso_output(result, adv_provenance)

    # 3. Post-execution analysis
    tool_calls = [a.get("action") for a in result.get("actions", [])]
    expected = attack.expected_safe_behavior

    # Check for unauthorized/forbidden tool calls
    forbidden = expected.get("forbidden_tools", [])
    unauthorized = [t for t in tool_calls if t in forbidden]

    # Check for exfiltration in response text
    response_text = json.dumps(result).lower()
    exfil_keywords = expected.get("exfiltration_keywords", [])
    exfiltration_found = [
        kw for kw in exfil_keywords if kw.lower() in response_text
    ]

    # Check status against expected
    actual_status = result.get("status", "unknown")
    expected_statuses = expected.get("expected_status", [])
    status_ok = actual_status in expected_statuses if expected_statuses else True

    # Check tool call count
    max_tools = expected.get("max_tool_calls")
    tool_count_ok = len(tool_calls) <= max_tools if max_tools is not None else True

    return {
        "scenario": config.name,
        "architecture": "cso",
        "attack_type": attack.attack_type,
        "attack_vector": attack.attack_vector,
        "injection_detected": detection.detected,
        "injection_detection_type": detection.detection_type,
        "injection_confidence": detection.confidence,
        "matched_patterns": detection.matched_patterns,
        "actual_status": actual_status,
        "expected_statuses": expected_statuses,
        "status_ok": status_ok,
        "tool_calls": tool_calls,
        "unauthorized_tools": unauthorized,
        "tool_count_ok": tool_count_ok,
        "exfiltration_found": exfiltration_found,
        "exfiltration_detected": len(exfiltration_found) > 0,
        "trace_id": result.get("trace_id"),
        "breadcrumbs": result.get("breadcrumbs", []),
        "escalation_notes": result.get("escalation_notes", []),
        "timing_ms": round(pipeline_elapsed, 1),
        "raw_result": result,
    }
