"""
Layer 4 — Scenario Pipeline

Defines 4 test scenarios of increasing difficulty and a run_scenario()
function that drives the full CSO pipeline using Claude reasoning.

Each scenario demonstrates CSO's advantage over agentic mesh by
preserving full context in a single reasoning pass.
"""

from __future__ import annotations

import json
import logging
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

            # ── 5. Two-phase handling for Test 4 ──────────────────────
            # If we have both pms_query_rooms and pms_reassign_room,
            # execute the query first and inject best room into reassign
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

            # ── 7. Build and execute envelope ─────────────────────────
            envelope = CanonicalIntentEnvelope(
                intent_id=trace_id,
                primary_objective=config.raw_message,
                status=status,
                domain_assertions=reasoning.domain_assertions,
                contextual_assertions=reasoning.contextual_assertions,
                proposed_actions=reasoning.actions,
                escalation_notes=reasoning.escalation_notes,
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

            return {
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
            }
