"""
Agentic Mesh — Real Claude-Powered Agent Pipelines

Each scenario runs a chain of Claude Haiku agents that communicate through
text summaries.  Context degrades naturally because each Coordinator handoff
is a lossy 2-3 sentence compression.

Agent chains per scenario:
  1. single_benefit:        Profile → Coordinator → Loyalty
  2. tier_gated_denial:     Profile → Coordinator → Reservation → Coordinator → Loyalty
  3. multi_intent_compromise: Profile → Coordinator → Reservation → Coordinator → Loyalty → Coordinator (wine)
  4. proactive_recovery:    Profile → Coordinator → Rooms
"""

from __future__ import annotations

import json
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import AnyUrl

from cso_poc.mesh_agents import (
    AgentHandoff,
    AgentResult,
    MeshResult,
    MeshStep,
    run_agent,
)
from cso_poc.orchestrator import MCP_GATEWAY_URL
from cso_poc.scenarios import ScenarioConfig

log = logging.getLogger("cso.mesh")


# ---------------------------------------------------------------------------
# Helper: convert AgentResult into MeshStep + AgentHandoff
# ---------------------------------------------------------------------------

def _record_agent(
    result: MeshResult,
    agent_result: AgentResult,
    action_label: str,
    input_context: dict,
) -> None:
    """Append a MeshStep and AgentHandoff from an AgentResult."""
    # Build step
    tool_call = None
    tool_params: dict = {}
    tool_result_parsed: dict = {}
    if agent_result.tool_calls:
        last_tc = agent_result.tool_calls[-1]
        tool_call = last_tc["tool"]
        tool_params = last_tc["input"]
        try:
            tool_result_parsed = json.loads(last_tc["result"])
        except (json.JSONDecodeError, TypeError):
            tool_result_parsed = {"raw": last_tc["result"]}

    step = MeshStep(
        agent=agent_result.agent_name,
        action=action_label,
        input_context=input_context,
        output_summary=agent_result.summary[:200],
        tool_call=tool_call,
        tool_params=tool_params,
        tool_result=tool_result_parsed,
    )
    result.steps.append(step)

    # Build handoff
    handoff = AgentHandoff(
        agent_name=agent_result.agent_name,
        summary=agent_result.summary[:300],
    )
    result.degradation_chain.append(handoff)


def _collect_tool_actions(
    result: MeshResult, agent_result: AgentResult,
) -> None:
    """Add all tool calls from an AgentResult to final_actions."""
    for tc in agent_result.tool_calls:
        try:
            parsed = json.loads(tc["result"])
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw": tc["result"]}
        result.final_actions.append({
            "action": tc["tool"],
            "is_compromise": False,
            "result": parsed,
        })


# ---------------------------------------------------------------------------
# Scenario 1 — Single Benefit (control: both succeed)
# ---------------------------------------------------------------------------

async def _mesh_single_benefit(
    session: ClientSession, config: ScenarioConfig, profile: dict,
) -> MeshResult:
    result = MeshResult(scenario_name=config.name)
    trace_id = result.trace_id

    # ProfileAgent: summarise guest profile
    profile_msg = (
        f"Summarize this guest profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"The guest's request: {config.raw_message}"
    )
    profile_result = await run_agent(
        "ProfileAgent", profile_msg, session, trace_id,
    )
    _record_agent(result, profile_result, "summarize_profile",
                  {"guest_id": config.guest_id})

    # CoordinatorAgent: compress to handoff
    coord_msg = (
        f"Previous agent summary:\n{profile_result.summary}\n\n"
        f"Route the guest's request to the appropriate specialist. "
        f"Summarize what needs to happen in 2-3 sentences."
    )
    coord_result = await run_agent(
        "CoordinatorAgent", coord_msg, session, trace_id,
    )
    _record_agent(result, coord_result, "coordinate",
                  {"profile_summary": profile_result.summary[:100]})

    # LoyaltyAgent: allocate benefit
    loyalty_msg = (
        f"Coordinator handoff:\n{coord_result.summary}\n\n"
        f"Guest ID: {config.guest_id}\n"
        f"Allocate the requested benefit."
    )
    loyalty_result = await run_agent(
        "LoyaltyAgent", loyalty_msg, session, trace_id,
    )
    _record_agent(result, loyalty_result, "allocate_benefit",
                  {"coordinator_summary": coord_result.summary[:100]})
    _collect_tool_actions(result, loyalty_result)

    result.final_status = "Executable"
    return result


# ---------------------------------------------------------------------------
# Scenario 2 — Tier-Gated Denial (mesh fails: denial cascade)
# ---------------------------------------------------------------------------

async def _mesh_tier_gated_denial(
    session: ClientSession, config: ScenarioConfig, profile: dict,
) -> MeshResult:
    result = MeshResult(scenario_name=config.name)
    trace_id = result.trace_id

    # ProfileAgent
    profile_msg = (
        f"Summarize this guest profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"The guest's request: {config.raw_message}"
    )
    profile_result = await run_agent(
        "ProfileAgent", profile_msg, session, trace_id,
    )
    _record_agent(result, profile_result, "summarize_profile",
                  {"guest_id": config.guest_id})

    # CoordinatorAgent: route to reservation first
    coord1_msg = (
        f"Previous agent summary:\n{profile_result.summary}\n\n"
        f"The guest wants a late checkout AND breakfast. "
        f"Route the checkout request to the Reservation specialist first. "
        f"Summarize in 2-3 sentences."
    )
    coord1_result = await run_agent(
        "CoordinatorAgent", coord1_msg, session, trace_id,
    )
    _record_agent(result, coord1_result, "coordinate_checkout",
                  {"profile_summary": profile_result.summary[:100]})

    # ReservationAgent: try late checkout (will be denied for Gold)
    stay = profile.get("current_stay") or (profile.get("stays", [{}])[0])
    res_id = stay.get("reservation_id", "R-5002")
    res_msg = (
        f"Coordinator handoff:\n{coord1_result.summary}\n\n"
        f"Reservation ID: {res_id}\n"
        f"Process the late checkout request."
    )
    res_result = await run_agent(
        "ReservationAgent", res_msg, session, trace_id,
    )
    _record_agent(result, res_result, "checkout_request",
                  {"coordinator_summary": coord1_result.summary[:100]})
    _collect_tool_actions(result, res_result)

    # CoordinatorAgent: process reservation result
    coord2_msg = (
        f"Previous agent (Reservation) result:\n{res_result.summary}\n\n"
        f"Summarize the outcome and determine what to do next. "
        f"Summarize in 2-3 sentences."
    )
    coord2_result = await run_agent(
        "CoordinatorAgent", coord2_msg, session, trace_id,
    )
    _record_agent(result, coord2_result, "coordinate_after_reservation",
                  {"reservation_summary": res_result.summary[:100]})

    # LoyaltyAgent: may or may not get breakfast depending on
    # how the coordinator compressed the denial
    loyalty_msg = (
        f"Coordinator handoff:\n{coord2_result.summary}\n\n"
        f"Guest ID: {config.guest_id}\n"
        f"Allocate any requested benefits."
    )
    loyalty_result = await run_agent(
        "LoyaltyAgent", loyalty_msg, session, trace_id,
    )
    _record_agent(result, loyalty_result, "allocate_benefits",
                  {"coordinator_summary": coord2_result.summary[:100]})
    _collect_tool_actions(result, loyalty_result)

    # Determine status based on what actually happened
    has_actions = len(result.final_actions) > 0
    has_error = any(
        a.get("result", {}).get("error") for a in result.final_actions
    )
    if not has_actions or (has_actions and all(
        a.get("result", {}).get("error") for a in result.final_actions
    )):
        result.final_status = "Rejected"
    elif has_error:
        result.final_status = "Partial_Fulfillment"
    else:
        result.final_status = "Executable"

    # Build context loss summary from the degradation chain
    result.context_loss_summary = [
        f"{h.agent_name}: {h.summary[:80]}"
        for h in result.degradation_chain
    ]
    return result


# ---------------------------------------------------------------------------
# Scenario 3 — Multi-Intent Compromise (mesh loses compensation)
# ---------------------------------------------------------------------------

async def _mesh_multi_intent(
    session: ClientSession, config: ScenarioConfig, profile: dict,
) -> MeshResult:
    result = MeshResult(scenario_name=config.name)
    trace_id = result.trace_id

    stays = profile.get("stays", [])
    lhrw_stay = next(
        (s for s in stays if s.get("property_code") == "LHRW01"), stays[0]
    )
    res_id = lhrw_stay.get("reservation_id", "R-5001")

    # ProfileAgent
    profile_msg = (
        f"Summarize this guest profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"The guest's request: {config.raw_message}"
    )
    profile_result = await run_agent(
        "ProfileAgent", profile_msg, session, trace_id,
    )
    _record_agent(result, profile_result, "summarize_profile",
                  {"guest_id": config.guest_id})

    # CoordinatorAgent: route checkout first
    coord1_msg = (
        f"Previous agent summary:\n{profile_result.summary}\n\n"
        f"The guest has multiple requests. Route the checkout extension "
        f"to the Reservation specialist first. Summarize in 2-3 sentences."
    )
    coord1_result = await run_agent(
        "CoordinatorAgent", coord1_msg, session, trace_id,
    )
    _record_agent(result, coord1_result, "coordinate_checkout",
                  {"profile_summary": profile_result.summary[:100]})

    # ReservationAgent: try 5PM checkout (will be denied, should retry 4PM)
    res_msg = (
        f"Coordinator handoff:\n{coord1_result.summary}\n\n"
        f"Reservation ID: {res_id}\n"
        f"Guest requested checkout at 5PM (17:00). Process this request. "
        f"The checkout date is 2026-02-03."
    )
    res_result = await run_agent(
        "ReservationAgent", res_msg, session, trace_id,
    )
    _record_agent(result, res_result, "checkout_request",
                  {"coordinator_summary": coord1_result.summary[:100]})
    _collect_tool_actions(result, res_result)

    # CoordinatorAgent: compress reservation result → route to loyalty
    coord2_msg = (
        f"Previous agent (Reservation) result:\n{res_result.summary}\n\n"
        f"Now route the loyalty benefits to the Loyalty specialist. "
        f"The guest also requested a Suite Night Award. "
        f"Summarize in 2-3 sentences."
    )
    coord2_result = await run_agent(
        "CoordinatorAgent", coord2_msg, session, trace_id,
    )
    _record_agent(result, coord2_result, "coordinate_loyalty",
                  {"reservation_summary": res_result.summary[:100]})

    # LoyaltyAgent: allocate SNA (and maybe not voucher)
    loyalty_msg = (
        f"Coordinator handoff:\n{coord2_result.summary}\n\n"
        f"Guest ID: {config.guest_id}\n"
        f"Allocate the requested loyalty benefits."
    )
    loyalty_result = await run_agent(
        "LoyaltyAgent", loyalty_msg, session, trace_id,
    )
    _record_agent(result, loyalty_result, "allocate_benefits",
                  {"coordinator_summary": coord2_result.summary[:100]})
    _collect_tool_actions(result, loyalty_result)

    # CoordinatorAgent: handle wine request (should escalate)
    coord3_msg = (
        f"Previous agents have handled checkout and loyalty.\n"
        f"Remaining request: guest wants a bottle of Chateau Margaux "
        f"sent to room 1412. There is no tool available for wine delivery. "
        f"Summarize the situation in 2-3 sentences."
    )
    coord3_result = await run_agent(
        "CoordinatorAgent", coord3_msg, session, trace_id,
    )
    _record_agent(result, coord3_result, "coordinate_wine",
                  {"remaining": "wine delivery"})

    # Check for escalation in wine handling
    wine_summary = coord3_result.summary.lower()
    if any(w in wine_summary for w in ["no tool", "manual", "staff",
                                        "escalat", "cannot", "unavailable"]):
        result.escalation_notes.append(
            "Wine delivery requires manual staff intervention"
        )

    result.final_status = "Human_Escalation_Required"
    result.context_loss_summary = [
        f"{h.agent_name}: {h.summary[:80]}"
        for h in result.degradation_chain
    ]
    return result


# ---------------------------------------------------------------------------
# Scenario 4 — Proactive Recovery (mesh assigns wrong room)
# ---------------------------------------------------------------------------

async def _mesh_proactive_recovery(
    session: ClientSession, config: ScenarioConfig, profile: dict,
) -> MeshResult:
    result = MeshResult(scenario_name=config.name)
    trace_id = result.trace_id
    stays = profile.get("stays", [])
    lhrw_stay = next(
        (s for s in stays if s.get("property_code") == "LHRW01"), stays[0]
    )
    res_id = lhrw_stay.get("reservation_id", "R-6001")

    # ProfileAgent: summarise — this is where lossy compression begins
    profile_msg = (
        f"Summarize this guest profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"The guest's situation: {config.raw_message}"
    )
    profile_result = await run_agent(
        "ProfileAgent", profile_msg, session, trace_id,
    )
    _record_agent(result, profile_result, "summarize_profile",
                  {"guest_id": config.guest_id})

    # CoordinatorAgent: compress profile → route to rooms
    coord_msg = (
        f"Previous agent summary:\n{profile_result.summary}\n\n"
        f"The guest needs a room change due to flight delay. "
        f"Route to the Rooms specialist with the relevant details. "
        f"Summarize in 2-3 sentences."
    )
    coord_result = await run_agent(
        "CoordinatorAgent", coord_msg, session, trace_id,
    )
    _record_agent(result, coord_result, "coordinate_rooms",
                  {"profile_summary": profile_result.summary[:100]})

    # RoomsAgent: query + reassign based on coordinator's (lossy) handoff
    rooms_msg = (
        f"Coordinator handoff:\n{coord_result.summary}\n\n"
        f"Property code: LHRW01\n"
        f"Reservation ID: {res_id}\n"
        f"Find available rooms and reassign the guest."
    )
    rooms_result = await run_agent(
        "RoomsAgent", rooms_msg, session, trace_id,
    )
    _record_agent(result, rooms_result, "query_and_reassign",
                  {"coordinator_summary": coord_result.summary[:100]})
    _collect_tool_actions(result, rooms_result)

    result.final_status = "Partial_Fulfillment"
    result.context_loss_summary = [
        f"{h.agent_name}: {h.summary[:80]}"
        for h in result.degradation_chain
    ]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MESH_HANDLERS = {
    "single_benefit": _mesh_single_benefit,
    "tier_gated_denial": _mesh_tier_gated_denial,
    "multi_intent_compromise": _mesh_multi_intent,
    "proactive_recovery": _mesh_proactive_recovery,
}


async def run_mesh_scenario(config: ScenarioConfig) -> MeshResult:
    """Execute the real Claude-powered mesh pipeline for a scenario."""
    handler = _MESH_HANDLERS.get(config.name)
    if handler is None:
        return MeshResult(
            scenario_name=config.name,
            final_status="error",
            context_loss_summary=[f"No mesh handler for scenario: {config.name}"],
        )

    async with sse_client(MCP_GATEWAY_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Fetch guest profile (same as CSO)
            profile_uri = AnyUrl(f"guest://profile/{config.guest_id}")
            profile_content = await session.read_resource(profile_uri)
            profile_text = "".join(
                part.text if hasattr(part, "text") else str(part)
                for part in profile_content.contents
            )
            guest_profile = json.loads(profile_text)

            return await handler(session, config, guest_profile)


def build_comparison(cso_result: dict, mesh_result: MeshResult) -> dict:
    """Produce a side-by-side structural comparison."""
    cso_actions = cso_result.get("actions", [])
    mesh_actions = mesh_result.final_actions

    cso_tool_names = [a.get("action") for a in cso_actions]
    mesh_tool_names = [a.get("action") for a in mesh_actions]

    # Action diff
    action_diff = []
    all_tools = set(cso_tool_names + mesh_tool_names)
    for tool in all_tools:
        cso_count = cso_tool_names.count(tool)
        mesh_count = mesh_tool_names.count(tool)
        if cso_count != mesh_count:
            action_diff.append({
                "tool": tool,
                "cso_count": cso_count,
                "mesh_count": mesh_count,
                "difference": "CSO has more" if cso_count > mesh_count else "Mesh has more",
            })

    # Check specific results for correctness
    cso_room = None
    mesh_room = None
    for a in cso_actions:
        if a.get("action") == "pms_reassign_room":
            cso_room = a.get("result", {}).get("new_room")
    for a in mesh_actions:
        if a.get("action") == "pms_reassign_room":
            mesh_room = a.get("result", {}).get("new_room")

    cso_has_voucher = any(
        a.get("action") == "loyalty_allocate_benefit"
        and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
        for a in cso_actions
    )
    mesh_has_voucher = any(
        a.get("action") == "loyalty_allocate_benefit"
        and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
        for a in mesh_actions
    )

    both_succeed = (
        cso_result.get("status") == mesh_result.final_status
        and len(action_diff) == 0
    )

    # Build advantage explanation
    advantages = []
    if cso_has_voucher and not mesh_has_voucher:
        advantages.append(
            "CSO issued compensatory DrinkVoucher for checkout clamp — "
            "mesh lost this context at the Coordinator handoff"
        )
    if cso_room and mesh_room and cso_room != mesh_room:
        advantages.append(
            f"CSO assigned room {cso_room} (correct) vs mesh room {mesh_room} — "
            f"mesh lost pet/constraint details in agent handoff compression"
        )
    if not mesh_actions and cso_actions:
        advantages.append(
            "CSO preserved independent intents (e.g. breakfast) even when "
            "one intent was denied — mesh cascaded the denial to all intents"
        )

    return {
        "both_succeed": both_succeed,
        "cso_status": cso_result.get("status"),
        "mesh_status": mesh_result.final_status,
        "cso_action_count": len(cso_actions),
        "mesh_action_count": len(mesh_actions),
        "action_diff": action_diff,
        "context_lost": mesh_result.context_loss_summary,
        "cso_advantage": advantages if advantages else ["Both pipelines produced equivalent results"],
        "room_comparison": {
            "cso_room": cso_room,
            "mesh_room": mesh_room,
        } if cso_room or mesh_room else None,
        "voucher_comparison": {
            "cso_has_voucher": cso_has_voucher,
            "mesh_has_voucher": mesh_has_voucher,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_result(call_result) -> dict:
    """Extract and parse JSON from an MCP tool call result."""
    content_text = "".join(
        block.text for block in call_result.content if hasattr(block, "text")
    )
    try:
        return json.loads(content_text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": content_text}
