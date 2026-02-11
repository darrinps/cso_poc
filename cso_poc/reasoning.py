"""
Layer 4 — Claude Reasoning Engine

Uses Claude to decompose raw guest messages into structured SubIntent
objects, informed by guest profile, available tools, and policy rules.

The system prompt encodes all hospitality policies so Claude can reason
about tier gates, checkout ceilings, pet constraints, and proactive
recovery patterns.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from cso_poc.model_config import get_cso_model

log = logging.getLogger("cso.reasoning")

_CSO_SPEC = get_cso_model()
ANTHROPIC_MODEL = _CSO_SPEC.name

SYSTEM_PROMPT = """\
You are the reasoning core of the Cognitive Singularity Orchestrator (CSO), \
a hospitality AI that decomposes guest requests into structured sub-intents.

You will receive:
- A guest message (raw text or system event)
- A guest profile (JSON with guest_id, name, loyalty_tier, preferences, stays)
- A tool manifest (list of available MCP tool names)
- Optional context (e.g. flight delay info)

Your job: produce a JSON array of SubIntent objects. Each SubIntent has:
{
  "description": "what this sub-intent does",
  "domain": "PMS" | "Loyalty" | "Provisions" | "Rooms" | ...,
  "required_tool": "tool_name" | null,
  "original_parameter_value": "value or null",
  "policy_ceiling": "value or null",
  "extra_params": { ... },
  "tier_violation": "explanation string or null"
}

## Available Tools
- pms_update_reservation(res_id, checkout_time, notes): Update checkout time
- pms_update_checkin(res_id, checkin_time, notes): Update check-in time
- loyalty_allocate_benefit(guest_id, benefit_type): Allocate a benefit
  - BenefitTypes: SuiteNightAward, LateCheckout, PointsBonus, ComplimentaryBreakfast, ComplimentaryDrinkVoucher
- pms_query_rooms(property_code, pet_friendly, max_floor, near_exit, room_type): Query available rooms
- pms_reassign_room(res_id, new_room_number, reason, notes): Reassign guest to different room

## Policy Rules
1. **Checkout ceiling**: Maximum checkout is 4 PM (16:00). If guest asks for later, set \
original_parameter_value to what they asked and policy_ceiling to "16:00".
2. **Late checkout extension (past 11 AM)**: Only Diamond and Titanium tier guests can \
extend checkout past 11 AM. If a lower-tier guest requests this, set tier_violation.
3. **Diamond-only benefits**: SuiteNightAward and LateCheckout benefits require Diamond \
or Titanium tier. If a lower-tier guest requests these, set tier_violation.
4. **General benefits**: PointsBonus, ComplimentaryBreakfast, ComplimentaryDrinkVoucher \
are available to ALL tiers.
5. **Proactive reassignment**: pms_reassign_room for proactive reasons requires Titanium tier.
6. **Pet room constraints**: For guests with large dogs, query rooms with pet_friendly=true, \
max_floor=2, near_exit=true, room_type=suite.
7. **No tool for provisions/wine/spirits**: There is no MCP tool for delivering items like \
wine or spirits. Set required_tool=null and the orchestrator will escalate.
8. **Compromise pattern**: When checkout exceeds ceiling, the orchestrator will clamp to \
the ceiling AND issue a ComplimentaryDrinkVoucher as compensation. You just need to set \
original_parameter_value and policy_ceiling correctly.
9. **Contradictory intents**: If a guest's request contains logically contradictory actions \
(e.g. late checkout AND early check-in on the same day that would overlap), set \
required_tool=null for the conflicting sub-intents and include a description explaining \
the contradiction. The orchestrator will escalate.
10. **Ambiguous / vague requests**: If the guest message expresses sentiment but contains \
no actionable intent (e.g. "This isn't what I expected"), do NOT hallucinate tool calls. \
Return a single SubIntent with required_tool=null, domain="Escalation", and a description \
requesting clarification. The orchestrator will escalate to staff.
11. **Informational queries**: If the guest asks a simple informational question \
(e.g. "What time is checkout?"), return a SubIntent with required_tool=null and \
domain="Informational". No tool calls are needed for pure lookups.

## Extra Params Guide
- For pms_update_reservation: include res_id, date (YYYY-MM-DD portion)
- For pms_update_checkin: include res_id, date
- For loyalty_allocate_benefit: include guest_id, benefit_type
- For pms_query_rooms: include property_code, pet_friendly, max_floor, near_exit, room_type
- For pms_reassign_room: include res_id, new_room_number, reason, notes

## Output Format
Respond with ONLY a JSON array. No markdown, no explanation, no code fences.
Example: [{"description": "...", "domain": "...", ...}]
"""


async def decompose_with_claude(
    raw_message: str,
    guest_profile: dict[str, Any],
    tool_manifest: list[str],
    context: str | None = None,
) -> list[dict]:
    """
    Use Claude to decompose a raw guest message into SubIntent dicts.

    Returns a list of dicts matching the SubIntent fields.
    On failure, returns a single escalation SubIntent.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — returning escalation fallback")
        return [_escalation_fallback(raw_message, "No API key configured")]

    client = anthropic.AsyncAnthropic(api_key=api_key)

    user_content = json.dumps({
        "guest_message": raw_message,
        "guest_profile": guest_profile,
        "available_tools": tool_manifest,
        "context": context,
    }, default=str, indent=2)

    try:
        response = await client.messages.create(
            model=_CSO_SPEC.name,
            max_tokens=_CSO_SPEC.max_tokens,
            temperature=_CSO_SPEC.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        text = response.content[0].text.strip()
        log.info("Claude raw response: %s", text[:500])

        # Strip markdown fences if Claude added them despite instructions
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        sub_intents = json.loads(text)
        if not isinstance(sub_intents, list):
            sub_intents = [sub_intents]

        # Validate structure
        validated = []
        for si in sub_intents:
            validated.append({
                "description": si.get("description", "unknown"),
                "domain": si.get("domain", "unknown"),
                "required_tool": si.get("required_tool"),
                "original_parameter_value": si.get("original_parameter_value"),
                "policy_ceiling": si.get("policy_ceiling"),
                "extra_params": si.get("extra_params", {}),
                "tier_violation": si.get("tier_violation"),
            })
        return validated

    except json.JSONDecodeError as e:
        log.error("Claude response not valid JSON: %s", e)
        return [_escalation_fallback(raw_message, f"Unparseable Claude response: {e}")]
    except Exception as e:
        log.error("Claude API error: %s", e)
        return [_escalation_fallback(raw_message, f"Claude API error: {e}")]


def _escalation_fallback(raw_message: str, reason: str) -> dict:
    """Return a single escalation SubIntent when Claude fails."""
    return {
        "description": f"Escalation: {raw_message[:100]}",
        "domain": "Escalation",
        "required_tool": None,
        "original_parameter_value": None,
        "policy_ceiling": None,
        "extra_params": {},
        "tier_violation": None,
    }
