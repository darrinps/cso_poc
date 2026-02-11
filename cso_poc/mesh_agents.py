"""
Agentic Mesh — Real Claude-Powered Agents

Each agent makes its own LLM call (Haiku), can invoke MCP tools via
Anthropic's tool_use feature, and produces a text handoff for the next
agent.  Context degrades naturally because each handoff is a lossy text
summary — no hardcoding required.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import anthropic
from mcp import ClientSession

from cso_poc.model_config import MESH_MODEL as _MESH_SPEC

log = logging.getLogger("cso.mesh_agents")

# ---------------------------------------------------------------------------
# Model configuration (imported from model_config.py)
# ---------------------------------------------------------------------------

MESH_MODEL = _MESH_SPEC.name
MESH_TEMPERATURE = _MESH_SPEC.temperature
MESH_MAX_TOKENS = _MESH_SPEC.max_tokens


@dataclass
class AgentResult:
    """Result of a single agent's LLM call (possibly with tool use)."""

    agent_name: str
    summary: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool_use format) per agent type
# ---------------------------------------------------------------------------

AGENT_TOOLS: dict[str, list[dict]] = {
    "ProfileAgent": [],  # profile injected via pre-fetched resource
    "CoordinatorAgent": [],  # text compression only
    "ReservationAgent": [
        {
            "name": "pms_update_reservation",
            "description": (
                "Update a hotel reservation's checkout time. "
                "Returns confirmation or a PolicyViolation error."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "res_id": {"type": "string", "description": "Reservation ID"},
                    "checkout_time": {
                        "type": "string",
                        "description": "New checkout datetime in ISO format",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Operational notes",
                        "default": "",
                    },
                },
                "required": ["res_id", "checkout_time"],
            },
        },
    ],
    "LoyaltyAgent": [
        {
            "name": "loyalty_allocate_benefit",
            "description": (
                "Allocate a loyalty benefit for a guest. Valid benefit_type values: "
                "SuiteNightAward, LateCheckout, PointsBonus, "
                "ComplimentaryBreakfast, ComplimentaryDrinkVoucher."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "guest_id": {"type": "string", "description": "Guest loyalty ID"},
                    "benefit_type": {
                        "type": "string",
                        "description": "Benefit to allocate",
                    },
                },
                "required": ["guest_id", "benefit_type"],
            },
        },
    ],
    "RoomsAgent": [
        {
            "name": "pms_query_rooms",
            "description": (
                "Query available rooms at a property with optional filters."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "property_code": {"type": "string"},
                    "pet_friendly": {"type": "boolean", "default": False},
                    "max_floor": {"type": "integer", "default": 100},
                    "near_exit": {"type": "boolean", "default": False},
                    "room_type": {"type": "string", "default": "standard"},
                },
                "required": ["property_code"],
            },
        },
        {
            "name": "pms_reassign_room",
            "description": "Reassign a reservation to a different room.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "res_id": {"type": "string"},
                    "new_room_number": {"type": "string"},
                    "reason": {"type": "string"},
                    "notes": {"type": "string", "default": ""},
                },
                "required": ["res_id", "new_room_number", "reason"],
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# System prompts per agent type
# ---------------------------------------------------------------------------

AGENT_PROMPTS: dict[str, str] = {
    "ProfileAgent": (
        "You are a Profile Agent in a hotel management system.\n"
        "Summarize the guest profile in 3-5 sentences. Include ONLY stated facts.\n"
        "Do NOT infer, assume, or add information not present in the profile.\n"
        "Include: guest ID, name, loyalty tier, current reservation details, "
        "any preferences, and pet details if present."
    ),
    "CoordinatorAgent": (
        "You are a Coordinator Agent routing tasks between specialist agents.\n"
        "Summarize the previous agent's output in 2-3 sentences MAX.\n"
        "Include ONLY facts explicitly stated. Do NOT add assumptions.\n"
        "Do NOT infer intent beyond what is explicitly stated.\n"
        "Your summary will be passed to the next specialist agent."
    ),
    "ReservationAgent": (
        "You are a Reservation Agent handling hotel checkout changes.\n"
        "Use the pms_update_reservation tool to process checkout requests.\n"
        "If a checkout time is DENIED (PolicyViolation), retry at 4PM "
        "(2026-02-03T16:00:00 or the appropriate date) as a fallback.\n"
        "Report factually what happened: what was requested, what was the result.\n"
        "Do NOT issue compensatory benefits — that is another agent's job."
    ),
    "LoyaltyAgent": (
        "You are a Loyalty Agent allocating guest benefits.\n"
        "Use the loyalty_allocate_benefit tool to allocate ONLY benefits "
        "explicitly requested in the handoff message.\n"
        "Do NOT allocate benefits that were not mentioned.\n"
        "Do NOT infer compensatory benefits unless explicitly instructed.\n"
        "Report what you allocated."
    ),
    "RoomsAgent": (
        "You are a Rooms Agent handling room queries and assignments.\n"
        "Use pms_query_rooms to find available rooms using ONLY the constraints "
        "mentioned in the handoff message.\n"
        "Then use pms_reassign_room to assign the best room from results.\n"
        "Pick the first/best room returned by the query.\n"
        "Report which room was assigned and why."
    ),
}


# ---------------------------------------------------------------------------
# Core agent runner — tool-use loop
# ---------------------------------------------------------------------------

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def run_agent(
    agent_name: str,
    handoff_message: str,
    mcp_session: ClientSession,
    trace_id: str,
) -> AgentResult:
    """
    Run a single mesh agent: send to Claude Haiku, handle tool_use blocks
    by executing them via MCP, then feed tool_result back until we get a
    text-only response (max 5 iterations).
    """
    client = _get_client()
    prompt = AGENT_PROMPTS.get(agent_name, "You are a helpful assistant.")
    agent_tools = AGENT_TOOLS.get(agent_name, [])

    messages: list[dict] = [{"role": "user", "content": handoff_message}]
    all_tool_calls: list[dict[str, Any]] = []

    create_kwargs: dict[str, Any] = {
        "model": MESH_MODEL,
        "system": prompt,
        "messages": messages,
        "max_tokens": MESH_MAX_TOKENS,
        "temperature": MESH_TEMPERATURE,
    }
    if agent_tools:
        create_kwargs["tools"] = agent_tools

    for iteration in range(5):
        try:
            response = await client.messages.create(**create_kwargs)
        except Exception as exc:
            log.error("Agent %s API error: %s", agent_name, exc)
            return AgentResult(
                agent_name=agent_name,
                summary=f"Error: {exc}",
                error=str(exc),
            )

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_blocks:
            summary = " ".join(b.text for b in text_blocks).strip()
            log.info("Agent %s completed (iter %d): %s",
                     agent_name, iteration, summary[:120])
            return AgentResult(
                agent_name=agent_name,
                summary=summary,
                tool_calls=all_tool_calls,
                raw_response=summary,
            )

        # Append assistant response with tool_use blocks
        messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        # Execute each tool via MCP and build tool_result messages
        tool_results = []
        for block in tool_blocks:
            tool_input = dict(block.input)
            tool_input["trace_id"] = trace_id
            log.info("Agent %s calling tool %s(%s)",
                     agent_name, block.name, json.dumps(tool_input))
            try:
                mcp_result = await mcp_session.call_tool(block.name, tool_input)
                result_text = "".join(
                    part.text for part in mcp_result.content
                    if hasattr(part, "text")
                )
            except Exception as exc:
                log.error("MCP tool %s error: %s", block.name, exc)
                result_text = json.dumps({"error": str(exc)})

            all_tool_calls.append({
                "tool": block.name,
                "input": {k: v for k, v in tool_input.items()
                          if k != "trace_id"},
                "result": result_text,
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})
        create_kwargs["messages"] = messages

    # Exhausted iterations — return whatever we have
    summary = "Agent reached max iterations without final text response."
    log.warning("Agent %s hit max iterations", agent_name)
    return AgentResult(
        agent_name=agent_name,
        summary=summary,
        tool_calls=all_tool_calls,
        error="max_iterations",
    )


@dataclass
class AgentHandoff:
    """A single handoff between two agents in the mesh."""

    agent_name: str
    summary: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    dropped_fields: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "summary": self.summary,
            "structured_data": self.structured_data,
            "dropped_fields": self.dropped_fields,
            "confidence": self.confidence,
        }


@dataclass
class MeshStep:
    """One step in the mesh pipeline execution."""

    agent: str
    action: str
    input_context: dict[str, Any]
    output_summary: str
    dropped_context: list[str] = field(default_factory=list)
    tool_call: str | None = None
    tool_params: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] = field(default_factory=dict)
    is_correct: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "input_context": self.input_context,
            "output_summary": self.output_summary,
            "dropped_context": self.dropped_context,
            "tool_call": self.tool_call,
            "tool_params": self.tool_params,
            "tool_result": self.tool_result,
            "is_correct": self.is_correct,
        }


@dataclass
class MeshResult:
    """Complete result of a mesh pipeline run."""

    scenario_name: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[MeshStep] = field(default_factory=list)
    final_actions: list[dict[str, Any]] = field(default_factory=list)
    final_status: str = "unknown"
    escalation_notes: list[str] = field(default_factory=list)
    context_loss_summary: list[str] = field(default_factory=list)
    degradation_chain: list[AgentHandoff] = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "trace_id": self.trace_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_actions": self.final_actions,
            "final_status": self.final_status,
            "escalation_notes": self.escalation_notes,
            "context_loss_summary": self.context_loss_summary,
            "degradation_chain": [h.to_dict() for h in self.degradation_chain],
            "timing": self.timing,
        }
