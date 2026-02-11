"""
Quantitative Scoring Module

Scores CSO and mesh results against predefined criteria for each scenario.
Produces structured JSON with: intents detected/resolved, tools called/expected,
context items preserved/dropped, policy violations, escalation accuracy,
compromise detection score, overall score, and model config.
"""

from __future__ import annotations

from typing import Any

from cso_poc.model_config import MODEL_CONFIG

# ---------------------------------------------------------------------------
# Scenario criteria — expected outcomes for all 8 scenarios
# ---------------------------------------------------------------------------

SCENARIO_CRITERIA: dict[str, dict[str, Any]] = {
    "single_benefit": {
        "expected_intents": 1,
        "expected_tools": ["loyalty_allocate_benefit"],
        "expected_context_items": ["guest_id", "benefit_type"],
        "expected_escalations": 0,
        "expected_compromises": 0,
    },
    "tier_gated_denial": {
        "expected_intents": 2,
        "expected_tools": ["loyalty_allocate_benefit"],
        "expected_context_items": ["guest_id", "loyalty_tier", "checkout_request"],
        "expected_escalations": 1,
        "expected_compromises": 0,
    },
    "multi_intent_compromise": {
        "expected_intents": 3,
        "expected_tools": [
            "pms_update_reservation",
            "loyalty_allocate_benefit",
            "loyalty_allocate_benefit",
            "loyalty_allocate_benefit",
        ],
        "expected_context_items": [
            "checkout_time", "policy_ceiling", "suite_night_award", "wine_request",
        ],
        "expected_escalations": 1,
        "expected_compromises": 2,
    },
    "proactive_recovery": {
        "expected_intents": 2,
        "expected_tools": ["pms_query_rooms", "pms_reassign_room"],
        "expected_context_items": [
            "pet_friendly", "max_floor", "near_exit", "room_type", "flight_delay",
        ],
        "expected_escalations": 0,
        "expected_compromises": 1,
    },
    "vip_concierge_bundle": {
        "expected_intents": 5,
        "expected_tools": [
            "pms_update_reservation",
            "loyalty_allocate_benefit",
            "pms_query_rooms",
            "pms_reassign_room",
            "loyalty_allocate_benefit",
            "loyalty_allocate_benefit",
        ],
        "expected_context_items": [
            "checkout_time", "policy_ceiling", "pet_friendly", "suite_night_award",
            "breakfast",
        ],
        "expected_escalations": 0,
        "expected_compromises": 2,
    },
    "contradictory_intent": {
        "expected_intents": 2,
        "expected_tools": [],
        "expected_context_items": ["contradiction_detected"],
        "expected_escalations": 1,
        "expected_compromises": 0,
    },
    "ambiguous_escalation": {
        "expected_intents": 1,
        "expected_tools": [],
        "expected_context_items": ["ambiguity_detected"],
        "expected_escalations": 1,
        "expected_compromises": 0,
    },
    "mesh_favorable_baseline": {
        "expected_intents": 1,
        "expected_tools": [],
        "expected_context_items": ["informational_query"],
        "expected_escalations": 0,
        "expected_compromises": 0,
    },
}


def score_cso_result(scenario_name: str, cso_result: dict) -> dict[str, Any]:
    """Score a CSO pipeline result against scenario criteria."""
    criteria = SCENARIO_CRITERIA.get(scenario_name, {})
    if not criteria:
        return {"error": f"No criteria for scenario: {scenario_name}", "overall_score": 0}

    actions = cso_result.get("actions", [])
    escalations = cso_result.get("escalation_notes", [])
    sub_intents = cso_result.get("sub_intents", [])

    # Intents detected
    intents_detected = len(sub_intents)
    intents_expected = criteria.get("expected_intents", 0)

    # Tools called
    tools_called = [a.get("action") for a in actions]
    tools_expected = criteria.get("expected_tools", [])

    # Tool match: count how many expected tools were actually called
    tools_matched = 0
    remaining = list(tools_called)
    for t in tools_expected:
        if t in remaining:
            tools_matched += 1
            remaining.remove(t)

    # Escalation accuracy
    escalation_count = len(escalations)
    expected_escalations = criteria.get("expected_escalations", 0)
    escalation_accurate = escalation_count >= expected_escalations

    # Compromise detection
    compromises = [a for a in actions if a.get("is_compromise")]
    expected_compromises = criteria.get("expected_compromises", 0)
    compromise_score = min(len(compromises), expected_compromises) / max(expected_compromises, 1)

    # Policy violations (errors in results)
    policy_violations = [a for a in actions if a.get("result", {}).get("error")]

    # Overall score (weighted)
    intent_score = min(intents_detected, intents_expected) / max(intents_expected, 1)
    tool_score = tools_matched / max(len(tools_expected), 1)
    escalation_score = 1.0 if escalation_accurate else 0.5
    violation_penalty = len(policy_violations) * 0.1
    overall = max(0, (intent_score * 0.25 + tool_score * 0.35 +
                       escalation_score * 0.2 + compromise_score * 0.2) - violation_penalty)

    return {
        "scenario": scenario_name,
        "intents_detected": intents_detected,
        "intents_expected": intents_expected,
        "tools_called": tools_called,
        "tools_expected": tools_expected,
        "tools_matched": tools_matched,
        "escalation_count": escalation_count,
        "expected_escalations": expected_escalations,
        "escalation_accurate": escalation_accurate,
        "compromise_count": len(compromises),
        "expected_compromises": expected_compromises,
        "compromise_detection_score": round(compromise_score, 2),
        "policy_violations": len(policy_violations),
        "overall_score": round(overall * 100, 1),
        "model_config": MODEL_CONFIG["cso"],
    }


def score_mesh_result(scenario_name: str, mesh_dict: dict) -> dict[str, Any]:
    """Score a mesh pipeline result against scenario criteria."""
    criteria = SCENARIO_CRITERIA.get(scenario_name, {})
    if not criteria:
        return {"error": f"No criteria for scenario: {scenario_name}", "overall_score": 0}

    actions = mesh_dict.get("final_actions", [])
    escalations = mesh_dict.get("escalation_notes", [])
    degradation_chain = mesh_dict.get("degradation_chain", [])

    # Tools called
    tools_called = [a.get("action") for a in actions]
    tools_expected = criteria.get("expected_tools", [])
    tools_matched = 0
    remaining = list(tools_called)
    for t in tools_expected:
        if t in remaining:
            tools_matched += 1
            remaining.remove(t)

    # Escalation accuracy
    escalation_count = len(escalations)
    expected_escalations = criteria.get("expected_escalations", 0)
    escalation_accurate = escalation_count >= expected_escalations

    # Compromise detection (mesh rarely detects compromises)
    compromises = [a for a in actions if a.get("is_compromise")]
    expected_compromises = criteria.get("expected_compromises", 0)
    compromise_score = min(len(compromises), expected_compromises) / max(expected_compromises, 1)

    # Context items preserved vs dropped
    context_items_expected = len(criteria.get("expected_context_items", []))
    context_items_dropped = len(degradation_chain)  # proxy: more handoffs = more loss

    # Policy violations
    policy_violations = [a for a in actions if a.get("result", {}).get("error")]

    # Overall score
    tool_score = tools_matched / max(len(tools_expected), 1)
    escalation_score_val = 1.0 if escalation_accurate else 0.5
    violation_penalty = len(policy_violations) * 0.1
    overall = max(0, (tool_score * 0.4 + escalation_score_val * 0.25 +
                       compromise_score * 0.2 + 0.15) - violation_penalty)

    return {
        "scenario": scenario_name,
        "tools_called": tools_called,
        "tools_expected": tools_expected,
        "tools_matched": tools_matched,
        "escalation_count": escalation_count,
        "expected_escalations": expected_escalations,
        "escalation_accurate": escalation_accurate,
        "compromise_count": len(compromises),
        "expected_compromises": expected_compromises,
        "compromise_detection_score": round(compromise_score, 2),
        "context_items_expected": context_items_expected,
        "context_items_dropped": context_items_dropped,
        "policy_violations": len(policy_violations),
        "overall_score": round(overall * 100, 1),
        "model_config": MODEL_CONFIG["mesh"],
    }


def build_scorecard_comparison(
    scenario_name: str,
    cso_result: dict,
    mesh_dict: dict,
) -> dict[str, Any]:
    """Build a side-by-side quantitative scorecard for CSO vs mesh."""
    cso_score = score_cso_result(scenario_name, cso_result)
    mesh_score = score_mesh_result(scenario_name, mesh_dict)

    return {
        "cso_scorecard": cso_score,
        "mesh_scorecard": mesh_score,
        "cso_overall": cso_score.get("overall_score", 0),
        "mesh_overall": mesh_score.get("overall_score", 0),
        "delta": round(
            cso_score.get("overall_score", 0) - mesh_score.get("overall_score", 0), 1
        ),
    }
