"""
CSO Scenario Test Suite — 4 scenarios, ~20 structural assertions.

These tests validate structure, not text.  Claude's phrasing varies;
we check envelope status, action counts, tool names, and policy references.

Prerequisites:
    1. docker compose up -d --build   (with ANTHROPIC_API_KEY set)
    2. pytest tests/test_scenarios.py -v
"""

import pytest
from tests.conftest import run_comparison, run_scenario


# =========================================================================
# Test 1 (Easy): Single Benefit Allocation
# =========================================================================

class TestScenario1SingleBenefit:
    """
    Guest G-2002 (Gold) asks for complimentary breakfast.
    Expected: 1 action, status=Executable, no escalations.
    """

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.data = run_scenario(client, "single_benefit")

    def test_status_executable(self):
        assert self.data["status"] == "Executable"

    def test_single_action(self):
        actions = self.data["actions"]
        assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"

    def test_action_is_loyalty_allocate(self):
        action = self.data["actions"][0]
        assert action["action"] == "loyalty_allocate_benefit"

    def test_benefit_type_breakfast(self):
        action = self.data["actions"][0]
        result = action["result"]
        assert result.get("benefit_type") == "ComplimentaryBreakfast" or \
               result.get("status") == "allocated"

    def test_no_escalations(self):
        assert len(self.data["escalation_notes"]) == 0

    def test_no_compromises(self):
        for action in self.data["actions"]:
            assert action.get("is_compromise") is False


# =========================================================================
# Test 2 (Moderate-Easy): Tier-Gated Denial + Valid Benefit
# =========================================================================

class TestScenario2TierGatedDenial:
    """
    Guest G-2002 (Gold) asks for late checkout at 3 PM AND breakfast.
    Expected: Late checkout denied (Gold tier), breakfast allocated.
    """

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.data = run_scenario(client, "tier_gated_denial")

    def test_status_includes_escalation(self):
        assert self.data["status"] in (
            "Human_Escalation_Required",
            "Partial_Fulfillment",
        )

    def test_breakfast_allocated(self):
        """At least one action should be a breakfast allocation."""
        breakfast_found = False
        for action in self.data["actions"]:
            result = action.get("result", {})
            if (action.get("action") == "loyalty_allocate_benefit" and
                    result.get("benefit_type") == "ComplimentaryBreakfast"):
                breakfast_found = True
                break
        assert breakfast_found, "Breakfast should be allocated for Gold guest"

    def test_late_checkout_denied_or_escalated(self):
        """Late checkout should be denied or escalated for Gold tier."""
        # Check escalation notes
        has_checkout_escalation = any(
            "checkout" in note.lower() or "late" in note.lower()
            or "denied" in note.lower() or "tier" in note.lower()
            for note in self.data["escalation_notes"]
        )
        # Check contextual assertions for tier violation
        has_tier_assertion = any(
            "gold" in ca.get("assertion", "").lower()
            or "tier" in ca.get("assertion", "").lower()
            or "diamond" in ca.get("assertion", "").lower()
            for ca in self.data.get("contextual_assertions", [])
        )
        # Check sub_intents for tier_violation
        has_tier_violation = any(
            si.get("tier_violation") is not None
            for si in self.data.get("sub_intents", [])
        )
        assert has_checkout_escalation or has_tier_assertion or has_tier_violation, \
            "Late checkout should be denied for Gold tier"

    def test_mesh_annotation_present(self):
        assert len(self.data["mesh_annotation"]) > 0


# =========================================================================
# Test 3 (Moderate-Hard): Multi-Intent with Compromise + Escalation
# =========================================================================

class TestScenario3MultiIntentCompromise:
    """
    Guest G-1001 (Diamond) asks for 5 PM checkout, Suite Night Award,
    and a bottle of Chateau Margaux.
    Expected: Checkout clamped 5PM->4PM + drink voucher, SNA allocated,
              wine escalated, status=Human_Escalation_Required.
    """

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.data = run_scenario(client, "multi_intent_compromise")

    def test_status_human_escalation(self):
        assert self.data["status"] == "Human_Escalation_Required"

    def test_checkout_clamped_to_4pm(self):
        """A checkout update should clamp to 16:00."""
        for action in self.data["actions"]:
            if action.get("action") == "pms_update_reservation":
                result = action.get("result", {})
                new_checkout = result.get("new_checkout", "")
                assert "T16:00" in new_checkout, \
                    f"Checkout should be clamped to 16:00, got {new_checkout}"
                return
        pytest.fail("No pms_update_reservation action found")

    def test_drink_voucher_compromise(self):
        """A drink voucher should be issued as compensation."""
        voucher_found = False
        for action in self.data["actions"]:
            if action.get("action") == "loyalty_allocate_benefit":
                result = action.get("result", {})
                if result.get("benefit_type") == "ComplimentaryDrinkVoucher":
                    assert action.get("is_compromise") is True
                    voucher_found = True
                    break
        assert voucher_found, "Drink voucher compromise not found"

    def test_suite_night_award_allocated(self):
        """Suite Night Award should be allocated for Diamond guest."""
        sna_found = False
        for action in self.data["actions"]:
            if action.get("action") == "loyalty_allocate_benefit":
                result = action.get("result", {})
                if result.get("benefit_type") == "SuiteNightAward":
                    sna_found = True
                    break
        assert sna_found, "Suite Night Award not allocated"

    def test_wine_escalated(self):
        """Wine request should be escalated (no MCP tool)."""
        wine_escalated = any(
            "wine" in note.lower() or "margaux" in note.lower()
            or "provision" in note.lower() or "chateau" in note.lower()
            for note in self.data["escalation_notes"]
        )
        wine_in_assertions = any(
            "wine" in ca.get("assertion", "").lower()
            or "margaux" in ca.get("assertion", "").lower()
            or "provision" in ca.get("assertion", "").lower()
            for ca in self.data.get("contextual_assertions", [])
            if ca.get("requires_escalation")
        )
        assert wine_escalated or wine_in_assertions, \
            "Wine should be escalated — no MCP tool exists"

    def test_compromise_breadcrumbs(self):
        """At least one breadcrumb should reference compromise/ceiling."""
        compromise_crumbs = [
            b for b in self.data["breadcrumbs"]
            if "COMPROMISE" in b.get("result", "").upper()
            or "COMPROMISE" in b.get("policy_reference", "").upper()
            or "CEILING" in b.get("result", "").upper()
        ]
        assert len(compromise_crumbs) > 0, "No compromise breadcrumbs found"


# =========================================================================
# Test 4 (Tough — CSO Shines): Proactive Cross-Domain Recovery
# =========================================================================

class TestScenario4ProactiveRecovery:
    """
    Guest G-3003 (Titanium), 2 Cane Corso dogs (225 lbs total).
    Flight delay → arriving 1 AM. Current room 1415 (floor 14).
    Expected: Room query → reassign to room 101.
    """

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.data = run_scenario(client, "proactive_recovery")

    def test_status_partial_fulfillment(self):
        assert self.data["status"] in (
            "Partial_Fulfillment",
            "Executable",
        )

    def test_room_query_executed(self):
        """A pms_query_rooms action should have been executed."""
        query_found = any(
            a.get("action") == "pms_query_rooms"
            for a in self.data["actions"]
        )
        assert query_found, "Room query should have been executed"

    def test_reassigned_to_room_101(self):
        """Guest should be reassigned to room 101."""
        for action in self.data["actions"]:
            if action.get("action") == "pms_reassign_room":
                result = action.get("result", {})
                assert result.get("new_room") == "101", \
                    f"Expected room 101, got {result.get('new_room')}"
                return
        pytest.fail("No pms_reassign_room action found")

    def test_old_room_was_1415(self):
        """The old room should have been 1415."""
        for action in self.data["actions"]:
            if action.get("action") == "pms_reassign_room":
                result = action.get("result", {})
                assert result.get("old_room") == "1415", \
                    f"Expected old room 1415, got {result.get('old_room')}"
                return
        pytest.fail("No pms_reassign_room action found")

    def test_core_memory_has_pet_facts(self):
        """Core memory should contain pet information."""
        core = self.data.get("memory_snapshot", {}).get("core_block", [])
        pet_facts = [f for f in core if "pet" in f.get("fact", "").lower()
                     or "cane corso" in f.get("fact", "").lower()
                     or "dog" in str(f.get("tags", [])).lower()]
        assert len(pet_facts) > 0, "Core memory should have pet facts"

    def test_mesh_annotation_explains_telephone_game(self):
        annotation = self.data.get("mesh_annotation", "")
        assert "telephone" in annotation.lower() or "context" in annotation.lower(), \
            "Mesh annotation should explain the telephone game problem"


# =========================================================================
# Comparison Tests — CSO vs Mesh side-by-side
# =========================================================================

class TestComparison1SingleBenefit:
    """Control case: both CSO and mesh should produce identical results."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "single_benefit")

    def test_both_succeed(self):
        assert self.comp["comparison"]["both_succeed"] is True

    def test_same_action_counts(self):
        c = self.comp["comparison"]
        assert c["cso_action_count"] == c["mesh_action_count"]


class TestComparison2TierGatedDenial:
    """CSO preserves breakfast despite checkout denial; mesh may lose it."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "tier_gated_denial")

    def test_cso_outperforms_or_matches_mesh(self):
        c = self.comp["comparison"]
        assert c["cso_action_count"] >= c["mesh_action_count"]


class TestComparison3MultiIntent:
    """CSO issues drink voucher as compensation; mesh may not."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "multi_intent_compromise")

    def test_cso_has_voucher(self):
        vc = self.comp["comparison"]["voucher_comparison"]
        assert vc["cso_has_voucher"] is True


class TestComparison4ProactiveRecovery:
    """CSO assigns room 101; mesh assigns some room (may differ)."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "proactive_recovery")

    def test_cso_assigns_101(self):
        rc = self.comp["comparison"]["room_comparison"]
        assert rc is not None
        assert rc["cso_room"] == "101"


# =========================================================================
# Test 5 (Hardest): VIP Concierge Bundle
# =========================================================================

class TestScenario5VIPConciergeBundle:
    """
    Guest G-1001 (Diamond) at LHRW01: extend checkout to 5PM, move to
    ground-floor pet-friendly suite near exit, apply SNA, add breakfast.
    Expected: 6 actions (checkout clamped 4PM + drink voucher + room query
    + room 101 + SNA + breakfast).
    """

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.data = run_scenario(client, "vip_concierge_bundle")

    def test_status_partial_or_escalation(self):
        assert self.data["status"] in (
            "Partial_Fulfillment",
            "Human_Escalation_Required",
            "Executable",
        )

    def test_checkout_clamped_to_4pm(self):
        for action in self.data["actions"]:
            if action.get("action") == "pms_update_reservation":
                result = action.get("result", {})
                new_checkout = result.get("new_checkout", "")
                assert "T16:00" in new_checkout, \
                    f"Checkout should be clamped to 16:00, got {new_checkout}"
                return
        pytest.fail("No pms_update_reservation action found")

    def test_drink_voucher_compromise(self):
        voucher_found = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
            for a in self.data["actions"]
        )
        assert voucher_found, "Drink voucher should be issued as compensation"

    def test_suite_night_award(self):
        sna_found = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "SuiteNightAward"
            for a in self.data["actions"]
        )
        assert sna_found, "Suite Night Award should be allocated"

    def test_breakfast_allocated(self):
        breakfast_found = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryBreakfast"
            for a in self.data["actions"]
        )
        assert breakfast_found, "Complimentary breakfast should be allocated"

    def test_room_query_executed(self):
        query_found = any(
            a.get("action") == "pms_query_rooms"
            for a in self.data["actions"]
        )
        assert query_found, "Room query should have been executed"

    def test_reassigned_to_room_101(self):
        for action in self.data["actions"]:
            if action.get("action") == "pms_reassign_room":
                result = action.get("result", {})
                assert result.get("new_room") == "101", \
                    f"Expected room 101, got {result.get('new_room')}"
                return
        pytest.fail("No pms_reassign_room action found")


class TestComparison5VIPConciergeBundle:
    """CSO handles all sub-intents; mesh loses voucher and/or room accuracy."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "vip_concierge_bundle")

    def test_cso_has_voucher(self):
        vc = self.comp["comparison"]["voucher_comparison"]
        assert vc["cso_has_voucher"] is True

    def test_cso_assigns_101(self):
        rc = self.comp["comparison"]["room_comparison"]
        assert rc is not None
        assert rc["cso_room"] == "101"

    def test_cso_outperforms_or_matches_mesh(self):
        c = self.comp["comparison"]
        assert c["cso_action_count"] >= c["mesh_action_count"]
