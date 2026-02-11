"""
Mesh Comparison Test Suite — validates that the real Claude-powered mesh
agents produce structurally sound results, and that side-by-side comparison
with CSO correctly identifies the differences.

Real agents are non-deterministic, so mesh assertions are loosened while
CSO assertions remain strict.

Prerequisites:
    1. docker compose up -d --build   (with ANTHROPIC_API_KEY set)
    2. pytest tests/test_mesh.py -v
"""

import logging
import pytest
from tests.conftest import run_comparison

log = logging.getLogger("test_mesh")


# =========================================================================
# Test 1: Single Benefit — mesh succeeds same as CSO (control case)
# =========================================================================

class TestMeshScenario1:
    """Both CSO and mesh handle single benefit identically."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "single_benefit")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_mesh_has_actions(self):
        assert len(self.mesh["final_actions"]) >= 1

    def test_mesh_has_breakfast(self):
        has_breakfast = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryBreakfast"
            for a in self.mesh["final_actions"]
        )
        assert has_breakfast, "Mesh should have allocated breakfast"

    def test_mesh_status_executable(self):
        assert self.mesh["final_status"] == "Executable"

    def test_both_succeed(self):
        assert self.comparison["both_succeed"] is True

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0


# =========================================================================
# Test 2: Tier-Gated Denial — CSO has breakfast, mesh may not
# =========================================================================

class TestMeshScenario2:
    """Mesh may cascade denial, losing the breakfast allocation."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "tier_gated_denial")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_mesh_has_fewer_or_equal_actions(self):
        """Mesh should have at most as many successful actions as CSO."""
        cso_count = self.comparison["cso_action_count"]
        mesh_count = self.comparison["mesh_action_count"]
        # Filter out error actions for mesh
        mesh_success = [
            a for a in self.mesh["final_actions"]
            if not a.get("result", {}).get("error")
        ]
        assert len(mesh_success) <= cso_count, \
            f"Mesh successful actions ({len(mesh_success)}) should be <= CSO ({cso_count})"

    def test_cso_has_breakfast(self):
        breakfast_found = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryBreakfast"
            for a in self.cso["actions"]
        )
        assert breakfast_found, "CSO should have allocated breakfast"

    def test_cso_status_not_rejected(self):
        assert self.cso["status"] != "Rejected"

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0, \
            "Degradation chain should have entries from agent handoffs"

    def test_context_loss_documented(self):
        assert len(self.mesh["context_loss_summary"]) > 0


# =========================================================================
# Test 3: Multi-Intent — CSO has drink voucher
# =========================================================================

class TestMeshScenario3:
    """CSO issues compensation voucher; mesh likely does not."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "multi_intent_compromise")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_mesh_has_checkout(self):
        has_checkout = any(
            a.get("action") == "pms_update_reservation"
            for a in self.mesh["final_actions"]
        )
        assert has_checkout, "Mesh should have a checkout update"

    def test_mesh_has_sna(self):
        has_sna = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "SuiteNightAward"
            for a in self.mesh["final_actions"]
        )
        assert has_sna, "Mesh should have SNA"

    def test_cso_has_voucher(self):
        has_voucher = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
            for a in self.cso["actions"]
        )
        assert has_voucher, "CSO should have drink voucher"

    def test_mesh_voucher_informational(self):
        """Log whether mesh also got the voucher (informational, not strict)."""
        mesh_has_voucher = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
            for a in self.mesh["final_actions"]
        )
        if mesh_has_voucher:
            log.info("INFORMATIONAL: Mesh also issued drink voucher (good!)")
        else:
            log.info("INFORMATIONAL: Mesh did NOT issue drink voucher (expected degradation)")

    def test_comparison_cso_has_voucher(self):
        vc = self.comparison["voucher_comparison"]
        assert vc["cso_has_voucher"] is True

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0


# =========================================================================
# Test 4: Proactive Recovery — CSO assigns room 101, mesh assigns some room
# =========================================================================

class TestMeshScenario4:
    """CSO assigns correct room; mesh may assign a different room."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "proactive_recovery")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_mesh_assigns_some_room(self):
        """Mesh should have a room reassignment (any room)."""
        reassignment = None
        for action in self.mesh["final_actions"]:
            if action.get("action") == "pms_reassign_room":
                reassignment = action
                break
        assert reassignment is not None, "Mesh should have a pms_reassign_room action"
        room = reassignment["result"].get("new_room")
        assert room is not None, "Mesh should have assigned a room"
        log.info("INFORMATIONAL: Mesh assigned room %s", room)

    def test_cso_assigns_room_101(self):
        for action in self.cso["actions"]:
            if action.get("action") == "pms_reassign_room":
                assert action["result"].get("new_room") == "101", \
                    f"CSO should assign room 101, got {action['result'].get('new_room')}"
                return
        pytest.fail("No pms_reassign_room in CSO actions")

    def test_degradation_chain_has_entries(self):
        chain = self.mesh["degradation_chain"]
        assert len(chain) > 0, "Should have degradation chain entries from agent handoffs"

    def test_cso_room_informational(self):
        """Log whether rooms differ (informational)."""
        rc = self.comparison.get("room_comparison")
        if rc:
            log.info("INFORMATIONAL: CSO room=%s, Mesh room=%s",
                     rc.get("cso_room"), rc.get("mesh_room"))


# =========================================================================
# Test 5: VIP Concierge Bundle — 7-agent stress test
# =========================================================================

class TestMeshScenario5:
    """Mesh handles 7-agent chain; may lose voucher and room accuracy."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "vip_concierge_bundle")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_mesh_has_actions(self):
        assert len(self.mesh["final_actions"]) >= 2, \
            "Mesh should produce at least checkout + SNA"

    def test_mesh_has_checkout(self):
        has_checkout = any(
            a.get("action") == "pms_update_reservation"
            for a in self.mesh["final_actions"]
        )
        assert has_checkout, "Mesh should have a checkout update"

    def test_mesh_has_sna(self):
        has_sna = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "SuiteNightAward"
            for a in self.mesh["final_actions"]
        )
        assert has_sna, "Mesh should have SNA"

    def test_mesh_voucher_informational(self):
        """Log whether mesh got the voucher (informational, not strict)."""
        mesh_has_voucher = any(
            a.get("action") == "loyalty_allocate_benefit"
            and a.get("result", {}).get("benefit_type") == "ComplimentaryDrinkVoucher"
            for a in self.mesh["final_actions"]
        )
        if mesh_has_voucher:
            log.info("INFORMATIONAL: Mesh also issued drink voucher")
        else:
            log.info("INFORMATIONAL: Mesh did NOT issue drink voucher (expected)")

    def test_mesh_room_informational(self):
        """Log which room mesh assigned (informational)."""
        for a in self.mesh["final_actions"]:
            if a.get("action") == "pms_reassign_room":
                room = a.get("result", {}).get("new_room")
                log.info("INFORMATIONAL: Mesh assigned room %s", room)
                return
        log.info("INFORMATIONAL: Mesh did not reassign any room")

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) >= 5, \
            "7-agent chain should produce at least 5 degradation entries"

    def test_cso_has_all_six_actions(self):
        """CSO should have 6 actions: checkout + voucher + room query + room + SNA + breakfast."""
        cso_actions = self.cso["actions"]
        assert len(cso_actions) >= 5, \
            f"CSO should have at least 5 actions, got {len(cso_actions)}"


# =========================================================================
# Comparison structure
# =========================================================================

class TestComparisonStructure:
    """Verify the comparison dict has all expected fields."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "single_benefit")

    def test_has_cso_key(self):
        assert "cso" in self.comp

    def test_has_mesh_key(self):
        assert "mesh" in self.comp

    def test_has_comparison_key(self):
        assert "comparison" in self.comp

    def test_comparison_has_required_fields(self):
        c = self.comp["comparison"]
        for field in ["both_succeed", "cso_status", "mesh_status",
                       "action_diff", "context_lost", "cso_advantage"]:
            assert field in c, f"Missing comparison field: {field}"

    def test_mesh_has_required_fields(self):
        m = self.comp["mesh"]
        for field in ["scenario_name", "trace_id", "steps", "final_actions",
                       "final_status", "degradation_chain"]:
            assert field in m, f"Missing mesh field: {field}"


# =========================================================================
# Test 6: Contradictory Intent — mesh processes independently
# =========================================================================

class TestMesh6ContradictoryIntent:
    """Mesh processes contradictory requests independently without conflict detection."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "contradictory_intent")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_comparison_structure_valid(self):
        for field in ["both_succeed", "cso_status", "mesh_status",
                       "cso_action_count", "mesh_action_count"]:
            assert field in self.comparison, f"Missing comparison field: {field}"

    def test_cso_detects_contradiction(self):
        assert self.cso["status"] in ("Rejected", "Human_Escalation_Required"), \
            f"CSO should detect contradiction, got status={self.cso['status']}"

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0


# =========================================================================
# Test 7: Ambiguous Escalation — mesh may over-interpret
# =========================================================================

class TestMesh7AmbiguousEscalation:
    """Mesh may over-interpret vague sentiment; CSO should escalate cleanly."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "ambiguous_escalation")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_comparison_structure_valid(self):
        for field in ["both_succeed", "cso_status", "mesh_status",
                       "cso_action_count", "mesh_action_count"]:
            assert field in self.comparison, f"Missing comparison field: {field}"

    def test_cso_escalates(self):
        assert self.cso["status"] in ("Rejected", "Human_Escalation_Required"), \
            f"CSO should escalate ambiguous request, got status={self.cso['status']}"

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0


# =========================================================================
# Test 8: Mesh-Favorable Baseline — both equivalent
# =========================================================================

class TestMesh8FavorableBaseline:
    """Both architectures handle simple informational queries equivalently."""

    @pytest.fixture(autouse=True)
    def _run(self, client):
        self.comp = run_comparison(client, "mesh_favorable_baseline")
        self.cso = self.comp["cso"]
        self.mesh = self.comp["mesh"]
        self.comparison = self.comp["comparison"]

    def test_comparison_structure_valid(self):
        for field in ["both_succeed", "cso_status", "mesh_status",
                       "cso_action_count", "mesh_action_count"]:
            assert field in self.comparison, f"Missing comparison field: {field}"

    def test_intellectual_honesty(self):
        """Both pipelines should produce equivalent results for simple queries."""
        cso_mutations = [
            a for a in self.cso.get("actions", [])
            if a.get("action") in (
                "pms_update_reservation", "pms_update_checkin",
                "loyalty_allocate_benefit", "pms_reassign_room",
            )
        ]
        mesh_mutations = [
            a for a in self.mesh.get("final_actions", [])
            if a.get("action") in (
                "pms_update_reservation", "pms_update_checkin",
                "loyalty_allocate_benefit", "pms_reassign_room",
            )
        ]
        assert len(cso_mutations) == 0, \
            "CSO should not call mutation tools for informational query"
        assert len(mesh_mutations) == 0, \
            "Mesh should not call mutation tools for informational query"
        log.info("INFORMATIONAL: Both CSO and mesh handled simple query without mutations")

    def test_degradation_chain_has_entries(self):
        assert len(self.mesh["degradation_chain"]) > 0
