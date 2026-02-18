"""
Layer 4 — Memory Controller  (Mock Letta MemoryBlock Architecture)

Three-block memory model with decay policy:

  core_block      Permanent facts that never expire.
                  Guest tier, preferences, identity.
                  Analogous to Letta's "core memory".

  recall_block    Searchable tactical history of the current multi-domain
                  journey.  Each fact carries a TTL (default 48 h).
                  Compromise rationales, tool results, confirmed actions.

  transient_block Short-lived conversational state — sentiment, working
                  hypotheses, in-flight context.  Expires when the
                  originating intent is resolved (TTL ≈ minutes).

Every fact is wrapped in a MemoryFact envelope that carries:
  - fact:        The content string
  - tier:        core | tactical | transient
  - timestamp:   When the CSO recorded it
  - expires_at:  None (permanent) or UTC datetime
  - trace_id:    Which intent produced this fact
  - domain:      Which capability domain it relates to
  - tags:        Free-form labels for search

scrub_expired_context() MUST be called before every reasoning cycle
to prevent zombie context from influencing new decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from cso_poc.schemas import CanonicalIntentEnvelope, DecisionBreadcrumb


# ---------------------------------------------------------------------------
# Memory fact model
# ---------------------------------------------------------------------------

# Three tiers rather than flat storage because different facts have
# fundamentally different lifecycles:
#   - Core: "Guest is Diamond tier" never changes mid-session
#   - Tactical: "Checkout was clamped to 4PM" matters for 48h, then ages out
#   - Transient: "Guest sounds frustrated" is relevant only during active reasoning
# A flat store with no TTL would accumulate stale context that pollutes
# future reasoning passes — the zombie context problem.
class MemoryTier(str, Enum):
    CORE = "core"
    TACTICAL = "tactical"
    TRANSIENT = "transient"


@dataclass
class MemoryFact:
    """Metadata-wrapped fact stored in a memory block."""
    fact: str
    tier: MemoryTier
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None       # None = permanent
    trace_id: str = ""
    domain: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def ttl_seconds(self) -> float | None:
        """Seconds remaining, or None if permanent."""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(remaining, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact,
            "tier": self.tier.value,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "ttl_seconds": self.ttl_seconds,
            "trace_id": self.trace_id,
            "domain": self.domain,
            "tags": self.tags,
            "is_expired": self.is_expired,
        }


# Default TTLs: 48 hours for tactical facts bridges multiple guest interactions
# within a single stay.  10 minutes for transient state ensures sentiment and
# working hypotheses don't bleed across unrelated requests.
TACTICAL_TTL = timedelta(hours=48)
TRANSIENT_TTL = timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Turn record (kept for backward-compatible conversation_summary)
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """One conversational turn stored alongside the block facts."""
    turn_number: int
    guest_message: str
    envelope: CanonicalIntentEnvelope
    execution_results: list[dict]
    breadcrumbs: list[DecisionBreadcrumb]
    stored_at: datetime = field(default_factory=datetime.utcnow)

    def to_summary(self) -> dict[str, Any]:
        compromises = [
            {
                "action": a.tool_name,
                "rationale": a.compromise_rationale,
                "parameters": a.parameters,
            }
            for a in self.envelope.proposed_actions
            if a.is_compromise
        ]
        escalations = list(self.envelope.escalation_notes)
        actions_taken = [
            {
                "tool": r["action"],
                "result": r["result"],
                "is_compromise": r.get("is_compromise", False),
            }
            for r in self.execution_results
        ]
        return {
            "turn": self.turn_number,
            "trace_id": self.envelope.intent_id,
            "status": self.envelope.status.value,
            "objective": self.envelope.primary_objective,
            "domain_assertions": self.envelope.domain_assertions,
            "compromises": compromises,
            "escalations": escalations,
            "actions_taken": actions_taken,
        }


# ---------------------------------------------------------------------------
# Memory Controller
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Three-block memory with decay policy.

    Replaces the flat CSOMemoryBlock.  Provides the same recall interface
    plus tier-aware storage and expiration.
    """

    def __init__(self) -> None:
        self.core_block: list[MemoryFact] = []
        self.recall_block: list[MemoryFact] = []      # "tactical"
        self.transient_block: list[MemoryFact] = []

        self._entries: list[MemoryEntry] = []
        self._turn_counter: int = 0
        self._scrub_log: list[dict] = []   # audit trail of decayed facts

    def reset(self) -> None:
        """
        Clear all memory state.

        Uses .clear() on each list rather than reassigning to new lists.
        This is critical because scenarios.py imports the memory object
        by reference — reassignment would leave the imported reference
        pointing at the old (now-disconnected) lists.
        """
        self.core_block.clear()
        self.recall_block.clear()
        self.transient_block.clear()
        self._entries.clear()
        self._turn_counter = 0
        self._scrub_log.clear()

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    # ------------------------------------------------------------------
    # Decay policy
    # ------------------------------------------------------------------

    def scrub_expired_context(self) -> list[MemoryFact]:
        """
        Remove all expired facts from recall and transient blocks.

        MUST be called before every reasoning cycle to prevent zombie
        context — stale facts from prior intents that could mislead
        the current reasoning pass.  Every scrubbed fact is audit-logged
        so the dashboard can show exactly what was forgotten and when.
        """
        scrubbed: list[MemoryFact] = []

        live_recall = []
        for f in self.recall_block:
            if f.is_expired:
                scrubbed.append(f)
            else:
                live_recall.append(f)
        self.recall_block = live_recall

        live_transient = []
        for f in self.transient_block:
            if f.is_expired:
                scrubbed.append(f)
            else:
                live_transient.append(f)
        self.transient_block = live_transient

        # Audit log
        for f in scrubbed:
            self._scrub_log.append({
                "fact": f.fact,
                "tier": f.tier.value,
                "expired_at": f.expires_at.isoformat() if f.expires_at else None,
                "scrubbed_at": datetime.utcnow().isoformat(),
                "trace_id": f.trace_id,
            })

        return scrubbed

    def expire_transient_for_trace(self, trace_id: str) -> list[MemoryFact]:
        """
        Force-expire all transient facts tied to a specific intent.

        Called when an intent is fully resolved — the CSO "forgets"
        in-flight sentiment and working hypotheses.
        """
        expired: list[MemoryFact] = []
        remaining: list[MemoryFact] = []
        for f in self.transient_block:
            if f.trace_id == trace_id:
                f.expires_at = datetime.utcnow()
                expired.append(f)
            else:
                remaining.append(f)
        self.transient_block = remaining

        for f in expired:
            self._scrub_log.append({
                "fact": f.fact,
                "tier": "transient",
                "expired_at": f.expires_at.isoformat() if f.expires_at else None,
                "scrubbed_at": datetime.utcnow().isoformat(),
                "trace_id": f.trace_id,
                "reason": "intent_resolved",
            })

        return expired

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    def add_core_fact(
        self, fact: str, domain: str = "", tags: list[str] | None = None,
    ) -> MemoryFact:
        """Store a permanent fact in the core block."""
        mf = MemoryFact(
            fact=fact,
            tier=MemoryTier.CORE,
            expires_at=None,
            domain=domain,
            tags=tags or [],
        )
        # Deduplicate by fact string: core facts are populated on every
        # scenario run (since the guest profile is re-fetched each time).
        # Without deduplication, repeated runs would bloat the core block
        # with identical entries.
        if not any(existing.fact == fact for existing in self.core_block):
            self.core_block.append(mf)
        return mf

    def add_tactical_fact(
        self,
        fact: str,
        trace_id: str,
        domain: str = "",
        tags: list[str] | None = None,
        ttl: timedelta = TACTICAL_TTL,
    ) -> MemoryFact:
        """Store a journey fact in the recall (tactical) block."""
        mf = MemoryFact(
            fact=fact,
            tier=MemoryTier.TACTICAL,
            expires_at=datetime.utcnow() + ttl,
            trace_id=trace_id,
            domain=domain,
            tags=tags or [],
        )
        self.recall_block.append(mf)
        return mf

    def add_transient_fact(
        self,
        fact: str,
        trace_id: str,
        domain: str = "",
        tags: list[str] | None = None,
        ttl: timedelta = TRANSIENT_TTL,
    ) -> MemoryFact:
        """Store short-lived conversational state."""
        mf = MemoryFact(
            fact=fact,
            tier=MemoryTier.TRANSIENT,
            expires_at=datetime.utcnow() + ttl,
            trace_id=trace_id,
            domain=domain,
            tags=tags or [],
        )
        self.transient_block.append(mf)
        return mf

    # ------------------------------------------------------------------
    # Turn storage (envelope + results)
    # ------------------------------------------------------------------

    def store_intent(
        self,
        guest_message: str,
        envelope: CanonicalIntentEnvelope,
        execution_results: list[dict],
        breadcrumbs: list[DecisionBreadcrumb],
    ) -> MemoryEntry:
        self._turn_counter += 1
        entry = MemoryEntry(
            turn_number=self._turn_counter,
            guest_message=guest_message,
            envelope=envelope,
            execution_results=execution_results,
            breadcrumbs=breadcrumbs,
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Recall — block-aware search
    # ------------------------------------------------------------------

    def recall_by_trace(self, trace_id: str) -> MemoryEntry | None:
        for e in self._entries:
            if e.envelope.intent_id == trace_id:
                return e
        return None

    def recall_by_property(self, property_code: str) -> list[dict]:
        hits: list[dict] = []
        code_lower = property_code.lower()
        for entry in self._entries:
            for action in entry.envelope.proposed_actions:
                params_str = json.dumps(action.parameters, default=str).lower()
                if code_lower in params_str:
                    hits.append({
                        "trace_id": entry.envelope.intent_id,
                        "turn": entry.turn_number,
                        "tool": action.tool_name,
                        "parameters": action.parameters,
                        "is_compromise": action.is_compromise,
                        "compromise_rationale": action.compromise_rationale,
                    })
            for assertion in entry.envelope.domain_assertions:
                if code_lower in assertion.lower():
                    hits.append({
                        "trace_id": entry.envelope.intent_id,
                        "turn": entry.turn_number,
                        "assertion": assertion,
                    })
        return hits

    def search(self, query: str) -> list[dict]:
        """Keyword search across all three blocks."""
        hits: list[dict] = []
        q = query.lower()

        for block_name, block in [
            ("core", self.core_block),
            ("tactical", self.recall_block),
            ("transient", self.transient_block),
        ]:
            for f in block:
                if q in f.fact.lower() or any(q in t.lower() for t in f.tags):
                    hits.append({
                        "block": block_name,
                        "fact": f.fact,
                        "ttl_seconds": f.ttl_seconds,
                        "trace_id": f.trace_id,
                        "domain": f.domain,
                    })

        # Also search turn entries for backward compat
        for entry in self._entries:
            for a in entry.envelope.domain_assertions:
                if q in a.lower():
                    hits.append({
                        "block": "turn_assertion",
                        "fact": a,
                        "trace_id": entry.envelope.intent_id,
                        "turn": entry.turn_number,
                    })

        return hits

    def search_tactical(self, query: str) -> list[MemoryFact]:
        """Search only the tactical block — used for persistence checks."""
        q = query.lower()
        return [
            f for f in self.recall_block
            if q in f.fact.lower() or any(q in t.lower() for t in f.tags)
        ]

    # ------------------------------------------------------------------
    # Summaries for API / dashboard
    # ------------------------------------------------------------------

    def conversation_summary(self) -> list[dict]:
        return [e.to_summary() for e in self._entries]

    def vault_snapshot(self) -> dict[str, Any]:
        """
        Full memory vault state for the dashboard Memory Inspector.

        Returns all three blocks with TTL data plus the scrub audit log.
        """
        return {
            "core_block": [f.to_dict() for f in self.core_block],
            "recall_block": [f.to_dict() for f in self.recall_block],
            "transient_block": [f.to_dict() for f in self.transient_block],
            "scrub_log": self._scrub_log[-50:],  # last 50 entries
            "stats": {
                "core_count": len(self.core_block),
                "tactical_count": len(self.recall_block),
                "transient_count": len(self.transient_block),
                "total_scrubbed": len(self._scrub_log),
                "turns_stored": self._turn_counter,
            },
        }
