"""
Layer 5 — MCP Interconnect Server  (the "USB-C" of the architecture)

FastMCP server exposing deterministic tools and resources for the
Cognitive Singularity Orchestrator.  Every tool enforces business policy
*before* mutation and logs a Decision Breadcrumb (Layer 6) on every call.

Backing store: Postgres via asyncpg (Layer 7 — System of Record).
Transport:     SSE so the orchestrator can reach us over the Docker network.

No agent-to-agent communication exists here; the CSO reasons centrally
and this layer executes.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from cso_poc.db import close_pool, get_pool
from cso_poc.schemas import (
    BenefitType,
    DecisionBreadcrumb,
    LoyaltyAllocateBenefitParams,
    LoyaltyTier,
    PmsUpdateCheckinParams,
    PmsUpdateReservationParams,
)

# ---------------------------------------------------------------------------
# Logging setup (Decision Breadcrumbs go here)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
breadcrumb_log = logging.getLogger("cso.breadcrumbs")

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

DIAMOND_ONLY_BENEFITS = {BenefitType.SUITE_NIGHT_AWARD, BenefitType.LATE_CHECKOUT}
GENERAL_BENEFITS = {
    BenefitType.POINTS_BONUS,
    BenefitType.COMPLIMENTARY_BREAKFAST,
    BenefitType.COMPLIMENTARY_DRINK_VOUCHER,
}
MAX_CHECKOUT_HOUR = 16  # 4 PM


class PolicyViolation(Exception):
    """Raised when a business-rule check fails (maps to 403)."""

    def __init__(self, policy: str, detail: str) -> None:
        self.policy = policy
        self.detail = detail
        super().__init__(f"Policy [{policy}]: {detail}")


ELITE_TIERS = {LoyaltyTier.DIAMOND.value, LoyaltyTier.TITANIUM.value}


def _require_diamond(tier: str, guest_id: str, policy: str) -> None:
    if tier not in ELITE_TIERS:
        raise PolicyViolation(
            policy=policy,
            detail=f"Guest {guest_id} is {tier}, requires Diamond or Titanium.",
        )


# ---------------------------------------------------------------------------
# Layer 6 — Decision Breadcrumb decorator
# ---------------------------------------------------------------------------

def decision_breadcrumb(policy_reference: str):
    """
    Decorator that wraps every MCP tool call with a Decision Breadcrumb log.

    Format:  [TraceID] | [Policy_Reference] | [Action_Taken] | [Result]
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("trace_id", "no-trace")
            action = f"{fn.__name__}({json.dumps(kwargs, default=str)})"

            try:
                result = await fn(*args, **kwargs)
                crumb = DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference=policy_reference,
                    action_taken=action,
                    result="OK",
                )
                breadcrumb_log.info(crumb.format_log_line())
                return result

            except PolicyViolation as exc:
                crumb = DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference=exc.policy,
                    action_taken=action,
                    result=f"403 DENIED — {exc.detail}",
                )
                breadcrumb_log.warning(crumb.format_log_line())
                return {"error": 403, "policy": exc.policy, "detail": exc.detail}

            except Exception as exc:
                crumb = DecisionBreadcrumb(
                    trace_id=trace_id,
                    policy_reference=policy_reference,
                    action_taken=action,
                    result=f"500 ERROR — {exc}",
                )
                breadcrumb_log.error(crumb.format_log_line())
                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# FastMCP server definition
# ---------------------------------------------------------------------------

mcp = FastMCP("CognitiveSingularityOrchestrator", host="0.0.0.0")


# --- Resource: guest://profile/{id} --------------------------------------

@mcp.resource("guest://profile/{guest_id}")
async def get_guest_profile(guest_id: str) -> str:
    """Retrieve guest status, preferences, and all stay data."""
    pool = await get_pool()

    rows = await pool.fetch(
        """
        SELECT g.guest_id, g.name, g.loyalty_tier, g.preferences,
               s.reservation_id, s.property_code, s.room_number,
               s.check_in, s.check_out, s.notes
          FROM guests g
          LEFT JOIN stays s ON s.guest_id = g.guest_id
         WHERE g.guest_id = $1
         ORDER BY s.check_in
        """,
        guest_id,
    )
    if not rows:
        return json.dumps({"error": 404, "detail": f"Guest {guest_id} not found"})

    first = rows[0]
    profile: dict[str, Any] = {
        "guest_id": first["guest_id"],
        "name": first["name"],
        "loyalty_tier": first["loyalty_tier"],
        "preferences": json.loads(first["preferences"])
        if isinstance(first["preferences"], str)
        else first["preferences"],
    }

    stays = []
    for row in rows:
        if row["reservation_id"]:
            stays.append({
                "reservation_id": row["reservation_id"],
                "property_code": row["property_code"],
                "room_number": row["room_number"],
                "check_in": row["check_in"].isoformat(),
                "check_out": row["check_out"].isoformat(),
                "notes": row["notes"],
            })

    profile["stays"] = stays
    # Backward compat: current_stay = first stay
    if stays:
        profile["current_stay"] = stays[0]

    return json.dumps(profile)


# --- Tool: pms_update_reservation ----------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="POLICY-CHECKOUT-EXTENSION")
async def pms_update_reservation(
    res_id: str,
    checkout_time: str,
    notes: str = "",
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Deterministic update to the Property Management System.

    Policy enforced:
      - Only Diamond-tier guests may extend checkout past the standard
        11 AM window (up to 4 PM max).
      - Non-Diamond guests receive a 403 rejection.
    """
    parsed_dt = datetime.fromisoformat(checkout_time)
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    params = PmsUpdateReservationParams(
        res_id=res_id,
        checkout_time=parsed_dt,
        notes=notes,
    )
    pool = await get_pool()

    # Locate the reservation and guest tier
    row = await pool.fetchrow(
        """
        SELECT s.check_out, g.loyalty_tier, g.guest_id
          FROM stays s
          JOIN guests g ON g.guest_id = s.guest_id
         WHERE s.reservation_id = $1
        """,
        params.res_id,
    )
    if row is None:
        raise PolicyViolation(
            policy="POLICY-RES-EXISTS",
            detail=f"Reservation {params.res_id} not found.",
        )

    # Policy gate: late checkout restricted to Diamond
    current_checkout: datetime = row["check_out"]
    standard_checkout = current_checkout.replace(hour=11, minute=0, second=0)
    if params.checkout_time > standard_checkout:
        _require_diamond(row["loyalty_tier"], row["guest_id"], "POLICY-DIAMOND-LATE-CHECKOUT")

    if params.checkout_time.hour > MAX_CHECKOUT_HOUR:
        raise PolicyViolation(
            policy="POLICY-MAX-CHECKOUT-HOUR",
            detail=f"Checkout cannot exceed {MAX_CHECKOUT_HOUR}:00.",
        )

    # Apply mutation
    await pool.execute(
        """
        UPDATE stays
           SET check_out = $1, notes = $2
         WHERE reservation_id = $3
        """,
        params.checkout_time,
        params.notes,
        params.res_id,
    )

    return {
        "status": "confirmed",
        "reservation_id": params.res_id,
        "new_checkout": params.checkout_time.isoformat(),
    }


# --- Tool: pms_update_checkin --------------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="POLICY-CHECKIN-UPDATE")
async def pms_update_checkin(
    res_id: str,
    checkin_time: str,
    notes: str = "",
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Update the check-in time for a reservation.

    Policy enforced:
      - Check-in time must not be earlier than 14:00 (2 PM).
      - Reservation must exist.
    """
    parsed_ci = datetime.fromisoformat(checkin_time)
    if parsed_ci.tzinfo is None:
        parsed_ci = parsed_ci.replace(tzinfo=timezone.utc)
    params = PmsUpdateCheckinParams(
        res_id=res_id,
        checkin_time=parsed_ci,
        notes=notes,
    )
    pool = await get_pool()

    row = await pool.fetchrow(
        """
        SELECT s.check_in, g.loyalty_tier, g.guest_id
          FROM stays s
          JOIN guests g ON g.guest_id = s.guest_id
         WHERE s.reservation_id = $1
        """,
        params.res_id,
    )
    if row is None:
        raise PolicyViolation(
            policy="POLICY-RES-EXISTS",
            detail=f"Reservation {params.res_id} not found.",
        )

    # Apply mutation
    await pool.execute(
        """
        UPDATE stays
           SET check_in = $1, notes = $2
         WHERE reservation_id = $3
        """,
        params.checkin_time,
        params.notes,
        params.res_id,
    )

    return {
        "status": "confirmed",
        "reservation_id": params.res_id,
        "new_checkin": params.checkin_time.isoformat(),
    }


# --- Tool: loyalty_allocate_benefit --------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="POLICY-LOYALTY-BENEFIT")
async def loyalty_allocate_benefit(
    guest_id: str,
    benefit_type: str,
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Validate and apply a loyalty benefit (Suite Night Award, points, etc.).

    Policy enforced:
      - Suite Night Awards and Late Checkout are Diamond-exclusive.
      - Duplicate benefit allocations for the same stay are rejected.
    """
    params = LoyaltyAllocateBenefitParams(
        guest_id=guest_id,
        benefit_type=BenefitType(benefit_type),
    )
    pool = await get_pool()

    # Fetch guest
    guest_row = await pool.fetchrow(
        "SELECT loyalty_tier FROM guests WHERE guest_id = $1", params.guest_id,
    )
    if guest_row is None:
        raise PolicyViolation(
            policy="POLICY-GUEST-EXISTS",
            detail=f"Guest {params.guest_id} not found.",
        )

    # Policy gate: Diamond-only benefits
    if params.benefit_type in DIAMOND_ONLY_BENEFITS:
        _require_diamond(guest_row["loyalty_tier"], params.guest_id, "POLICY-DIAMOND-ONLY-BENEFIT")

    # Current stay
    stay_row = await pool.fetchrow(
        "SELECT reservation_id FROM stays WHERE guest_id = $1", params.guest_id,
    )
    stay_id = stay_row["reservation_id"] if stay_row else None

    # Idempotency guard via UNIQUE constraint
    try:
        await pool.execute(
            """
            INSERT INTO allocated_benefits (guest_id, reservation_id, benefit_type)
            VALUES ($1, $2, $3)
            """,
            params.guest_id,
            stay_id,
            params.benefit_type.value,
        )
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise PolicyViolation(
                policy="POLICY-NO-DUPLICATE-BENEFIT",
                detail=f"{params.benefit_type.value} already allocated for stay {stay_id}.",
            )
        raise

    return {
        "status": "allocated",
        "guest_id": params.guest_id,
        "benefit_type": params.benefit_type.value,
        "reservation_id": stay_id,
    }


# --- Tool: pms_query_rooms -------------------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="POLICY-ROOM-QUERY")
async def pms_query_rooms(
    property_code: str,
    pet_friendly: bool = False,
    max_floor: int = 100,
    near_exit: bool = False,
    room_type: str = "standard",
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Query available rooms in the hotel inventory.

    Filters by pet-friendliness, floor ceiling, exit proximity, and room type.
    Only returns rooms with status='available'.
    """
    pool = await get_pool()

    rows = await pool.fetch(
        """
        SELECT room_number, floor_number, pet_friendly, near_exit, room_type
          FROM room_inventory
         WHERE property_code = $1
           AND status = 'available'
           AND pet_friendly >= $2
           AND near_exit >= $3
           AND floor_number <= $4
           AND room_type = $5
         ORDER BY floor_number, room_number
        """,
        property_code,
        pet_friendly,
        near_exit,
        max_floor,
        room_type,
    )

    results = [
        {
            "room_number": r["room_number"],
            "floor_number": r["floor_number"],
            "pet_friendly": r["pet_friendly"],
            "near_exit": r["near_exit"],
            "room_type": r["room_type"],
        }
        for r in rows
    ]

    return {
        "status": "ok",
        "property_code": property_code,
        "available_rooms": results,
        "count": len(results),
    }


# --- Tool: pms_reassign_room -----------------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="POLICY-ROOM-REASSIGNMENT")
async def pms_reassign_room(
    res_id: str,
    new_room_number: str,
    reason: str,
    notes: str = "",
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Reassign a guest to a different room.

    Policy enforced:
      - Reservation must exist.
      - New room must be available in room_inventory.
      - Proactive reassignment requires Titanium tier.
    """
    pool = await get_pool()

    # Fetch current reservation and guest info
    res_row = await pool.fetchrow(
        """
        SELECT s.room_number AS old_room, s.property_code,
               g.loyalty_tier, g.guest_id
          FROM stays s
          JOIN guests g ON g.guest_id = s.guest_id
         WHERE s.reservation_id = $1
        """,
        res_id,
    )
    if res_row is None:
        raise PolicyViolation(
            policy="POLICY-RES-EXISTS",
            detail=f"Reservation {res_id} not found.",
        )

    # Policy: proactive reassignment requires Titanium
    if "proactive" in reason.lower() or "compromise" in reason.lower():
        if res_row["loyalty_tier"] != LoyaltyTier.TITANIUM.value:
            raise PolicyViolation(
                policy="POLICY-TITANIUM-PROACTIVE-REASSIGNMENT",
                detail=f"Proactive reassignment requires Titanium. "
                       f"Guest is {res_row['loyalty_tier']}.",
            )

    # Verify new room is available
    room_row = await pool.fetchrow(
        """
        SELECT status FROM room_inventory
         WHERE property_code = $1 AND room_number = $2
        """,
        res_row["property_code"],
        new_room_number,
    )
    if room_row is None or room_row["status"] != "available":
        raise PolicyViolation(
            policy="POLICY-ROOM-AVAILABLE",
            detail=f"Room {new_room_number} is not available "
                   f"at {res_row['property_code']}.",
        )

    # Atomic swap: update stay + room statuses
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE stays SET room_number = $1, notes = $2 "
                "WHERE reservation_id = $3",
                new_room_number, notes, res_id,
            )
            await conn.execute(
                "UPDATE room_inventory SET status = 'available' "
                "WHERE property_code = $1 AND room_number = $2",
                res_row["property_code"], res_row["old_room"],
            )
            await conn.execute(
                "UPDATE room_inventory SET status = 'occupied' "
                "WHERE property_code = $1 AND room_number = $2",
                res_row["property_code"], new_room_number,
            )

    return {
        "status": "confirmed",
        "reservation_id": res_id,
        "old_room": res_row["old_room"],
        "new_room": new_room_number,
        "reason": reason,
    }


# --- Tool: _admin_reset_db -------------------------------------------------

@mcp.tool()
@decision_breadcrumb(policy_reference="ADMIN-RESET")
async def _admin_reset_db(
    trace_id: str = "no-trace",
) -> dict[str, Any]:
    """
    Reset database to seed state.

    Truncates allocated_benefits, resets stays and room_inventory
    to their original seed values.  Used by test harness.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM allocated_benefits")
            await conn.execute("DELETE FROM stays")
            await conn.execute("DELETE FROM room_inventory")

            await conn.execute("""
                INSERT INTO stays (reservation_id, guest_id, property_code, room_number, check_in, check_out)
                VALUES
                    ('R-5001', 'G-1001', 'LHRW01', '1412', '2026-02-01 15:00:00+00', '2026-02-03 11:00:00+00'),
                    ('R-5002', 'G-2002', 'LHRW01', '803',  '2026-02-01 14:00:00+00', '2026-02-02 11:00:00+00'),
                    ('R-5003', 'G-1001', 'WVGB01', '601',  '2026-02-03 15:00:00+00', '2026-02-05 11:00:00+00'),
                    ('R-6001', 'G-3003', 'LHRW01', '1415', '2026-02-04 22:00:00+00', '2026-02-07 11:00:00+00')
                ON CONFLICT (reservation_id) DO UPDATE SET
                    guest_id = EXCLUDED.guest_id,
                    property_code = EXCLUDED.property_code,
                    room_number = EXCLUDED.room_number,
                    check_in = EXCLUDED.check_in,
                    check_out = EXCLUDED.check_out,
                    notes = ''
            """)

            await conn.execute("""
                INSERT INTO room_inventory (room_id, property_code, room_number, floor_number, pet_friendly, near_exit, room_type, status)
                VALUES
                    ('LHRW01-1415', 'LHRW01', '1415', 14, TRUE,  FALSE, 'suite',    'occupied'),
                    ('LHRW01-1412', 'LHRW01', '1412', 14, FALSE, FALSE, 'suite',    'occupied'),
                    ('LHRW01-101',  'LHRW01', '101',   1, TRUE,  TRUE,  'suite',    'available'),
                    ('LHRW01-102',  'LHRW01', '102',   1, TRUE,  TRUE,  'standard', 'available'),
                    ('LHRW01-803',  'LHRW01', '803',   8, FALSE, FALSE, 'standard', 'occupied'),
                    ('LHRW01-201',  'LHRW01', '201',   2, FALSE, TRUE,  'standard', 'available')
                ON CONFLICT (room_id) DO UPDATE SET
                    property_code = EXCLUDED.property_code,
                    room_number = EXCLUDED.room_number,
                    floor_number = EXCLUDED.floor_number,
                    pet_friendly = EXCLUDED.pet_friendly,
                    near_exit = EXCLUDED.near_exit,
                    room_type = EXCLUDED.room_type,
                    status = EXCLUDED.status
            """)

    return {"status": "reset", "detail": "Database restored to seed state"}


# ---------------------------------------------------------------------------
# Entrypoint — SSE transport for networked access
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse")
