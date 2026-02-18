-- Layer 7 — System of Record (Mock PMS Database)
-- Seed schema and data for the hospitality POC.
--
-- Architectural Decision: Seed data covers a deliberate tier mix
--   G-1001 (Diamond): exercises policy ceiling compromises and premium benefits
--   G-2002 (Gold):    exercises tier-gate denials (Gold can't get late checkout)
--   G-3003 (Titanium): exercises proactive reassignment (Titanium-only policy)
--   This mix ensures every policy branch in the MCP gateway is exercised.
--
-- Architectural Decision: Room inventory designed for constraint-based queries
--   Room 101 (floor 1, pet-friendly, near exit, suite) is the only room that
--   satisfies all constraints for proactive recovery.  Room 201 (floor 2, not
--   pet-friendly) is a trap: mesh agents that drop the pet-friendly constraint
--   may assign it incorrectly, proving the context degradation thesis.

CREATE TABLE IF NOT EXISTS guests (
    guest_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL CHECK (loyalty_tier IN ('Member','Silver','Gold','Platinum','Diamond','Titanium')),
    preferences JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS stays (
    reservation_id TEXT PRIMARY KEY,
    guest_id       TEXT NOT NULL REFERENCES guests(guest_id),
    property_code  TEXT NOT NULL,
    room_number    TEXT NOT NULL,
    check_in       TIMESTAMPTZ NOT NULL,
    check_out      TIMESTAMPTZ NOT NULL,
    notes          TEXT NOT NULL DEFAULT ''
);

-- The UNIQUE constraint on (guest_id, reservation_id, benefit_type) provides
-- database-level idempotency: allocating the same benefit twice for the same
-- stay is rejected, regardless of whether the orchestrator or mesh makes the
-- duplicate call.  This is the last line of defense against double-allocation.
CREATE TABLE IF NOT EXISTS allocated_benefits (
    id             SERIAL PRIMARY KEY,
    guest_id       TEXT NOT NULL REFERENCES guests(guest_id),
    reservation_id TEXT REFERENCES stays(reservation_id),
    benefit_type   TEXT NOT NULL,
    allocated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (guest_id, reservation_id, benefit_type)
);

CREATE TABLE IF NOT EXISTS room_inventory (
    room_id        TEXT PRIMARY KEY,
    property_code  TEXT NOT NULL,
    room_number    TEXT NOT NULL,
    floor_number   INTEGER NOT NULL,
    pet_friendly   BOOLEAN NOT NULL DEFAULT FALSE,
    near_exit      BOOLEAN NOT NULL DEFAULT FALSE,
    room_type      TEXT NOT NULL DEFAULT 'standard',
    status         TEXT NOT NULL DEFAULT 'available'
                   CHECK (status IN ('available','occupied','blocked')),
    UNIQUE (property_code, room_number)
);

-- Seed data
INSERT INTO guests (guest_id, name, loyalty_tier, preferences) VALUES
    ('G-1001', 'Alexandra Mercer', 'Diamond',  '{"room_type":"suite","pillow":"firm","minibar":"stocked"}'),
    ('G-2002', 'Jordan Whitfield', 'Gold',     '{"room_type":"standard","pillow":"soft"}'),
    ('G-3003', 'Marcus Wolfe',     'Titanium', '{"room_type":"suite","pets":[{"breed":"Cane Corso mix","name":"Atlas","weight_lbs":110},{"breed":"Cane Corso mix","name":"Zeus","weight_lbs":115}]}')
ON CONFLICT (guest_id) DO NOTHING;

INSERT INTO stays (reservation_id, guest_id, property_code, room_number, check_in, check_out) VALUES
    ('R-5001', 'G-1001', 'LHRW01', '1412', '2026-02-01 15:00:00+00', '2026-02-03 11:00:00+00'),
    ('R-5002', 'G-2002', 'LHRW01', '803',  '2026-02-01 14:00:00+00', '2026-02-02 11:00:00+00'),
    ('R-5003', 'G-1001', 'WVGB01', '601',  '2026-02-03 15:00:00+00', '2026-02-05 11:00:00+00'),
    ('R-6001', 'G-3003', 'LHRW01', '1415', '2026-02-04 22:00:00+00', '2026-02-07 11:00:00+00')
ON CONFLICT (reservation_id) DO NOTHING;

INSERT INTO room_inventory (room_id, property_code, room_number, floor_number, pet_friendly, near_exit, room_type, status) VALUES
    ('LHRW01-1415', 'LHRW01', '1415', 14, TRUE,  FALSE, 'suite',    'occupied'),
    ('LHRW01-1412', 'LHRW01', '1412', 14, FALSE, FALSE, 'suite',    'occupied'),
    ('LHRW01-101',  'LHRW01', '101',   1, TRUE,  TRUE,  'suite',    'available'),
    ('LHRW01-102',  'LHRW01', '102',   1, TRUE,  TRUE,  'standard', 'available'),
    ('LHRW01-803',  'LHRW01', '803',   8, FALSE, FALSE, 'standard', 'occupied'),
    ('LHRW01-201',  'LHRW01', '201',   2, FALSE, TRUE,  'standard', 'available')
ON CONFLICT (room_id) DO NOTHING;
