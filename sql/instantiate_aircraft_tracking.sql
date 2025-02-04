DROP TABLE IF EXISTS aircraft_tracking;

CREATE TABLE aircraft_tracking (
    -- Primary key columns
    snapshot_timestamp BIGINT,
    hex CHAR(7),
    -- Timestamp related fields
    actual_timestamp BIGINT NOT NULL,
    snapshot_messages BIGINT NOT NULL,
    seen DECIMAL(8,1),

    -- Aircraft identification
    flight VARCHAR(20),
    registration VARCHAR(20),
    type VARCHAR(4),
    aircraft_type VARCHAR(20),
    category VARCHAR(2),

    -- Position and movement
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    alt_baro INTEGER,
    alt_geom INTEGER,
    ground_speed DECIMAL(6,1),
    track DECIMAL(6,2),
    baro_rate INTEGER,

    -- Navigation and integrity
    nic INTEGER,
    rc INTEGER,
    seen_pos DECIMAL(10,3),
    version INTEGER,
    nav_modes TEXT[],
    nav_qnh DECIMAL(6,1),
    nav_altitude_mcp INTEGER,
    nav_heading DECIMAL(6,2),

    -- Communication
    squawk VARCHAR(4),
    emergency TEXT,

    -- Signal and message information
    messages BIGINT,
    rssi DECIMAL(8,1),

    -- Additional data arrays
    mlat TEXT[],
    tisb TEXT[],

    -- Primary key constraint
    PRIMARY KEY (snapshot_timestamp, hex)
);

-- Add comment descriptions to columns
COMMENT ON COLUMN aircraft_tracking.snapshot_timestamp IS 'Time the file was generated, in milliseconds since Unix epoch';
COMMENT ON COLUMN aircraft_tracking.hex IS 'The 24-bit ICAO identifier of the aircraft as 6 hex digits. May start with ~ for non-ICAO address';
COMMENT ON COLUMN aircraft_tracking.actual_timestamp IS 'Timestamp derived from snapshot_timestamp and seen';
COMMENT ON COLUMN aircraft_tracking.flight IS 'Callsign, flight name or aircraft registration as 8 chars';
COMMENT ON COLUMN aircraft_tracking.registration IS 'Aircraft registration pulled from database';
COMMENT ON COLUMN aircraft_tracking.type IS 'Aircraft type pulled from database';
COMMENT ON COLUMN aircraft_tracking.aircraft_type IS 'Type of underlying messages/best source (e.g., adsb_icao, mlat, tisb_icao)';
COMMENT ON COLUMN aircraft_tracking.alt_baro IS 'Aircraft barometric altitude in feet (or "ground")';
COMMENT ON COLUMN aircraft_tracking.alt_geom IS 'Geometric (GNSS/INS) altitude in feet referenced to WGS84 ellipsoid';
COMMENT ON COLUMN aircraft_tracking.ground_speed IS 'Ground speed in knots';
COMMENT ON COLUMN aircraft_tracking.track IS 'True track over ground in degrees (0-359)';
COMMENT ON COLUMN aircraft_tracking.baro_rate IS 'Rate of change of barometric altitude, feet/minute';
COMMENT ON COLUMN aircraft_tracking.squawk IS 'Mode A code (Squawk), encoded as 4 octal digits';
COMMENT ON COLUMN aircraft_tracking.category IS 'Emitter category to identify aircraft/vehicle classes (A0-D7)';
COMMENT ON COLUMN aircraft_tracking.latitude IS 'Aircraft position latitude in decimal degrees';
COMMENT ON COLUMN aircraft_tracking.longitude IS 'Aircraft position longitude in decimal degrees';
COMMENT ON COLUMN aircraft_tracking.nic IS 'Navigation Integrity Category';
COMMENT ON COLUMN aircraft_tracking.rc IS 'Radius of Containment in meters, derived from NIC & supplementary bits';
COMMENT ON COLUMN aircraft_tracking.seen_pos IS 'Seconds before "now" that position was last updated';
COMMENT ON COLUMN aircraft_tracking.version IS 'ADS-B Version Number (0, 1, 2; 3-7 reserved)';
COMMENT ON COLUMN aircraft_tracking.mlat IS 'List of fields derived from MLAT data';
COMMENT ON COLUMN aircraft_tracking.tisb IS 'List of fields derived from TIS-B data';
COMMENT ON COLUMN aircraft_tracking.messages IS 'Total number of Mode S messages received from this aircraft';
COMMENT ON COLUMN aircraft_tracking.seen IS 'Seconds before "now" that a message was last received';
COMMENT ON COLUMN aircraft_tracking.rssi IS 'Recent average signal power in dbFS (always negative)';
COMMENT ON COLUMN aircraft_tracking.nav_modes IS 'Set of engaged automation modes (autopilot, vnav, althold, approach, lnav, tcas)';
COMMENT ON COLUMN aircraft_tracking.nav_qnh IS 'Altimeter setting (QFE or QNH/QNE) in hPa';
COMMENT ON COLUMN aircraft_tracking.nav_altitude_mcp IS 'Selected altitude from Mode Control Panel/Flight Control Unit';
COMMENT ON COLUMN aircraft_tracking.nav_heading IS 'Selected heading (typically magnetic)';
COMMENT ON COLUMN aircraft_tracking.emergency IS 'ADS-B emergency/priority status';