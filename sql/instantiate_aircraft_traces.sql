-- Set timezone to UTC for this session
SET timezone = 'UTC';

-- Drop existing table if it exists
DROP TABLE IF EXISTS aircraft_traces;

-- Create the aircraft traces table
CREATE TABLE aircraft_traces (
    -- Primary key fields
    icao CHAR(7) NOT NULL,                        -- ICAO hex identifier of aircraft
    actual_timestamp TIMESTAMPTZ NOT NULL,        -- Computed timestamp in UTC
    base_timestamp BIGINT NOT NULL,               -- Original unix timestamp from file
    seconds_offset DECIMAL(10,2) NOT NULL,        -- Seconds after base timestamp

    -- Position data
    latitude DECIMAL(9,6),                        -- Latitude in decimal degrees
    longitude DECIMAL(9,6),                       -- Longitude in decimal degrees
    altitude_baro INTEGER,                        -- Barometric altitude in feet
    ground_speed DECIMAL(6,1),                    -- Ground speed in knots
    track DECIMAL(6,2),                           -- Track/heading in degrees

    -- Status flags and rates
    flags INTEGER,                                -- Bitfield flags for data quality/status
    vertical_rate_baro INTEGER,                   -- Barometric vertical rate in feet per minute

    -- Geometric/GPS data
    altitude_geom INTEGER,                        -- Geometric altitude in feet
    vertical_rate_geom INTEGER,                   -- Geometric vertical rate in feet per minute

    -- Additional fields (2022+ format)
    indicated_airspeed INTEGER,                   -- Indicated airspeed in knots
    roll_angle DECIMAL(6,2),                      -- Roll angle in degrees

    -- Source metadata
    source_type VARCHAR(20),                      -- Type/source of position data
    source_file_path TEXT,                        -- Path to source trace file

    -- Extra JSON metadata when available
    flight VARCHAR(20),                           -- Flight number/callsign
    squawk VARCHAR(4),                            -- Mode A squawk code
    category VARCHAR(2),                          -- Aircraft category
    nav_qnh DECIMAL(6,1),                         -- Altimeter setting (QNH)
    nav_altitude_mcp INTEGER,                     -- Selected altitude from MCP/FCU
    nav_heading DECIMAL(6,2),                     -- Selected heading
    emergency VARCHAR(20),                        -- Emergency status

    -- Quality indicators
    nic INTEGER,                                 -- Navigation Integrity Category
    rc INTEGER,                                  -- Radius of Containment
    version INTEGER,                             -- ADS-B Version
    nic_baro INTEGER,                            -- Barometric altitude integrity
    nac_p INTEGER,                               -- Navigation Accuracy for Position
    nac_v INTEGER,                               -- Navigation Accuracy for Velocity
    sil INTEGER,                                 -- Source Integrity Level
    sil_type VARCHAR(20),                        -- SIL reference type
    gva INTEGER,                                 -- Geometric Vertical Accuracy
    sda INTEGER,                                 -- System Design Assurance

    -- Composite primary key
    PRIMARY KEY (icao, actual_timestamp)
);

-- Add indexes for common query patterns
CREATE INDEX idx_traces_timestamp ON aircraft_traces (actual_timestamp);
CREATE INDEX idx_traces_icao ON aircraft_traces (icao);

-- Add table comment
COMMENT ON TABLE aircraft_traces IS 'Aircraft trajectory traces from ADS-B data with UTC timestamps and position information';

-- Add column comments
COMMENT ON COLUMN aircraft_traces.icao IS 'ICAO 24-bit address as 6 hex digits';
COMMENT ON COLUMN aircraft_traces.actual_timestamp IS 'Computed UTC timestamp from base_timestamp + seconds_offset';
COMMENT ON COLUMN aircraft_traces.base_timestamp IS 'Original unix timestamp from trace file';
COMMENT ON COLUMN aircraft_traces.seconds_offset IS 'Seconds after base timestamp for this position';
COMMENT ON COLUMN aircraft_traces.flags IS 'Bitfield: 1=stale position, 2=new leg, 4=geometric vertical rate, 8=geometric altitude';
COMMENT ON COLUMN aircraft_traces.source_type IS 'Type of position data (e.g., adsb_icao, mlat, tisb)';

-- Ensure UTC timezone is enforced for this table
ALTER TABLE aircraft_traces
  ALTER COLUMN actual_timestamp
  SET DEFAULT CURRENT_TIMESTAMP AT TIME ZONE 'UTC';