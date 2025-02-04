-- Set timezone to UTC for this session
SET timezone = 'UTC';

-- Drop existing table if it exists
DROP TABLE IF EXISTS aircraft_trace_segments;

-- Create the processed aircraft traces table
CREATE TABLE aircraft_trace_segments (
    -- Primary key field
    icao CHAR(7) NOT NULL,

    -- Timestamp and duration fields
    start_timestamp TIMESTAMPTZ NOT NULL,         -- Start time in UTC
    duration INTERVAL NOT NULL,                   -- Duration of trace
    time_offsets INTEGER[] NOT NULL,             -- Array of microsecond offsets

    -- Position and movement arrays (matching source datatypes)
    nav_heading DECIMAL(6,2)[],                  -- Array of headings
    longitude DECIMAL(9,6)[] NOT NULL,           -- Array of longitudes
    vertical_rate_baro INTEGER[],                -- Array of vertical rates
    latitude DECIMAL(9,6)[] NOT NULL,            -- Array of latitudes
    ground_speed DECIMAL(6,1)[],                 -- Array of ground speeds
    flags INTEGER[],                             -- Array of status flags
    altitude_baro INTEGER[],                     -- Array of barometric altitudes
    nav_altitude_mcp INTEGER[],                  -- Array of selected altitudes
    track DECIMAL(6,2)[],                        -- Array of tracks/headings

    -- Additional fields
    indicated_airspeed INTEGER[],                -- Array of indicated airspeeds
    roll_angle DECIMAL(6,2)[],                  -- Array of roll angles
    squawk VARCHAR(4)[],                        -- Array of squawk codes
    flight VARCHAR(20),                         -- Flight number/callsign
    category VARCHAR(2),                        -- Aircraft category
    emergency VARCHAR(20),                      -- Emergency status

    -- Composite primary key using icao and start time
    PRIMARY KEY (icao, start_timestamp)
);

-- Add indexes for common query patterns
CREATE INDEX idx_trace_segments_timestamp ON aircraft_trace_segments (start_timestamp);
CREATE INDEX idx_trace_segments_icao ON aircraft_trace_segments (icao);

-- Add table comment
COMMENT ON TABLE aircraft_trace_segments IS 'Processed aircraft trajectory traces with arrays of position data points';

-- Add column comments
COMMENT ON COLUMN aircraft_trace_segments.icao IS 'ICAO 24-bit address as 6 hex digits';
COMMENT ON COLUMN aircraft_trace_segments.start_timestamp IS 'Start time of the trace in UTC';
COMMENT ON COLUMN aircraft_trace_segments.duration IS 'Total duration of the trace (max 24 hours)';
COMMENT ON COLUMN aircraft_trace_segments.time_offsets IS 'Array of microsecond offsets from start_timestamp';
COMMENT ON COLUMN aircraft_trace_segments.longitude IS 'Array of longitude measurements in decimal degrees';
COMMENT ON COLUMN aircraft_trace_segments.latitude IS 'Array of latitude measurements in decimal degrees';

-- Ensure UTC timezone is enforced for this table
ALTER TABLE aircraft_trace_segments
  ALTER COLUMN start_timestamp
  SET DEFAULT CURRENT_TIMESTAMP AT TIME ZONE 'UTC';