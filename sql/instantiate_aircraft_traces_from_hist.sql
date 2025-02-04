DROP TABLE IF EXISTS aircraft_traces_from_hist;

CREATE TABLE aircraft_traces_from_hist(
    -- primary key columns
    time_stamp_start BIGINT,        -- first timestamp in the trace
    time_stamp_end BIGINT,          -- last timestamp in the trace
    hex CHAR(7),                    -- 24-bit ICAO identifier of the aircraft as 6 hex digits
    type VARCHAR(4),                -- aircraft type pulled from database
    category VARCHAR(2),            -- emitter category to identify aircraft/vehicle classes (A0-D7)
    point_count INTEGER,            -- number of points in the trace
    duration INTEGER,               -- duration in seconds

    -- data quality fields
    quality_passed BOOLEAN DEFAULT NULL,
    quality_flags TEXT[],

    -- will keep in order to potentially disambiguate traces
    flight VARCHAR(20),
    registration VARCHAR(20),

    -- fields to be used for validation
    squawk VARCHAR(4),              -- mode A code (Squawk), encoded as 4 octal digits
    emergency TEXT,                 -- ADS-B emergency/priority status
    nav_modes TEXT[],               -- set of engaged automation modes

    -- Position and movement arrays
    timestamp BIGINT[],             -- timestamps in milliseconds
    latitude DECIMAL(9,6)[],        -- decimal degrees
    longitude DECIMAL(9,6)[],       -- decimal degrees
    alt_baro INTEGER[],             -- feet
    ground_speed DECIMAL(6,1)[],    -- knots
    track DECIMAL(6,2)[],           -- degrees (0-359)
    baro_rate INTEGER[],            -- feet/minute
    nav_altitude_mcp INTEGER[],
    nav_heading DECIMAL(6,2)[],

    -- Primary key constraint
    PRIMARY KEY (time_stamp_start, hex),

    -- Array length constraint
    CHECK (
        array_length(timestamp, 1) = array_length(latitude, 1) AND
        array_length(timestamp, 1) = array_length(longitude, 1) AND
        array_length(timestamp, 1) = array_length(alt_baro, 1) AND
        array_length(timestamp, 1) = array_length(ground_speed, 1) AND
        array_length(timestamp, 1) = array_length(track, 1) AND
        array_length(timestamp, 1) = array_length(baro_rate, 1) AND
        array_length(timestamp, 1) = array_length(nav_altitude_mcp, 1) AND
        array_length(timestamp, 1) = array_length(nav_heading, 1)
    )
);