DROP TABLE IF EXISTS autoencoder_training_unscaled;

CREATE TABLE autoencoder_training_unscaled (
   -- Identification
   segment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
   icao CHAR(7) NOT NULL,
   start_timestamp TIMESTAMPTZ NOT NULL,

   -- Segment metadata
   segment_duration INTERVAL NOT NULL,
   point_count INTEGER NOT NULL CHECK (point_count = 50),

   -- Raw features
   vertical_rates DOUBLE PRECISION[] NOT NULL,
   ground_speeds DOUBLE PRECISION[] NOT NULL,
   headings DOUBLE PRECISION[] NOT NULL,
   altitudes DOUBLE PRECISION[] NOT NULL,
   time_offsets INTEGER[] NOT NULL,

   -- Derived features
   vertical_accels DOUBLE PRECISION[] NOT NULL,
   ground_accels DOUBLE PRECISION[] NOT NULL,
   turn_rates DOUBLE PRECISION[] NOT NULL,
   climb_descent_accels DOUBLE PRECISION[] NOT NULL,

   -- Indexing
   UNIQUE (icao, start_timestamp),
   CHECK (
       array_length(vertical_rates, 1) = 50 AND
       array_length(ground_speeds, 1) = 50 AND
       array_length(headings, 1) = 50 AND
       array_length(altitudes, 1) = 50 AND
       array_length(time_offsets, 1) = 50 AND
       array_length(vertical_accels, 1) = 50 AND
       array_length(ground_accels, 1) = 50 AND
       array_length(turn_rates, 1) = 50 AND
       array_length(climb_descent_accels, 1) = 50
   )
);

COMMENT ON COLUMN autoencoder_training_unscaled.vertical_rates IS 'Direct copy of aircraft_trace_segments.vertical_rate_baro array, segmented into 50-point chunks';
COMMENT ON COLUMN autoencoder_training_unscaled.ground_speeds IS 'Direct copy of aircraft_trace_segments.ground_speed array, segmented into 50-point chunks';
COMMENT ON COLUMN autoencoder_training_unscaled.headings IS 'Direct copy of aircraft_trace_segments.track array (or nav_heading if track is null), segmented into 50-point chunks';
COMMENT ON COLUMN autoencoder_training_unscaled.altitudes IS 'Direct copy of aircraft_trace_segments.altitude_baro array, segmented into 50-point chunks';
COMMENT ON COLUMN autoencoder_training_unscaled.time_offsets IS 'Direct copy of aircraft_trace_segments.time_offsets array, segmented into 50-point chunks';

COMMENT ON COLUMN autoencoder_training_unscaled.vertical_accels IS 'Δ(vertical_rate_baro)/Δt where Δt is difference between consecutive time_offsets in microseconds. Units: feet/min²';
COMMENT ON COLUMN autoencoder_training_unscaled.ground_accels IS 'Δ(ground_speed)/Δt where Δt is difference between consecutive time_offsets in microseconds. Units: knots/second';
COMMENT ON COLUMN autoencoder_training_unscaled.turn_rates IS 'Δ(heading)/Δt where Δt is difference between consecutive time_offsets in microseconds, with special handling for 0°/360° wraparound. Units: degrees/second';
COMMENT ON COLUMN autoencoder_training_unscaled.climb_descent_accels IS 'Δ²(altitude_baro)/Δt² where Δt is difference between consecutive time_offsets in microseconds. Units: feet/second²';

COMMENT ON COLUMN autoencoder_training_unscaled.segment_duration IS 'time_offsets[49] - time_offsets[0] from the source data, converted to interval';
COMMENT ON COLUMN autoencoder_training_unscaled.point_count IS 'Always 50, represents number of consecutive points in each feature array';

-- Index for time-based queries
CREATE INDEX idx_autoencoder_training_unscaled_time
ON autoencoder_training_unscaled (start_timestamp);

-- Index for ICAO-based queries
CREATE INDEX idx_autoencoder_training_unscaled_icao
ON autoencoder_training_unscaled (icao);