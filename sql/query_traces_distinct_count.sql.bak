SELECT
    COUNT(*) as total_rows,

    -- Basic fields
    COUNT(DISTINCT icao) as distinct_icao,
    ROUND(100.0 * COUNT(DISTINCT icao) / COUNT(*), 2) as icao_distinct_pct,

    COUNT(DISTINCT actual_timestamp) as distinct_actual_timestamp,
    ROUND(100.0 * COUNT(DISTINCT actual_timestamp) / COUNT(*), 2) as actual_timestamp_distinct_pct,

    COUNT(DISTINCT base_timestamp) as distinct_base_timestamp,
    ROUND(100.0 * COUNT(DISTINCT base_timestamp) / COUNT(*), 2) as base_timestamp_distinct_pct,

    COUNT(DISTINCT seconds_offset) as distinct_seconds_offset,
    ROUND(100.0 * COUNT(DISTINCT seconds_offset) / COUNT(*), 2) as seconds_offset_distinct_pct,

    -- Position data
    COUNT(DISTINCT latitude) as distinct_latitude,
    ROUND(100.0 * COUNT(DISTINCT latitude) / COUNT(*), 2) as latitude_distinct_pct,

    COUNT(DISTINCT longitude) as distinct_longitude,
    ROUND(100.0 * COUNT(DISTINCT longitude) / COUNT(*), 2) as longitude_distinct_pct,

    COUNT(DISTINCT altitude_baro) as distinct_altitude_baro,
    ROUND(100.0 * COUNT(DISTINCT altitude_baro) / COUNT(*), 2) as altitude_baro_distinct_pct,

    COUNT(DISTINCT ground_speed) as distinct_ground_speed,
    ROUND(100.0 * COUNT(DISTINCT ground_speed) / COUNT(*), 2) as ground_speed_distinct_pct,

    COUNT(DISTINCT track) as distinct_track,
    ROUND(100.0 * COUNT(DISTINCT track) / COUNT(*), 2) as track_distinct_pct,

    -- Status flags and rates
    COUNT(DISTINCT flags) as distinct_flags,
    ROUND(100.0 * COUNT(DISTINCT flags) / COUNT(*), 2) as flags_distinct_pct,

    COUNT(DISTINCT vertical_rate_baro) as distinct_vertical_rate_baro,
    ROUND(100.0 * COUNT(DISTINCT vertical_rate_baro) / COUNT(*), 2) as vertical_rate_baro_distinct_pct,

    -- Geometric/GPS data
    COUNT(DISTINCT altitude_geom) as distinct_altitude_geom,
    ROUND(100.0 * COUNT(DISTINCT altitude_geom) / COUNT(*), 2) as altitude_geom_distinct_pct,

    COUNT(DISTINCT vertical_rate_geom) as distinct_vertical_rate_geom,
    ROUND(100.0 * COUNT(DISTINCT vertical_rate_geom) / COUNT(*), 2) as vertical_rate_geom_distinct_pct,

    -- Additional fields
    COUNT(DISTINCT indicated_airspeed) as distinct_indicated_airspeed,
    ROUND(100.0 * COUNT(DISTINCT indicated_airspeed) / COUNT(*), 2) as indicated_airspeed_distinct_pct,

    COUNT(DISTINCT roll_angle) as distinct_roll_angle,
    ROUND(100.0 * COUNT(DISTINCT roll_angle) / COUNT(*), 2) as roll_angle_distinct_pct,

    -- Source metadata
    COUNT(DISTINCT source_type) as distinct_source_type,
    ROUND(100.0 * COUNT(DISTINCT source_type) / COUNT(*), 2) as source_type_distinct_pct,

    COUNT(DISTINCT source_file_path) as distinct_source_file_path,
    ROUND(100.0 * COUNT(DISTINCT source_file_path) / COUNT(*), 2) as source_file_path_distinct_pct,

    -- Extra metadata
    COUNT(DISTINCT flight) as distinct_flight,
    ROUND(100.0 * COUNT(DISTINCT flight) / COUNT(*), 2) as flight_distinct_pct,

    COUNT(DISTINCT squawk) as distinct_squawk,
    ROUND(100.0 * COUNT(DISTINCT squawk) / COUNT(*), 2) as squawk_distinct_pct,

    COUNT(DISTINCT category) as distinct_category,
    ROUND(100.0 * COUNT(DISTINCT category) / COUNT(*), 2) as category_distinct_pct,

    COUNT(DISTINCT nav_qnh) as distinct_nav_qnh,
    ROUND(100.0 * COUNT(DISTINCT nav_qnh) / COUNT(*), 2) as nav_qnh_distinct_pct,

    COUNT(DISTINCT nav_altitude_mcp) as distinct_nav_altitude_mcp,
    ROUND(100.0 * COUNT(DISTINCT nav_altitude_mcp) / COUNT(*), 2) as nav_altitude_mcp_distinct_pct,

    COUNT(DISTINCT nav_heading) as distinct_nav_heading,
    ROUND(100.0 * COUNT(DISTINCT nav_heading) / COUNT(*), 2) as nav_heading_distinct_pct,

    COUNT(DISTINCT emergency) as distinct_emergency,
    ROUND(100.0 * COUNT(DISTINCT emergency) / COUNT(*), 2) as emergency_distinct_pct,

    -- Quality indicators
    COUNT(DISTINCT nic) as distinct_nic,
    ROUND(100.0 * COUNT(DISTINCT nic) / COUNT(*), 2) as nic_distinct_pct,

    COUNT(DISTINCT rc) as distinct_rc,
    ROUND(100.0 * COUNT(DISTINCT rc) / COUNT(*), 2) as rc_distinct_pct,

    COUNT(DISTINCT version) as distinct_version,
    ROUND(100.0 * COUNT(DISTINCT version) / COUNT(*), 2) as version_distinct_pct,

    COUNT(DISTINCT nic_baro) as distinct_nic_baro,
    ROUND(100.0 * COUNT(DISTINCT nic_baro) / COUNT(*), 2) as nic_baro_distinct_pct,

    COUNT(DISTINCT nac_p) as distinct_nac_p,
    ROUND(100.0 * COUNT(DISTINCT nac_p) / COUNT(*), 2) as nac_p_distinct_pct,

    COUNT(DISTINCT nac_v) as distinct_nac_v,
    ROUND(100.0 * COUNT(DISTINCT nac_v) / COUNT(*), 2) as nac_v_distinct_pct,

    COUNT(DISTINCT sil) as distinct_sil,
    ROUND(100.0 * COUNT(DISTINCT sil) / COUNT(*), 2) as sil_distinct_pct,

    COUNT(DISTINCT sil_type) as distinct_sil_type,
    ROUND(100.0 * COUNT(DISTINCT sil_type) / COUNT(*), 2) as sil_type_distinct_pct,

    COUNT(DISTINCT gva) as distinct_gva,
    ROUND(100.0 * COUNT(DISTINCT gva) / COUNT(*), 2) as gva_distinct_pct,

    COUNT(DISTINCT sda) as distinct_sda,
    ROUND(100.0 * COUNT(DISTINCT sda) / COUNT(*), 2) as sda_distinct_pct

FROM postgres_db.public.aircraft_traces;