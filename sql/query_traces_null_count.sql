SELECT
    COUNT(*) as total_rows,

    -- Basic fields
    SUM(CASE WHEN icao IS NULL THEN 1 ELSE 0 END) as icao_nulls,
    ROUND(100.0 * SUM(CASE WHEN icao IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as icao_null_pct,

    SUM(CASE WHEN actual_timestamp IS NULL THEN 1 ELSE 0 END) as actual_timestamp_nulls,
    ROUND(100.0 * SUM(CASE WHEN actual_timestamp IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as actual_timestamp_null_pct,

    SUM(CASE WHEN base_timestamp IS NULL THEN 1 ELSE 0 END) as base_timestamp_nulls,
    ROUND(100.0 * SUM(CASE WHEN base_timestamp IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as base_timestamp_null_pct,

    SUM(CASE WHEN seconds_offset IS NULL THEN 1 ELSE 0 END) as seconds_offset_nulls,
    ROUND(100.0 * SUM(CASE WHEN seconds_offset IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as seconds_offset_null_pct,

    -- Position data
    SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) as latitude_nulls,
    ROUND(100.0 * SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as latitude_null_pct,

    SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) as longitude_nulls,
    ROUND(100.0 * SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as longitude_null_pct,

    SUM(CASE WHEN altitude_baro IS NULL THEN 1 ELSE 0 END) as altitude_baro_nulls,
    ROUND(100.0 * SUM(CASE WHEN altitude_baro IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as altitude_baro_null_pct,

    SUM(CASE WHEN ground_speed IS NULL THEN 1 ELSE 0 END) as ground_speed_nulls,
    ROUND(100.0 * SUM(CASE WHEN ground_speed IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as ground_speed_null_pct,

    SUM(CASE WHEN track IS NULL THEN 1 ELSE 0 END) as track_nulls,
    ROUND(100.0 * SUM(CASE WHEN track IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as track_null_pct,

    -- Status flags and rates
    SUM(CASE WHEN flags IS NULL THEN 1 ELSE 0 END) as flags_nulls,
    ROUND(100.0 * SUM(CASE WHEN flags IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as flags_null_pct,

    SUM(CASE WHEN vertical_rate_baro IS NULL THEN 1 ELSE 0 END) as vertical_rate_baro_nulls,
    ROUND(100.0 * SUM(CASE WHEN vertical_rate_baro IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as vertical_rate_baro_null_pct,

    -- Geometric/GPS data
    SUM(CASE WHEN altitude_geom IS NULL THEN 1 ELSE 0 END) as altitude_geom_nulls,
    ROUND(100.0 * SUM(CASE WHEN altitude_geom IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as altitude_geom_null_pct,

    SUM(CASE WHEN vertical_rate_geom IS NULL THEN 1 ELSE 0 END) as vertical_rate_geom_nulls,
    ROUND(100.0 * SUM(CASE WHEN vertical_rate_geom IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as vertical_rate_geom_null_pct,

    -- Additional fields
    SUM(CASE WHEN indicated_airspeed IS NULL THEN 1 ELSE 0 END) as indicated_airspeed_nulls,
    ROUND(100.0 * SUM(CASE WHEN indicated_airspeed IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as indicated_airspeed_null_pct,

    SUM(CASE WHEN roll_angle IS NULL THEN 1 ELSE 0 END) as roll_angle_nulls,
    ROUND(100.0 * SUM(CASE WHEN roll_angle IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as roll_angle_null_pct,

    -- Source metadata
    SUM(CASE WHEN source_type IS NULL THEN 1 ELSE 0 END) as source_type_nulls,
    ROUND(100.0 * SUM(CASE WHEN source_type IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as source_type_null_pct,

    SUM(CASE WHEN source_file_path IS NULL THEN 1 ELSE 0 END) as source_file_path_nulls,
    ROUND(100.0 * SUM(CASE WHEN source_file_path IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as source_file_path_null_pct,

    -- Extra metadata
    SUM(CASE WHEN flight IS NULL THEN 1 ELSE 0 END) as flight_nulls,
    ROUND(100.0 * SUM(CASE WHEN flight IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as flight_null_pct,

    SUM(CASE WHEN squawk IS NULL THEN 1 ELSE 0 END) as squawk_nulls,
    ROUND(100.0 * SUM(CASE WHEN squawk IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as squawk_null_pct,

    SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END) as category_nulls,
    ROUND(100.0 * SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as category_null_pct,

    SUM(CASE WHEN nav_qnh IS NULL THEN 1 ELSE 0 END) as nav_qnh_nulls,
    ROUND(100.0 * SUM(CASE WHEN nav_qnh IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nav_qnh_null_pct,

    SUM(CASE WHEN nav_altitude_mcp IS NULL THEN 1 ELSE 0 END) as nav_altitude_mcp_nulls,
    ROUND(100.0 * SUM(CASE WHEN nav_altitude_mcp IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nav_altitude_mcp_null_pct,

    SUM(CASE WHEN nav_heading IS NULL THEN 1 ELSE 0 END) as nav_heading_nulls,
    ROUND(100.0 * SUM(CASE WHEN nav_heading IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nav_heading_null_pct,

    SUM(CASE WHEN emergency IS NULL THEN 1 ELSE 0 END) as emergency_nulls,
    ROUND(100.0 * SUM(CASE WHEN emergency IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as emergency_null_pct,

    -- Quality indicators
    SUM(CASE WHEN nic IS NULL THEN 1 ELSE 0 END) as nic_nulls,
    ROUND(100.0 * SUM(CASE WHEN nic IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nic_null_pct,

    SUM(CASE WHEN rc IS NULL THEN 1 ELSE 0 END) as rc_nulls,
    ROUND(100.0 * SUM(CASE WHEN rc IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as rc_null_pct,

    SUM(CASE WHEN version IS NULL THEN 1 ELSE 0 END) as version_nulls,
    ROUND(100.0 * SUM(CASE WHEN version IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as version_null_pct,

    SUM(CASE WHEN nic_baro IS NULL THEN 1 ELSE 0 END) as nic_baro_nulls,
    ROUND(100.0 * SUM(CASE WHEN nic_baro IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nic_baro_null_pct,

    SUM(CASE WHEN nac_p IS NULL THEN 1 ELSE 0 END) as nac_p_nulls,
    ROUND(100.0 * SUM(CASE WHEN nac_p IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nac_p_null_pct,

    SUM(CASE WHEN nac_v IS NULL THEN 1 ELSE 0 END) as nac_v_nulls,
    ROUND(100.0 * SUM(CASE WHEN nac_v IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as nac_v_null_pct,

    SUM(CASE WHEN sil IS NULL THEN 1 ELSE 0 END) as sil_nulls,
    ROUND(100.0 * SUM(CASE WHEN sil IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as sil_null_pct,

    SUM(CASE WHEN sil_type IS NULL THEN 1 ELSE 0 END) as sil_type_nulls,
    ROUND(100.0 * SUM(CASE WHEN sil_type IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as sil_type_null_pct,

    SUM(CASE WHEN gva IS NULL THEN 1 ELSE 0 END) as gva_nulls,
    ROUND(100.0 * SUM(CASE WHEN gva IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as gva_null_pct,

    SUM(CASE WHEN sda IS NULL THEN 1 ELSE 0 END) as sda_nulls,
    ROUND(100.0 * SUM(CASE WHEN sda IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as sda_null_pct

FROM postgres_db.public.aircraft_traces;