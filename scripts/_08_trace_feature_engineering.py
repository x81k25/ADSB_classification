# Standard library imports
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time
from typing import Dict, List, Any
import warnings

# Third party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from tqdm import tqdm

################################################################################
# initial parameters and setup
################################################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_PARAMS: Dict[str, str] = {
    'dbname': os.getenv('DB_NAME', ''),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', ''),
    'port': os.getenv('DB_PORT', '')
}

# Column type definitions
DECIMAL_COLUMNS: set = {
    'ground_speed',         # DECIMAL(6,1)[]
    'track'                # DECIMAL(6,2)[]
}

INTEGER_COLUMNS: set = {
    'time_offsets',        # INTEGER[]
    'vertical_rate_baro',  # INTEGER[]
    'altitude_baro'        # INTEGER[]
}

STRING_COLUMNS: set = {
    'icao'                 # CHAR(7)
}

TIMESTAMP_COLUMNS: set = {
    'start_timestamp'      # TIMESTAMPTZ
}

INTERVAL_COLUMNS: set = set()  # No interval columns in query

SINGULAR_COLUMNS: set = {
    'icao',               # CHAR(7)
    'start_timestamp'     # TIMESTAMPTZ
}

ARRAY_COLUMNS: set = {
    'time_offsets',       # INTEGER[]
    'vertical_rate_baro', # INTEGER[]
    'ground_speed',       # DECIMAL(6,1)[]
    'track',             # DECIMAL(6,2)[]
    'altitude_baro'      # INTEGER[]
}

# ignore depracation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

################################################################################
# global supporting functions
################################################################################

def get_db_connection() -> psycopg2.extensions.connection:
    """
    Create a new database connection.

    Returns:
        psycopg2.extensions.connection: Database connection object

    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        return psycopg2.connect(**DB_PARAMS)
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise


################################################################################
# extract
################################################################################

def get_distinct_icaos() -> List[str]:
    """
    Retrieve all distinct ICAO codes from aircraft_trace_segments table.

    Returns:
        List[str]: List of unique ICAO codes sorted alphabetically

    Raises:
        psycopg2.Error: If database query fails
    """
    query = "SELECT DISTINCT icao FROM aircraft_trace_segments ORDER BY icao;"
    icao_list = []

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(query)
            icao_list = [row[0] for row in cur.fetchall()]
            logger.debug(f"Retrieved {len(icao_list)} distinct ICAO codes")
        return icao_list
    except psycopg2.Error as e:
        logger.error(f"Failed to retrieve ICAO codes: {str(e)}")
        raise
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def get_flight_data(icao_list: List[str]) -> pd.DataFrame:
    """
    Retrieve flight data for specified ICAO codes.

    Args:
        icao_list: List of ICAO codes to query

    Returns:
        DataFrame containing flight data with standardized columns

    Raises:
        ValueError: If icao_list is empty
        psycopg2.Error: If database query fails
    """
    if not icao_list:
        raise ValueError("ICAO list cannot be empty")

    icao_params = ','.join(['%s'] * len(icao_list))
    query = """
        SELECT 
            icao,
            start_timestamp,
            time_offsets,
            vertical_rate_baro,
            ground_speed,
            track,
            altitude_baro
        FROM aircraft_trace_segments
        WHERE 
            icao IN ({}) AND
            vertical_rate_baro IS NOT NULL AND
            ground_speed IS NOT NULL AND
            track IS NOT NULL AND
            altitude_baro IS NOT NULL AND
            array_length(time_offsets, 1) >= 50
        ORDER BY icao, start_timestamp;
    """.format(icao_params)

    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, icao_list)
            df = pd.DataFrame(cur.fetchall())

            if df.empty:
                logger.warning("No data found for provided ICAO codes")
                return df

            # Standardize data types
            df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])

            for col in STRING_COLUMNS & set(df.columns):
                df[col] = df[col].astype(str)

            for col in ARRAY_COLUMNS & set(df.columns):
                if col in INTEGER_COLUMNS:
                    df[col] = df[col].apply(
                        lambda x: np.array(x, dtype=np.int64))
                elif col in DECIMAL_COLUMNS:
                    df[col] = df[col].apply(
                        lambda x: np.array(x, dtype=np.float64))

            logger.info(f"Retrieved {len(df)} rows of flight data")
            return df

    except psycopg2.Error as e:
        logger.error(f"Failed to retrieve flight data: {str(e)}")
        raise
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


################################################################################
# feature engineering
################################################################################

def validate_segment(segment: Dict[str, Any]) -> bool:
    """
    Validate a segment against physics thresholds.

    Args:
        segment: Dictionary containing segment data

    Returns:
        bool: True if segment is valid, False otherwise
    """
    PHYSICS_LIMITS = {
        'vertical_rate': (-12000, 12000),  # feet/minute
        'ground_speed': (0, 1000),  # knots
        'altitude': (-1000, 82000),  # feet
        'heading': (0, 360)  # degrees
    }

    for check, (min_val, max_val) in PHYSICS_LIMITS.items():
        if check == 'vertical_rate':
            violations = np.where((segment['vertical_rates'] < min_val) |
                                  (segment['vertical_rates'] > max_val))[0]
        elif check == 'ground_speed':
            violations = np.where((segment['ground_speeds'] < min_val) |
                                  (segment['ground_speeds'] > max_val))[0]
        elif check == 'altitude':
            violations = np.where((segment['altitudes'] < min_val) |
                                  (segment['altitudes'] > max_val))[0]
        elif check == 'heading':
            violations = np.where((segment['headings'] < min_val) |
                                  (segment['headings'] > max_val))[0]

        if len(violations) > 0:
            logger.debug(f"ICAO {segment['icao']}: {check} threshold violated")
            return False

    return True


def calculate_features(
    time_offsets: np.ndarray,
    vertical_rates: np.ndarray,
    ground_speeds: np.ndarray,
    headings: np.ndarray,
    altitudes: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate derived features from input arrays.
    All input arrays must be length 50.

    Args:
        time_offsets: Array of microsecond offsets (starting from 0)
        vertical_rates: Array of vertical rates (feet/minute)
        ground_speeds: Array of ground speeds (knots)
        headings: Array of track/heading values (degrees)
        altitudes: Array of barometric altitudes (feet)

    Returns:
        Dictionary containing calculated features:
        - vertical_accels (feet/min²)
        - ground_accels (knots/second)
        - turn_rates (degrees/second)
        - climb_descent_accels (feet/second²)
    """
    logger.debug("Starting feature calculations")

    # Convert time differences to seconds for calculations
    time_diffs = np.diff(time_offsets) / 1_000_000  # microseconds to seconds

    # Initialize arrays with nan to handle last points
    vertical_accels = np.full(50, np.nan)
    ground_accels = np.full(50, np.nan)
    turn_rates = np.full(50, np.nan)
    climb_descent_accels = np.full(50, np.nan)

    # Ground acceleration (knots/second)
    ground_accels[:-1] = np.diff(ground_speeds) / time_diffs
    ground_accels[-1] = ground_accels[-2]  # Copy last value

    # Vertical acceleration (feet/min²)
    vertical_accels[:-1] = np.diff(vertical_rates) / time_diffs
    vertical_accels[-1] = vertical_accels[-2]

    # Turn rate with 0/360 wraparound handling (degrees/second)
    heading_diffs = np.diff(headings)
    # Handle 0/360 wraparound
    heading_diffs = np.where(heading_diffs > 180, heading_diffs - 360,
                             np.where(heading_diffs < -180, heading_diffs + 360,
                                      heading_diffs))
    turn_rates[:-1] = heading_diffs / time_diffs
    turn_rates[-1] = turn_rates[-2]

    # Climb/descent acceleration (feet/second²)
    # Need to handle the last two points specially
    for i in range(len(altitudes) - 2):
        dt1 = time_diffs[i]
        dt2 = time_diffs[i + 1]

        # First derivative at two consecutive points
        first_deriv = (altitudes[i + 1] - altitudes[i]) / dt1
        second_deriv = (altitudes[i + 2] - altitudes[i + 1]) / dt2

        # Second derivative
        climb_descent_accels[i] = (second_deriv - first_deriv) / dt1

    # Fill last two points
    climb_descent_accels[-2:] = climb_descent_accels[-3]

    logger.debug("Completed feature calculations")

    return {
        'vertical_accels': vertical_accels,
        'ground_accels': ground_accels,
        'turn_rates': turn_rates,
        'climb_descent_accels': climb_descent_accels
    }


def generate_segments(
    icao: str,
    start_timestamp: pd.Timestamp,
    time_offsets: np.ndarray,
    vertical_rate_baro: np.ndarray,
    ground_speed: np.ndarray,
    track: np.ndarray,
    altitude_baro: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Generate 50-point segments from flight data arrays.

    Args:
        icao: Aircraft identifier
        start_timestamp: Base timestamp for the sequence
        time_offsets: Array of microsecond offsets from start_timestamp
        vertical_rate_baro: Array of vertical rates
        ground_speed: Array of ground speeds
        track: Array of track/heading values
        altitude_baro: Array of barometric altitudes

    Returns:
        List of dictionaries containing segment data
    """
    logger.debug(
        f"Starting segment generation for ICAO {icao} with {len(time_offsets)} points")

    segments = []
    total_points = len(time_offsets)

    # Validate input arrays have same length
    arrays = [time_offsets, vertical_rate_baro, ground_speed, track,
              altitude_baro]
    if not all(len(arr) == total_points for arr in arrays):
        logger.debug(f"Input arrays have mismatched lengths for ICAO {icao}")
        return []

    def create_segment(indices: np.ndarray) -> Dict[str, Any]:
        """Helper function to create a segment dictionary from array indices"""
        if len(indices) != 50:
            logger.debug(
                f"Skipping segment creation - length {len(indices)} != 50")
            return None

        segment_start_offset = time_offsets[indices[0]]
        segment_end_offset = time_offsets[indices[-1]]

        # Reset time offsets to start from 0 for this segment
        segment_time_offsets = time_offsets[indices] - segment_start_offset

        return {
            'icao': icao,
            'start_timestamp': start_timestamp + pd.Timedelta(
                microseconds=int(segment_start_offset)),
            'segment_duration': pd.Timedelta(
                microseconds=int(segment_end_offset - segment_start_offset)),
            'point_count': 50,
            'time_offsets': segment_time_offsets,
            'vertical_rates': vertical_rate_baro[indices],
            'ground_speeds': ground_speed[indices],
            'headings': track[indices],
            'altitudes': altitude_baro[indices],
        }

    # Find gaps (>10 seconds between points)
    time_diffs = np.diff(time_offsets)
    gap_indices = np.where(time_diffs > 10_000_000)[0]
    logger.debug(f"Found {len(gap_indices)} gaps in data")

    # Process main segments (1-50, 51-100, etc.)
    for start_idx in range(0, total_points - 49, 50):
        indices = np.arange(start_idx, min(start_idx + 50, total_points))
        if len(indices) == 50:
            # Check if this segment crosses any gaps
            gap_in_segment = any(g in indices[:-1] for g in gap_indices)
            if not gap_in_segment:
                segment = create_segment(indices)
                if segment:
                    segments.append(segment)
                    logger.debug(
                        f"Created main segment starting at index {start_idx}")

    # Process overlapping segments (25-75, 75-125, etc.)
    for start_idx in range(25, total_points - 49, 50):
        indices = np.arange(start_idx, min(start_idx + 50, total_points))
        if len(indices) == 50:
            gap_in_segment = any(g in indices[:-1] for g in gap_indices)
            if not gap_in_segment:
                segment = create_segment(indices)
                if segment:
                    segments.append(segment)
                    logger.debug(
                        f"Created overlapping segment starting at index {start_idx}")

    # Process gaps - create segments before and after each gap
    for gap_idx in gap_indices:
        # Segment before gap (if enough points)
        if gap_idx >= 49:
            before_indices = np.arange(gap_idx - 49, gap_idx + 1)
            segment = create_segment(before_indices)
            if segment:
                segments.append(segment)
                logger.debug(
                    f"Created pre-gap segment ending at index {gap_idx}")

        # Segment after gap (if enough points)
        if gap_idx + 50 < total_points:
            after_indices = np.arange(gap_idx + 1, gap_idx + 51)
            segment = create_segment(after_indices)
            if segment:
                segments.append(segment)
                logger.debug(
                    f"Created post-gap segment starting at index {gap_idx + 1}")

    # Handle end of sequence (if not already covered by main segments)
    if total_points >= 50:
        end_indices = np.arange(total_points - 50, total_points)
        last_regular_end = ((total_points - 50) // 50) * 50 + 49
        if end_indices[-1] > last_regular_end:
            gap_in_segment = any(g in end_indices[:-1] for g in gap_indices)
            if not gap_in_segment:
                segment = create_segment(end_indices)
                if segment:
                    segments.append(segment)
                    logger.debug(
                        f"Created end segment starting at index {total_points - 50}")

    logger.debug(f"Generated {len(segments)} total segments for ICAO {icao}")
    return segments


def process_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process flight data and generate segments for all ICAOs.

    Args:
        df: Input DataFrame with columns:
            - icao: str
            - start_timestamp: pd.Timestamp
            - time_offsets: np.ndarray
            - vertical_rate_baro: np.ndarray
            - ground_speed: np.ndarray
            - track: np.ndarray
            - altitude_baro: np.ndarray

    Returns:
        DataFrame with 50-point segments and derived features
    """
    logger.debug(f"Starting flight data processing for {len(df)} rows")
    all_segments = []

    for idx, row in df.iterrows():
        logger.debug(f"Processing row {idx} with ICAO {row['icao']}")
        segments = generate_segments(
            icao=row['icao'],
            start_timestamp=row['start_timestamp'],
            time_offsets=row['time_offsets'],
            vertical_rate_baro=row['vertical_rate_baro'],
            ground_speed=row['ground_speed'],
            track=row['track'],
            altitude_baro=row['altitude_baro']
        )
        all_segments.extend(segments)

    # Create DataFrame from segments
    result_df = pd.DataFrame(all_segments)
    initial_segment_count = len(result_df)

    # Calculate derived features for each segment
    derived_features = []
    for idx, row in result_df.iterrows():
        features = calculate_features(
            time_offsets=row['time_offsets'],
            vertical_rates=row['vertical_rates'],
            ground_speeds=row['ground_speeds'],
            headings=row['headings'],
            altitudes=row['altitudes']
        )
        derived_features.append(features)

    # Add calculated features to DataFrame
    for feature_name in ['vertical_accels', 'ground_accels', 'turn_rates',
                         'climb_descent_accels']:
        result_df[feature_name] = [features[feature_name] for features in
                                   derived_features]


    ## removing validiation for now, as it is being too extreme
    # # Validate segments and remove invalid ones
    # valid_segments = []
    # for idx, row in result_df.iterrows():
    #     if validate_segment(row):
    #         valid_segments.append(idx)
    #
    # result_df = result_df.loc[valid_segments].reset_index(drop=True)
    # rejected_count = initial_segment_count - len(result_df)
    #
    # logger.debug(
    #     f"Completed processing with {rejected_count} segments rejected due to physics violations")
    # logger.debug(f"Final segment count: {len(result_df)}")

    return result_df

################################################################################
# load
################################################################################

def insert_segments_to_db(df: pd.DataFrame) -> None:
    """
    Insert flight segments into database, updating on conflict.

    Args:
        df: DataFrame containing flight segments

    Raises:
        ValueError: If NULL values found or array lengths incorrect
        psycopg2.Error: If database operation fails
    """
    # Remove duplicates, keeping last occurrence
    df = df.drop_duplicates(subset=['icao', 'start_timestamp'], keep='last')
    logger.debug(f"Starting insert of {len(df)} segments after deduplication")

    # Validate no NULLs present
    null_columns = df.columns[df.isna().any()].tolist()
    if null_columns:
        err_msg = f"NULL values found in columns: {null_columns}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Validate array lengths
    array_columns = ['vertical_rates', 'ground_speeds', 'headings', 'altitudes',
                     'time_offsets', 'vertical_accels', 'ground_accels',
                     'turn_rates', 'climb_descent_accels']

    for col in array_columns:
        invalid_lengths = df[col].apply(len) != 50
        if invalid_lengths.any():
            err_msg = f"Invalid array lengths in column {col}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Convert numpy arrays to Python lists and ensure proper numeric types
            df_copy = df.copy()
            for col in array_columns:
                if col in ['time_offsets']:
                    # Convert int64 arrays to int32 for PostgreSQL integer array compatibility
                    df_copy[col] = df_copy[col].apply(
                        lambda x: x.astype(np.int32).tolist())
                else:
                    # Convert all other arrays to float64 for PostgreSQL double precision
                    df_copy[col] = df_copy[col].apply(
                        lambda x: x.astype(np.float64).tolist())

            # Prepare data as list of tuples in the correct order
            data = [
                (
                    row.icao,
                    row.start_timestamp,
                    row.segment_duration,
                    row.point_count,
                    row.vertical_rates,
                    row.ground_speeds,
                    row.headings,
                    row.altitudes,
                    row.time_offsets,
                    row.vertical_accels,
                    row.ground_accels,
                    row.turn_rates,
                    row.climb_descent_accels
                )
                for row in df_copy.itertuples()
            ]

            # SQL for insert with ON CONFLICT UPDATE
            sql = """
                INSERT INTO autoencoder_training_unscaled (
                    icao, start_timestamp, segment_duration, point_count,
                    vertical_rates, ground_speeds, headings, altitudes, time_offsets,
                    vertical_accels, ground_accels, turn_rates, climb_descent_accels
                )
                VALUES %s
                ON CONFLICT (icao, start_timestamp) DO UPDATE SET
                    segment_duration = EXCLUDED.segment_duration,
                    point_count = EXCLUDED.point_count,
                    vertical_rates = EXCLUDED.vertical_rates,
                    ground_speeds = EXCLUDED.ground_speeds,
                    headings = EXCLUDED.headings,
                    altitudes = EXCLUDED.altitudes,
                    time_offsets = EXCLUDED.time_offsets,
                    vertical_accels = EXCLUDED.vertical_accels,
                    ground_accels = EXCLUDED.ground_accels,
                    turn_rates = EXCLUDED.turn_rates,
                    climb_descent_accels = EXCLUDED.climb_descent_accels
            """

            # Execute batch insert
            logger.debug("Executing batch insert")
            execute_values(cur, sql, data)
            conn.commit()

            logger.info(f"Successfully processed {len(df)} segments")

    except psycopg2.Error as e:
        logger.error(f"Database operation failed: {str(e)}")
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")

################################################################################
# run
################################################################################

def run(
    batch_size: int = 50,
    test_mode: bool = False,
    parallel_workers: int = 4
) -> bool:
    """
    Main execution function for flight data processing.

    Args:
        batch_size: Number of ICAOs to process in each batch
        test_mode: If True, only process first batch
        parallel_workers: Number of parallel workers for data processing

    Returns:
        bool: True if execution successful, False otherwise
    """
    try:
        # Start timing total operation
        total_start_time = time.time()

        # Get all distinct ICAOs
        logger.info("Starting ICAO retrieval")
        try:
            all_icaos = get_distinct_icaos()
            total_icaos = len(all_icaos)
            logger.info(f"Retrieved {total_icaos} distinct ICAOs")
        except Exception as e:
            logger.error(f"Failed to retrieve ICAOs: {str(e)}")
            return False

        # Process in batches with progress tracking
        with tqdm(total=total_icaos, desc="Processing ICAOs") as pbar:
            for start_idx in range(0, total_icaos, batch_size):
                batch_start_time = time.time()

                try:
                    # Define current batch range
                    end_idx = min(start_idx + batch_size, total_icaos)
                    current_batch = all_icaos[start_idx:end_idx]
                    current_batch_size = len(current_batch)

                    logger.debug(
                        f"Processing batch {start_idx // batch_size + 1}: "
                        f"ICAOs {start_idx} to {end_idx}"
                    )

                    # Fetch flight data for current batch
                    flight_data = get_flight_data(current_batch)

                    if flight_data.empty:
                        logger.warning(
                            f"No valid flight data for batch {start_idx}-{end_idx}")
                        pbar.update(current_batch_size)
                        continue

                    # Process data with parallel workers
                    actual_workers = min(parallel_workers, len(flight_data))
                    if actual_workers != parallel_workers:
                        logger.debug(
                            f"Adjusted workers from {parallel_workers} to {actual_workers} "
                            "based on data size"
                        )

                    # Split data into chunks for parallel processing
                    chunks = np.array_split(flight_data, actual_workers)

                    logger.debug(
                        f"Processing with {actual_workers} parallel workers")
                    with ThreadPoolExecutor(
                        max_workers=actual_workers) as executor:
                        chunk_results = list(
                            executor.map(process_flight_data, chunks))

                    # Combine results from all workers
                    training_segments = pd.concat(chunk_results,
                                                  ignore_index=True)

                    if not training_segments.empty:
                        # Insert processed segments to database
                        insert_segments_to_db(training_segments)

                    # Update progress bar
                    pbar.update(current_batch_size)

                    # Log batch completion metrics
                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Completed batch {start_idx // batch_size + 1} "
                        f"({start_idx} to {end_idx}) in {batch_time:.2f}s\n"
                        f"Processed {len(training_segments)} segments from "
                        f"{current_batch_size} ICAOs"
                    )

                    # Handle test mode
                    if test_mode:
                        logger.info("Test mode - stopping after first batch")
                        break

                except Exception as e:
                    logger.error(
                        f"Failed processing batch {start_idx}-{end_idx}: {str(e)}"
                    )
                    if test_mode:
                        return False
                    continue  # Skip to next batch if not in test mode

        # Log final completion metrics
        total_time = time.time() - total_start_time
        logger.info(
            f"Processing completed successfully in {total_time:.2f}s\n"
            f"Average processing time: {total_time / total_icaos:.3f}s per ICAO"
        )
        return True

    except Exception as e:
        logger.error(f"Critical error in run function: {str(e)}")
        return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flight data processing script")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Number of ICAOs to process in each batch")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Run in test mode (process only first batch)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    success = run(
        batch_size=args.batch_size,
        test_mode=args.test,
        parallel_workers=args.workers
    )

    sys.exit(0 if success else 1)


################################################################################
# end of _08_trace_feature_engineering.py
################################################################################
