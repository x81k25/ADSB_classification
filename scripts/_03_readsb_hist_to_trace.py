# Standard library imports
import os
import logging
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

# Third party imports
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from tqdm import tqdm

################################################################################
# initial parameters and setup
################################################################################

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


################################################################################
# global supporting functions
################################################################################

def get_pg_connection_string() -> str:
    """
    Create PostgreSQL connection string from environment variables.

    Returns:
        str: Formatted PostgreSQL connection string

    Raises:
        ValueError: If required environment variables are missing
    """

    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }

    return f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"


################################################################################
# data extraction
################################################################################

def get_time_range(conn: psycopg2.extensions.connection) -> Tuple[int, int]:
    """
    Get the time range for aircraft tracking data in UTC.

    Args:
        conn: PostgreSQL connection object

    Returns:
        Tuple[int, int]: Start and end timestamps in UTC milliseconds

    Raises:
        psycopg2.Error: If database query fails
    """
    query = """
        SELECT 
            MIN(to_timestamp(actual_timestamp) AT TIME ZONE 'UTC'),
            MAX(to_timestamp(actual_timestamp) AT TIME ZONE 'UTC')
        FROM aircraft_tracking"""

    logger.debug(query)

    with conn.cursor() as cur:
        cur.execute(query)
        start_dt, end_dt = cur.fetchone()
        logger.debug(f"Raw datetime range: ({start_dt}, {end_dt})")

        # Convert datetime objects to milliseconds since epoch
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        return start_ms, end_ms


def fetch_chunk_data(conn: psycopg2.extensions.connection, start_time: int,
                     end_time: int) -> pd.DataFrame:
    """
    Fetch a chunk of aircraft tracking data.

    Args:
        conn: PostgreSQL connection object
        start_time: Start timestamp for the chunk
        end_time: End timestamp for the chunk

    Returns:
        pd.DataFrame: DataFrame containing the chunk data

    Raises:
        psycopg2.Error: If database query fails
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT *
            FROM aircraft_tracking
            WHERE actual_timestamp >= %s
              AND actual_timestamp < %s
            ORDER BY hex, actual_timestamp
        """, (start_time, end_time))

        colnames = [desc[0] for desc in cur.description]
        data = cur.fetchall()

    return pd.DataFrame(data, columns=colnames)


################################################################################
# data transformation
################################################################################

def find_trace_boundaries(group: pd.DataFrame,
                          max_gap_seconds: int) -> pd.Series:
    """
    Identify trace boundaries based on time gaps.

    Args:
        group: DataFrame group containing aircraft data
        max_gap_seconds: Maximum allowed gap between points

    Returns:
        pd.Series: Series containing trace boundary markers
    """
    time_diffs = group['actual_timestamp'].diff().astype('float64') / 1000
    new_trace = time_diffs > max_gap_seconds
    return new_trace.cumsum()


def prepare_trace_arrays(trace_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert trace DataFrame into arrays for PostgreSQL.

    Args:
        trace_df: DataFrame containing single trace data

    Returns:
        Dict[str, Any]: Dictionary containing formatted trace data
    """
    return {
        'time_stamp_start': int(trace_df['actual_timestamp'].min()),
        'time_stamp_end': int(trace_df['actual_timestamp'].max()),
        'hex': trace_df['hex'].iloc[0],
        'type': trace_df['type'].iloc[0],
        'category': trace_df['category'].iloc[0],
        'point_count': int(len(trace_df)),
        'duration': int((trace_df['actual_timestamp'].max() - trace_df[
            'actual_timestamp'].min()) / 1000),
        'quality_passed': None,
        'quality_flags': None,
        'flight': trace_df['flight'].iloc[0],
        'registration': trace_df['registration'].iloc[0],
        'squawk': trace_df['squawk'].iloc[0],
        'emergency': trace_df['emergency'].iloc[0],
        'nav_modes': trace_df['nav_modes'].iloc[0],
        'timestamp': [int(ts) for ts in trace_df['actual_timestamp'].tolist()],
        'latitude': trace_df['latitude'].tolist(),
        'longitude': trace_df['longitude'].tolist(),
        'alt_baro': [int(x) if pd.notna(x) else None for x in
                     trace_df['alt_baro']],
        'ground_speed': trace_df['ground_speed'].tolist(),
        'track': trace_df['track'].tolist(),
        'baro_rate': [int(x) if pd.notna(x) else None for x in
                      trace_df['baro_rate']],
        'nav_altitude_mcp': [int(x) if pd.notna(x) else None for x in
                             trace_df['nav_altitude_mcp']],
        'nav_heading': trace_df['nav_heading'].tolist()
    }


################################################################################
# data loading
################################################################################

def insert_traces(conn: psycopg2.extensions.connection,
                  traces: List[Dict[str, Any]]) -> None:
    """
    Bulk insert traces into the database.

    Args:
        conn: PostgreSQL connection object
        traces: List of trace dictionaries to insert

    Raises:
        psycopg2.Error: If database insertion fails
    """
    if not traces:
        return

    with conn.cursor() as cur:
        args_str = ','.join(
            cur.mogrify(
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    trace['time_stamp_start'], trace['time_stamp_end'],
                    trace['hex'], trace['type'], trace['category'],
                    trace['point_count'], trace['duration'],
                    trace['quality_passed'], trace['quality_flags'],
                    trace['flight'], trace['registration'], trace['squawk'],
                    trace['emergency'], trace['nav_modes'], trace['timestamp'],
                    trace['latitude'], trace['longitude'], trace['alt_baro'],
                    trace['ground_speed'], trace['track'], trace['baro_rate'],
                    trace['nav_altitude_mcp'], trace['nav_heading']
                )
            ).decode('utf-8')
            for trace in traces
        )

        cur.execute(f"""
            INSERT INTO aircraft_traces_from_hist VALUES {args_str}
            ON CONFLICT (time_stamp_start, hex) DO NOTHING
        """)


################################################################################
# run function
################################################################################

def run(max_gap_seconds: int = 600, chunk_size_hours: int = 1,
        test_mode: bool = False) -> bool:
    """
    Main execution function for creating aircraft traces from historical data.

    Args:
        max_gap_seconds: Maximum allowed gap between points in a trace
        chunk_size_hours: Size of time chunks to process in hours
        test_mode: If True, only processes one chunk of data. Defaults to False.

    Returns:
        bool: True if execution was successful, False otherwise

    Raises:
        ValueError: If environment variables are missing
        psycopg2.Error: If database operations fail
    """
    try:
        conn_string = get_pg_connection_string()
        logger.info("Starting trace creation process")
        logger.debug(
            f"Configuration - max_gap_seconds: {max_gap_seconds}, chunk_size_hours: {chunk_size_hours}, test_mode: {test_mode}")

        with psycopg2.connect(conn_string) as conn:
            logger.debug("Database connection established successfully")
            start_time, end_time = get_time_range(conn)
            logger.debug(
                f"Time range retrieved - start: {datetime.fromtimestamp(start_time / 1000)}, end: {datetime.fromtimestamp(end_time / 1000)}")

            chunk_size_ms = chunk_size_hours * 3600000  # Convert hours to milliseconds
            current_time = start_time

            # Calculate total chunks for progress bar
            total_chunks = (end_time - start_time) // chunk_size_ms + 1
            logger.debug(f"Total time chunks to process: {total_chunks}")

            if test_mode:
                logger.debug(
                    "Running in test mode - will only process one chunk")
                total_chunks = 1
                end_time = min(start_time + chunk_size_ms, end_time)

            with tqdm(total=total_chunks,
                      desc="Processing time chunks") as pbar:
                while current_time < end_time:
                    next_time = min(current_time + chunk_size_ms, end_time)

                    logger.debug(
                        f"Processing chunk from {datetime.fromtimestamp(current_time / 1000)} "
                        f"to {datetime.fromtimestamp(next_time / 1000)}")

                    chunk_df = fetch_chunk_data(conn, current_time, next_time)
                    logger.debug(
                        f"Fetched chunk data - {len(chunk_df)} rows retrieved")

                    if not chunk_df.empty:
                        unique_aircraft = chunk_df['hex'].nunique()
                        logger.debug(
                            f"Processing {unique_aircraft} unique aircraft in current chunk")

                        # Group by aircraft and find trace boundaries
                        logger.debug("Starting trace boundary identification")
                        chunk_df['trace_id'] = chunk_df.groupby('hex',
                                                                group_keys=False).apply(
                            lambda x: find_trace_boundaries(x, max_gap_seconds)
                        )

                        unique_traces = chunk_df.groupby(
                            ['hex', 'trace_id']).ngroups
                        logger.debug(
                            f"Identified {unique_traces} distinct traces")

                        # Create and insert traces
                        traces = []
                        for (hex_code, trace_id), trace_df in chunk_df.groupby(
                            ['hex', 'trace_id']):
                            logger.debug(
                                f"Preparing trace for aircraft {hex_code}, trace_id {trace_id} with {len(trace_df)} points")
                            traces.append(prepare_trace_arrays(trace_df))

                        logger.debug(
                            f"Inserting {len(traces)} traces into database")
                        insert_traces(conn, traces)
                        conn.commit()
                        logger.debug("Database commit successful")
                    else:
                        logger.debug(
                            "Chunk contained no data, skipping processing")

                    current_time = next_time
                    pbar.update(1)

                    if test_mode:
                        logger.debug("Test mode - exiting after first chunk")
                        break

        logger.info("Trace creation completed successfully")
        return True

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return False
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(f"Stack trace: ",
                     exc_info=True)  # Include stack trace in debug output
        return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
    # Configure logging - we'll set the actual level after parsing args
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create aircraft traces from historical ADSB data")
    parser.add_argument(
        "--max-gap-seconds",
        type=int,
        default=600,
        help="Maximum allowed gap between points in a trace (default: 600)"
    )
    parser.add_argument(
        "--chunk-size-hours",
        type=int,
        default=1,
        help="Size of time chunks to process in hours (default: 1)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test function"
    )

    args = parser.parse_args()

    # Set logging level based on mode
    if args.test:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Running test mode with DEBUG logging enabled")
        run(
            chunk_size_hours=1,
            test_mode=True
        )
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Running normal mode")
        success = run(max_gap_seconds=args.max_gap_seconds,
                      chunk_size_hours=args.chunk_size_hours)
        exit(0 if success else 1)


################################################################################
# end of aircraft_trace_creator.py
################################################################################