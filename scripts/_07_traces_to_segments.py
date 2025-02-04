# Standard library imports
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from tqdm import tqdm

################################################################################
# initial parameters and setup
################################################################################

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# define paths
PROJECT_PATH = Path(os.getenv("PROJECT_PATH"))

segments_sample_path = PROJECT_PATH / 'data' / '_07_traces_to_segments' / 'segments_sample.parquet'

# Column type definitions
DECIMAL_COLUMNS: set = {
    'latitude', 'longitude', 'ground_speed', 'track',
    'roll_angle', 'nav_heading'
}

INTEGER_COLUMNS: set = {
    'altitude_baro', 'flags', 'vertical_rate_baro',
    'indicated_airspeed', 'nav_altitude_mcp', 'time_offsets'
}

STRING_COLUMNS: set = {
    'flight', 'squawk', 'category', 'emergency', 'icao'
}

TIMESTAMP_COLUMNS: set = {'start_timestamp'}
INTERVAL_COLUMNS: set = {'duration'}

SINGULAR_COLUMNS: set = {
    'icao',                  # CHAR(7)
    'start_timestamp',       # TIMESTAMPTZ
    'duration',             # INTERVAL
    'flight',               # VARCHAR(20)
    'category',             # VARCHAR(2)
    'emergency'             # VARCHAR(20)
}

ARRAY_COLUMNS: set = {
    'time_offsets',         # INTEGER[]
    'nav_heading',          # DECIMAL(6,2)[]
    'longitude',           # DECIMAL(9,6)[]
    'vertical_rate_baro',   # INTEGER[]
    'latitude',            # DECIMAL(9,6)[]
    'ground_speed',        # DECIMAL(6,1)[]
    'flags',               # INTEGER[]
    'altitude_baro',       # INTEGER[]
    'nav_altitude_mcp',    # INTEGER[]
    'track',              # DECIMAL(6,2)[]
    'indicated_airspeed',  # INTEGER[]
    'roll_angle',         # DECIMAL(6,2)[]
    'squawk'              # VARCHAR(4)[]
}

# Global connection pool
pool: Optional[SimpleConnectionPool] = None

################################################################################
# database connection management
################################################################################

def create_db_url(
    username: str = os.getenv('DB_USER', ''),
    password: str = os.getenv('DB_PASSWORD', ''),
    hostname: str = os.getenv('DB_HOST', ''),
    port: str = os.getenv('DB_PORT', ''),
    dbname: str = os.getenv('DB_NAME', '')
) -> str:
    """
    Creates database URL from environment variables or passed parameters.

    Args:
        username: Database username
        password: Database password
        hostname: Database host
        port: Database port
        dbname: Database name

    Returns:
        str: Database connection string
    """
    return f"postgresql://{username}:{password}@{hostname}:{port}/{dbname}"


def init_db_pool(
    minconn: int = 1,
    maxconn: int = 10,
    **db_params: Dict[str, Union[str, int]]
) -> None:
    """
    Initializes the database connection pool.

    Args:
        minconn: Minimum number of connections in pool
        maxconn: Maximum number of connections in pool
        **db_params: Optional database parameters to override env variables
    """
    global pool
    if pool is not None:
        return

    if not db_params:
        db_params = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }

    pool = SimpleConnectionPool(minconn, maxconn, **db_params)


@contextmanager
def get_db_connection():
    """
    Context manager for handling database connections from the pool.

    Yields:
        psycopg2.connection: Database connection from the pool

    Raises:
        RuntimeError: If database pool is not initialized
    """
    if pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool first.")

    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

################################################################################
# data extraction
################################################################################

def get_aircraft_trace_data(table_name: str) -> pd.DataFrame:
    """
    Queries and returns all relevant aircraft trace data.

    Args:
        table_name: The name of the table to query (e.g., 'aircraft_traces_01')

    Returns:
        pd.DataFrame: DataFrame containing all relevant aircraft trace data

    Raises:
        RuntimeError: If database pool is not initialized
        Exception: For any database-related errors
    """
    sql = """
        SELECT 
            -- Core Position/Movement
            icao,
            actual_timestamp,
            latitude,
            longitude,
            altitude_baro,
            ground_speed,
            track,
            vertical_rate_baro,

            -- Enhanced Movement Features
            roll_angle,
            indicated_airspeed,

            -- Aircraft Classification
            flight,
            category,

            -- Navigation Features
            nav_heading,
            nav_altitude_mcp,

            -- Validation/Verification Features
            emergency,
            squawk,
            flags

        FROM {}
        ORDER BY actual_timestamp;
    """.format(table_name)

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()

            return pd.DataFrame(data, columns=columns)

    except Exception as e:
        logger.error(f"Error querying data from {table_name}: {str(e)}")
        raise

################################################################################
# data transformation
################################################################################

def process_flights_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process flight data segments based on timestamp gaps and predefined SQL schema types.

    Args:
        df: Input dataframe with flight data

    Returns:
        pd.DataFrame: Processed dataframe with condensed segments and strongly typed columns
    """
    df = df.sort_values('actual_timestamp')

    time_diffs = df['actual_timestamp'].diff()
    segment_breaks = np.where(time_diffs > timedelta(minutes=10))[0]

    start_indices = np.concatenate(([0], segment_breaks))
    end_indices = np.concatenate((segment_breaks, [len(df)]))

    valid_segments = False
    for start, end in zip(start_indices, end_indices):
        segment = df.iloc[start:end]
        duration = segment['actual_timestamp'].iloc[-1] - segment['actual_timestamp'].iloc[0]
        if duration >= timedelta(minutes=15):
            valid_segments = True
            break

    if not valid_segments:
        return pd.DataFrame()

    segments = []
    for start, end in zip(start_indices, end_indices):
        segment = df.iloc[start:end].copy()
        duration = segment['actual_timestamp'].iloc[-1] - segment['actual_timestamp'].iloc[0]

        if duration < timedelta(minutes=15):
            continue

        row_data = {}
        row_data['icao'] = segment['icao'].iloc[0]
        row_data['start_timestamp'] = segment['actual_timestamp'].iloc[0]

        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        microseconds = duration.microseconds
        row_data['duration'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"

        for col in SINGULAR_COLUMNS - {'icao', 'start_timestamp', 'duration'}:
            if col not in segment.columns:
                continue
            value = segment[col].iloc[0]
            row_data[col] = value if pd.notna(value) else np.nan

        start_time = segment['actual_timestamp'].iloc[0]
        time_offsets = (segment['actual_timestamp'] - start_time).dt.total_seconds() * 1_000_000
        row_data['time_offsets'] = time_offsets.astype(np.float64).values

        for col in ARRAY_COLUMNS:
            if col not in segment.columns:
                continue

            if col in DECIMAL_COLUMNS:
                row_data[col] = segment[col].astype(np.float64).values
            elif col in INTEGER_COLUMNS:
                row_data[col] = segment[col].astype(np.float64).values
            else:
                values = segment[col].values
                values = np.where(pd.isna(values), np.nan, values)
                row_data[col] = values.astype(object)

        segments.append(row_data)

    result_df = pd.DataFrame(segments)
    time_cols = ['icao', 'start_timestamp', 'duration']
    other_cols = [col for col in result_df.columns if col not in time_cols + ['time_offsets']]
    result_df = result_df[time_cols + ['time_offsets'] + other_cols]

    return result_df

################################################################################
# data loading
################################################################################

def insert_aircraft_traces(df: pd.DataFrame) -> None:
    """
    Insert aircraft trace data into PostgreSQL database with proper array handling.

    Args:
        df: DataFrame containing aircraft trace data

    Raises:
        Exception: For any database-related errors
    """
    def format_array_element(val: Union[float, int, str], col_name: str) -> str:
        """Format array element with proper type conversion and precision"""
        if pd.isna(val):
            return 'NULL'

        try:
            if col_name in INTEGER_COLUMNS:
                return str(int(float(val)))

            if col_name in DECIMAL_COLUMNS:
                if col_name in {'longitude', 'latitude'}:
                    return f"{float(val):.6f}"
                elif col_name in {'nav_heading', 'track', 'roll_angle'}:
                    return f"{float(val):.2f}"
                elif col_name == 'ground_speed':
                    return f"{float(val):.1f}"

            if col_name in STRING_COLUMNS:
                return f'"{str(val)}"' if val else 'NULL'

            return str(val)
        except (ValueError, TypeError):
            return 'NULL'

    def convert_to_pg_array(val: Union[np.ndarray, List, None], col_name: str) -> Optional[str]:
        """Convert numpy array to PostgreSQL array string"""
        if val is None:
            return None

        if isinstance(val, np.ndarray):
            val = val.tolist()
        elif not isinstance(val, list):
            val = [val]

        formatted = [format_array_element(x, col_name) for x in val]
        array_str = ','.join(x for x in formatted if x != 'NULL')
        return '{' + array_str + '}' if array_str else None

    try:
        df = df.copy()

        for col in ARRAY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: convert_to_pg_array(x, col))

        singular_strings = STRING_COLUMNS - ARRAY_COLUMNS
        for col in singular_strings:
            if col in df.columns:
                df[col] = df[col].replace({np.nan: None})

        columns = df.columns
        values = [tuple(x) for x in df.to_numpy()]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                insert_query = f"""
                    INSERT INTO aircraft_trace_segments ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT (icao, start_timestamp) DO UPDATE
                    SET {', '.join(f"{col} = EXCLUDED.{col}"
                                   for col in columns
                                   if col not in ['icao', 'start_timestamp'])}
                """
                execute_values(cur, insert_query, values)
                conn.commit()

    except Exception as e:
        logger.error(f"Error during data insertion: {str(e)}")
        raise

################################################################################
# run function
################################################################################


def get_sample_set() -> None:
    """
    Retrieves a sample set of 100 records from aircraft_trace_segments table
    and saves it as a parquet file.

    Raises:
        RuntimeError: If database pool is not initialized
        Exception: For any database-related errors
    """
    try:
        with get_db_connection() as conn:
            sql = "SELECT * FROM aircraft_trace_segments LIMIT 100;"
            sample_df = pd.read_sql_query(sql, conn)

            # Ensure the parent directory exists
            segments_sample_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to parquet
            sample_df.to_parquet(segments_sample_path, index=False)
            logger.info(f"Sample set saved to {segments_sample_path}")

    except Exception as e:
        logger.error(f"Error creating sample set: {str(e)}")
        raise


def run(
    start_hex: str = "00",
    end_hex: str = "ff",
    test_mode: bool = False,
    max_workers: int = 10
) -> bool:
    """
    Main execution function to process aircraft traces with improved parallelism.
    """
    start_time = time.time()
    logger.debug(f"Starting execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if test_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Running in test mode")
        max_workers = 2
        end_hex = start_hex
        logger.debug(
            f"Test mode parameters: max_workers={max_workers}, processing only hex={start_hex}")

    try:
        # Initialize database connection pool
        pool_start = time.time()
        init_db_pool(minconn=2, maxconn=20)
        logger.debug(
            f"Database pool initialization took {time.time() - pool_start:.2f} seconds")

        hex_values = [f"{i:02x}" for i in
                      range(int(start_hex, 16), int(end_hex, 16) + 1)]
        logger.info(
            f"Processing {len(hex_values)} hex partitions from aircraft_traces_{start_hex} to aircraft_traces_{end_hex}")

        def process_hex_partition(hex_val: str) -> bool:
            """Process a single hex partition with internal parallelism"""
            hex_start_time = time.time()
            table_name = f"aircraft_traces_{hex_val}"
            logger.debug(f"Starting {table_name}")
            error_count = 0

            try:
                # Load data
                data_load_start = time.time()
                all_trace_data = get_aircraft_trace_data(table_name)
                logger.debug(
                    f"{table_name}: Data load took {time.time() - data_load_start:.2f}s")

                # Group data
                group_start = time.time()
                grouped_data = all_trace_data.groupby('icao', sort=True)
                total_icaos = len(grouped_data)
                logger.info(f"{table_name}: Processing {total_icaos} ICAOs")

                # Process data in parallel
                processed_results = []
                processing_start = time.time()
                completed_icaos = 0

                with ProcessPoolExecutor(max_workers=max(2, max_workers // len(
                    hex_values))) as executor:
                    futures = []
                    for icao, group_data in grouped_data:
                        futures.append(
                            executor.submit(process_flights_data, group_data))

                    for future in futures:
                        try:
                            result = future.result()
                            if not result.empty:
                                processed_results.append(result)
                            completed_icaos += 1
                            if completed_icaos % 50 == 0 or completed_icaos == total_icaos:  # Log every 50 ICAOs
                                logger.debug(
                                    f"{table_name}: Processed {completed_icaos}/{total_icaos} ICAOs")
                        except Exception as e:
                            error_count += 1
                            logger.error(
                                f"{table_name}: Error processing ICAO: {str(e)}")

                if processed_results:
                    # Combine and insert results
                    insert_start = time.time()
                    combined_results = pd.concat(processed_results,
                                                 ignore_index=True)
                    insert_aircraft_traces(combined_results)
                    logger.debug(
                        f"{table_name}: Inserted {len(combined_results)} records in {time.time() - insert_start:.2f}s")

                processing_time = time.time() - hex_start_time
                logger.info(
                    f"Completed {table_name} in {processing_time:.2f}s ({error_count} errors)")

                # get and save sample set
                logger.info("Getting sample set...")
                get_sample_set()
                logger.info(f"Sample data set saved to {segments_sample_path}")

                return True

            except Exception as e:
                logger.error(f"Fatal error in {table_name}: {str(e)}")
                return False

        # Process hex partitions in parallel using threads
        logger.info(
            f"Starting parallel processing with {min(len(hex_values), 4)} concurrent partitions")
        with ThreadPoolExecutor(
            max_workers=min(len(hex_values), 4)) as executor:
            futures = [executor.submit(process_hex_partition, hex_val) for
                       hex_val in hex_values]
            results = [future.result() for future in futures]

        if not all(results):
            logger.error("One or more partitions failed to process")
            return False

        total_execution_time = time.time() - start_time
        logger.info(f"Completed all partitions in {total_execution_time:.2f}s")
        logger.info(
            f"Average time per partition: {total_execution_time / len(hex_values):.2f}s")
        return True

    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        return False

################################################################################
# main guard
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process aircraft traces with hex table partitioning'
    )

    parser.add_argument(
        '--start_hex',
        type=str,
        help='Two-character hex string to start processing from',
        default="00"
    )

    parser.add_argument(
        '--end_hex',
        type=str,
        help='Two-character hex string to stop processing at',
        default="ff"
    )

    parser.add_argument(
        '--test',
        '-t',
        action='store_true',
        help='Run script in test mode with debug logging'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        help='Maximum number of parallel workers',
        default=10
    )

    args = parser.parse_args()

    if args.test:
        # Override all parameters with test values when in test mode
        success = run(
            start_hex="00",  # Set your desired test start_hex
            end_hex="0f",    # Set your desired test end_hex
            test_mode=True,  # Keep test_mode True
            max_workers=16    # Set your desired test max_workers
        )
    else:
        # Use command line arguments when not in test mode
        success = run(
            start_hex=args.start_hex,
            end_hex=args.end_hex,
            test_mode=args.test,
            max_workers=args.max_workers
        )

    exit(0 if success else 1)

################################################################################
# end of aircraft_trace_processor.py
################################################################################