# Internal/built-in libraries
import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
import traceback
from typing import Dict, Any, Optional

# Third-party libraries
import duckdb
from dotenv import load_dotenv

################################################################################
# initial parameters and setup
################################################################################

load_dotenv()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# create file paths
PROJECT_PATH = Path(os.getenv("PROJECT_PATH"))

# create read paths
null_count_query_path = PROJECT_PATH / 'sql' / 'query_traces_null_count.sql'
distinct_count_query_path = PROJECT_PATH / 'sql' / 'query_traces_distinct_count.sql'

# create write paths
results_path = PROJECT_PATH / 'data' / '_05_traces_EDA' / 'results.json'
trace_sample_path = PROJECT_PATH / 'data' / '_05_traces_EDA' / 'traces_sample.parquet'

################################################################################
# database connection
################################################################################

def setup_duckdb_postgres(
    username: str = os.getenv('DB_USER'),
    password: str = os.getenv('DB_PASSWORD'),
    hostname: str = os.getenv('DB_HOST'),
    port: int = int(os.getenv('DB_PORT', '5432')),
    dbname: str = os.getenv('DB_NAME')
) -> duckdb.DuckDBPyConnection:
    """
    Sets up a DuckDB connection with PostgreSQL integration using environment variables.

    Args:
        username: Database username (default: DB_USER env var)
        password: Database password (default: DB_PASSWORD env var)
        hostname: Database host address (default: DB_HOST env var)
        port: Database port (default: DB_PORT env var)
        dbname: PostgreSQL database name (default: DB_NAME env var)

    Returns:
        DuckDB connection object configured with PostgreSQL

    Raises:
        duckdb.Error: If connection setup fails
    """
    try:
        con = duckdb.connect()
        logger.info("Created DuckDB connection")

        # Configure progress bar settings
        con.execute("SET progress_bar_time=1")
        con.execute("SET enable_progress_bar=true")

        con.execute("LOAD postgres")
        logger.debug("Loaded PostgreSQL extension")

        conn_string = ' '.join([
            f"dbname={dbname}",
            f"host={hostname}",
            f"port={port}",
            f"user={username}",
            f"password={password}"
        ])

        con.execute(f"ATTACH '{conn_string}' AS postgres_db (TYPE postgres)")
        logger.info("Successfully attached PostgreSQL database")

        return con

    except Exception as e:
        logger.error(f"Failed to setup DuckDB-PostgreSQL connection: {str(e)}")
        raise


################################################################################
# analysis functions
################################################################################

def get_total_row_count(con: duckdb.DuckDBPyConnection) -> int:
    """
    Get total number of rows in aircraft_traces table.

    Args:
        con: DuckDB connection object

    Returns:
        Total row count

    Raises:
        duckdb.Error: If query fails
    """
    try:
        result = con.execute("""
            SELECT COUNT(*) 
            FROM postgres_db.aircraft_traces;
        """).fetchone()[0]
        logger.debug(f"Retrieved total row count: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to get row count: {str(e)}")
        raise


def get_mean_point_count(con: duckdb.DuckDBPyConnection) -> float:
    """
    Calculate mean number of points per aircraft.

    Args:
        con: DuckDB connection object

    Returns:
        Mean point count

    Raises:
        duckdb.Error: If query fails
    """
    try:
        result = con.execute("""
            SELECT AVG(point_count) as mean_points
            FROM (
                SELECT COUNT(*) as point_count 
                FROM postgres_db.aircraft_traces
                GROUP BY icao
            ) subquery;
        """).fetchone()[0]
        logger.debug(f"Retrieved mean point count: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to get mean point count: {str(e)}")
        raise


def get_mean_duration(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """
    Get mean duration statistics for aircraft traces.

    Args:
        con: DuckDB connection object

    Returns:
        Dictionary containing mean duration in the format:
        {
            "value": float,  # Duration value
            "units": str     # Unit of measure (default: "minutes")
        }

    Raises:
        duckdb.Error: If query fails
    """
    try:
        mean_duration = con.execute("""
            SELECT 
                CAST(AVG(duration_ms) AS BIGINT)/1000.0/60.0 as mean_duration_minutes
            FROM (
                SELECT 
                    EPOCH_MS(MAX(actual_timestamp) - MIN(actual_timestamp)) as duration_ms
                FROM postgres_db.aircraft_traces
                GROUP BY icao
            ) subquery;
        """).fetchone()

        logger.debug("Retrieved mean duration statistics")
        return {
            "value": float(mean_duration[0]),
            "units": "minutes"
        }
    except Exception as e:
        logger.error(f"Failed to get mean duration statistics: {str(e)}")
        raise


def get_traces_over_duration(con: duckdb.DuckDBPyConnection,
                             min_duration_minutes: int) -> int:
    """
    Get count of traces longer than specified duration.

    Args:
        con: DuckDB connection object
        min_duration_minutes: Minimum duration threshold in minutes

    Returns:
        Count of traces exceeding the duration threshold

    Raises:
        duckdb.Error: If query fails
    """
    try:
        count = con.execute("""
            SELECT COUNT(DISTINCT icao)
            FROM (
                SELECT icao
                FROM postgres_db.aircraft_traces
                GROUP BY icao
                HAVING DATEDIFF('minute', MIN(actual_timestamp), MAX(actual_timestamp)) > ?
            ) subquery;
        """, [min_duration_minutes]).fetchone()[0]

        logger.debug(
            f"Retrieved count of traces longer than {min_duration_minutes} minutes")
        return count
    except Exception as e:
        logger.error(f"Failed to get traces over duration count: {str(e)}")
        raise


def get_null_counts(
    con: duckdb.DuckDBPyConnection,
    null_count_query_path: Path,
) -> dict:
    """
    Analyzes null counts in aircraft_traces table and returns results as a dictionary.

    Args:
        con: DuckDB connection with PostgreSQL configured
        null_count_query_path: Path to SQL file containing null count query

    Returns:
        dict: Dictionary with column names as keys and their null count stats as values.
        Format: {
            'column_name': {
                'null_count': int,
                'null_percentage': float
            },
            ...
        }
    """
    try:
        # Read SQL query from file and remove trailing semicolon
        with open(null_count_query_path, 'r') as f:
            null_count_query = f.read().strip().rstrip(';')

        # Execute query and get results as a DataFrame
        logger.info("Executing null count analysis query...")
        result_df = con.execute(null_count_query).df()

        # Convert wide format to dictionary
        null_stats = {}
        total_rows = result_df['total_rows'].iloc[0]

        # Process each column (skipping total_rows and percentage columns)
        for col in result_df.columns:
            if col == 'total_rows' or col.endswith('_null_pct'):
                continue

            if col.endswith('_nulls'):
                base_col = col[:-6]  # Remove '_nulls' suffix
                null_stats[base_col] = {
                    'null_count': int(result_df[col].iloc[0]),
                    'null_percentage': float(
                        result_df[f'{base_col}_null_pct'].iloc[0])
                }

        logger.info("Successfully analyzed null counts")
        return null_stats

    except Exception as e:
        logger.error(f"Failed to analyze null counts: {str(e)}")
        raise


def get_distinct_counts(
    con: duckdb.DuckDBPyConnection,
    distinct_count_query_path: Path,
) -> dict:
    try:
        with open(distinct_count_query_path, 'r') as f:
            distinct_count_query = f.read().strip().rstrip(';')

        query_result = con.execute(distinct_count_query)
        result_df = query_result.df()

        if result_df.empty:
            logger.error("Query returned no results")
            return {}

        distinct_stats = {}

        # Log all columns for debugging
        logger.info(f"All columns: {result_df.columns.tolist()}")

        # Find count columns (ones starting with 'distinct_')
        count_cols = [col for col in result_df.columns if
                      col.startswith('distinct_')]

        for count_col in count_cols:
            try:
                # Extract base column name (everything after 'distinct_')
                base_col = count_col.replace('distinct_', '')

                # Look for corresponding percentage column
                pct_col = f"{base_col}_distinct_pct"

                if pct_col not in result_df.columns:
                    logger.warning(
                        f"No percentage column found for {count_col}")
                    continue

                distinct_count = result_df[count_col].iloc[0]
                distinct_pct = result_df[pct_col].iloc[0]

                if pd.isna(distinct_count) or pd.isna(distinct_pct):
                    logger.warning(f"Null values found for {base_col}")
                    continue

                distinct_stats[base_col] = {
                    'distinct_count': int(distinct_count),
                    'distinct_percentage': float(distinct_pct)
                }

            except Exception as e:
                logger.error(f"Error processing column {count_col}: {str(e)}")
                continue

        return distinct_stats

    except Exception as e:
        logger.error(f"Failed to analyze distinct counts: {str(e)}")
        logger.error(f"Query that caused error:\n{distinct_count_query}")
        return {}


def get_distinct_values(
    con: duckdb.DuckDBPyConnection,
    distinct_counts: dict
) -> dict:
    """
    Get all distinct values for fields with cardinality <= 32.

    Args:
        con: DuckDB connection object
        distinct_counts: Dictionary containing distinct count statistics for each field

    Returns:
        Dictionary mapping field names to arrays of their distinct values

    Raises:
        duckdb.Error: If query fails
    """
    try:
        # Filter fields with <= 32 distinct values
        low_card_fields = [
            field for field, stats in distinct_counts.items()
            if stats['distinct_count'] <= 32
        ]

        if not low_card_fields:
            logger.debug("No fields found with <=32 distinct values")
            return {}

        # Build single query to get distinct values for all fields at once
        fields_sql = ", ".join([
            f"array_agg(DISTINCT {field}) AS {field}"
            for field in low_card_fields
        ])

        query = f"""
            SELECT {fields_sql}
            FROM postgres_db.aircraft_traces;
        """

        logger.debug(f"Executing query for {len(low_card_fields)} fields")
        result = con.execute(query).fetchone()

        # Convert result to dictionary
        output = {
            field: list(values) if values is not None else []
            for field, values in zip(low_card_fields, result)
        }

        logger.debug(f"Retrieved distinct values for {len(output)} fields")
        return output

    except Exception as e:
        logger.error(f"Failed to get distinct values: {str(e)}")
        raise


def fetch_and_save_sample(
    total_row_count: int,
    con: duckdb.DuckDBPyConnection,
    trace_sample_path: Path
) -> None:
    """
    Sample 0.5% of aircraft traces data and save to parquet file.
    Uses RANDOM() sampling and DuckDB COPY command.

    Args:
        total_row_count (int): Total number of rows in the aircraft_traces table
        con: DuckDB connection from setup_duckdb_postgres()
        trace_sample_path (Path): Path object specifying where to save the sample parquet file

    Raises:
        ValueError: If total_row_count is not a positive integer
        duckdb.Error: If query or file writing fails
    """
    logger.debug("Starting fetch_and_save_sample with %d rows to path: %s",
                 total_row_count, trace_sample_path)

    try:
        # Ensure output directory exists
        trace_sample_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Windows path to forward slashes for DuckDB
        path_str = str(trace_sample_path).replace('\\', '/')

        # Create view with RANDOM() sampling
        view_query = """
        CREATE OR REPLACE TEMPORARY VIEW sampled_traces AS 
        SELECT * 
        FROM postgres_db.aircraft_traces 
        WHERE RANDOM() <= 0.005
        """

        copy_query = f"COPY sampled_traces TO '{path_str}'"

        # Execute queries
        logger.debug("Creating sampling view...")
        con.execute(view_query)

        logger.debug("Copying to parquet file...")
        con.execute(copy_query)

        # Verify output
        if trace_sample_path.exists():
            file_size = trace_sample_path.stat().st_size
            logger.info("Successfully created sample file (%d bytes)",
                        file_size)
        else:
            raise FileNotFoundError(
                f"Expected output file not found: {trace_sample_path}")

    except Exception as e:
        logger.error("Failed to create sample file", exc_info=True)
        raise


################################################################################
# main function
################################################################################

def run(
    point_count_limit: int = 100,
    min_duration_minutes: Optional[int] = None,
    test_mode: bool = False
) -> bool:
    """
    Main execution function for aircraft traces analysis.

    Args:
        point_count_limit: Number of top point counts to retrieve
        min_duration_minutes: Minimum duration filter for traces
        test_mode: If True, runs in test mode with limited operations and debug logging

    Returns:
        bool: True if execution successful, False otherwise
    """
    try:
        if test_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Running in test mode")

        results = {}

        logger.info("Starting aircraft traces analysis")
        con = setup_duckdb_postgres()

        try:
            # Get and log total row count
            results['total_rows'] = get_total_row_count(con)
            logger.info(f"Total row count: {results['total_rows']}")

            # Get and log mean point count
            results['mean_point_count'] = get_mean_point_count(con)
            logger.info(f"Mean point count: {results['mean_point_count']}")

            # Get and log duration statistics
            results['mean_duration'] = get_mean_duration(con)
            logger.info(f"Mean trace duration: {results['mean_duration']}")

            if not test_mode:
                # Get and log traces over 15 minutes duration
                results['traves_over_15_min'] = get_traces_over_duration(con, 15)
                logger.info(f"Traces over 15 minutes: {results['traves_over_15_min']}")

                # Get null counts for all columns
                results['null_counts'] = get_null_counts(con, null_count_query_path)
                for col, stats in results['null_counts'].items():
                    logger.info(f"Null count for {col}: {stats['null_count']}")
                    logger.info(f"Null percentage for {col}: {stats['null_percentage']}")

                # Get distinct value counts for all columns
                results['distinct_counts'] = get_distinct_counts(con, distinct_count_query_path)
                for col, stats in results['distinct_counts'].items():
                   logger.info(f"Distinct count for {col}: {stats['distinct_count']}")
                   logger.info(f"Distinct percentage for {col}: {stats['distinct_percentage']}")

                # get distinct values
                results['distinct_values'] = get_distinct_values(con, results['distinct_counts'])
                for col, values in results['distinct_values'].items():
                    logger.info(f"Distinct values for {col}: {values}")

                # Fetch and save sample of aircraft traces data
                fetch_and_save_sample(results['total_rows'], con, trace_sample_path)
                logger.info("Saved sample of aircraft traces data")

            # Save results to data/_05_traces_EDA/results.json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
                logger.info(
                    "Saved results to data/_05_traces_EDA/results.json")

            return True

        finally:
            con.close()
            logger.info("Closed database connection")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze aircraft trace data")
    parser.add_argument(
        "--point-count-limit",
        type=int,
        default=100,
        help="Number of top point counts to retrieve"
    )
    parser.add_argument(
        "--min-duration-minutes",
        type=int,
        default=15,
        help="Minimum duration filter for traces"
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run in test mode with limited operations and debug logging"
    )

    args = parser.parse_args()
    success = run(
        point_count_limit=args.point_count_limit,
        min_duration_minutes=args.min_duration_minutes,
        test_mode=args.test
    )

    exit(0 if success else 1)

################################################################################
# end of _05_traces_EDA.py
################################################################################