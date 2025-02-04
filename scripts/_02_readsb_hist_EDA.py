# Standard library imports
import logging
import os
from typing import Dict, List, Optional, Tuple

# Third party imports
import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
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

# Constants
REQUIRED_ENV_VARS = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']


################################################################################
# database connection functions
################################################################################

def test_duckdb_postgres_connection(
	username: str = os.getenv('DB_USER'),
	password: str = os.getenv('DB_PASSWORD'),
	hostname: str = os.getenv('DB_HOST'),
	port: int = int(os.getenv('DB_PORT', 5432)),
	dbname: str = os.getenv('DB_NAME')
) -> bool:
	"""
    Test DuckDB's connection to PostgreSQL and return basic table info.

    Args:
        username: PostgreSQL username (default: from env)
        password: PostgreSQL password (default: from env)
        hostname: Host address (default: from env)
        port: Port number (default: from env or 5432)
        dbname: Database name (default: from env)

    Returns:
        bool: True if connection successful, False otherwise

    Raises:
        Exception: If connection fails
    """
	try:
		pg_connection_string = f"postgresql://{username}:{password}@{hostname}:{port}/{dbname}"
		conn = duckdb.connect()

		logger.info("Testing PostgreSQL connection...")
		conn.execute("LOAD postgres;")
		conn.execute(f"ATTACH '{pg_connection_string}' AS pg (TYPE postgres)")

		result = conn.execute(
			"SELECT COUNT(*) as row_count FROM pg.aircraft_tracking").fetchall()
		row_count = result[0][0]
		logger.info(
			f"Successfully connected! Found {row_count:,} rows in aircraft_tracking table")

		return True

	except Exception as e:
		logger.error(f"Connection failed: {str(e)}")
		return False
	finally:
		if 'conn' in locals():
			conn.close()


def setup_duckdb_postgres(
	username: str = os.getenv('DB_USER'),
	password: str = os.getenv('DB_PASSWORD'),
	hostname: str = os.getenv('DB_HOST'),
	port: int = int(os.getenv('DB_PORT', 5432)),
	dbname: str = os.getenv('DB_NAME')
) -> duckdb.DuckDBPyConnection:
	"""
    Sets up a DuckDB connection with PostgreSQL integration.

    Args:
        username: Database username (default: from env)
        password: Database password (default: from env)
        hostname: Database host address (default: from env)
        port: Database port (default: from env or 5432)
        dbname: PostgreSQL database name (default: from env)

    Returns:
        DuckDBPyConnection: Configured DuckDB connection object

    Raises:
        Exception: If connection setup fails
    """
	try:
		logger.info("Setting up DuckDB connection with PostgreSQL...")
		conn = duckdb.connect()

		conn.execute("INSTALL postgres")
		conn.execute("LOAD postgres")

		conn_string = (f"dbname={dbname} host={hostname} port={port} "
					   f"user={username} password={password}")

		conn.execute(f"ATTACH '{conn_string}' AS postgres_db (TYPE postgres)")
		logger.info("DuckDB connection successfully configured")

		return conn

	except Exception as e:
		logger.error(f"Failed to setup DuckDB connection: {str(e)}")
		raise


################################################################################
# data analysis functions
################################################################################

def get_table_stats(connection: duckdb.DuckDBPyConnection,
					table_name: str) -> Tuple[int, Dict, Dict, Dict]:
	"""
    Get basic statistics for a table including row count and column information.

    Args:
        connection: DuckDB connection object
        table_name: Full name of the table

    Returns:
        Tuple containing:
            - Total row count
            - Dictionary of column types
            - Dictionary of null counts per column
            - Dictionary of distinct value counts per column
    """
	try:
		# Get row count
		row_count = \
		connection.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

		# Get column types
		type_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name.split('.')[-1]}'
            ORDER BY ordinal_position
        """
		result = connection.sql(type_query).fetchall()
		column_types = {col: dtype for col, dtype in result}

		# Get null and distinct counts
		columns = list(column_types.keys())

		with tqdm(total=2, desc="Calculating column statistics") as pbar:
			# Null counts
			null_query = "SELECT " + ", ".join([
				f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as {col}_null_count"
				for col in columns
			]) + f" FROM {table_name}"
			null_result = connection.sql(null_query).fetchone()
			null_counts = {col: count for col, count in
						   zip(columns, null_result)}
			pbar.update(1)

			# Distinct counts
			distinct_query = "SELECT " + ", ".join([
				f"COUNT(DISTINCT {col}) as {col}_distinct_count"
				for col in columns
			]) + f" FROM {table_name}"
			distinct_result = connection.sql(distinct_query).fetchone()
			distinct_counts = {col: count for col, count in
							   zip(columns, distinct_result)}
			pbar.update(1)

		return row_count, column_types, null_counts, distinct_counts

	except Exception as e:
		logger.error(f"Error getting table statistics: {str(e)}")
		raise


def create_summary_df(
	stats_tuple: Tuple[int, Dict, Dict, Dict]) -> pd.DataFrame:
	"""
    Create a summary dataframe from table statistics.

    Args:
        stats_tuple: Tuple containing row count and column statistics dictionaries

    Returns:
        pd.DataFrame: Summary statistics for each column
    """
	try:
		total_rows, column_types, null_counts, distinct_counts = stats_tuple

		df = pd.DataFrame({
			'data_type': pd.Series(column_types),
			'distinct_count': pd.Series(distinct_counts),
			'null_count': pd.Series(null_counts)
		})

		df['total_rows'] = total_rows
		df['distinct_percentage'] = (
				df['distinct_count'] / total_rows * 100).round(2)
		df['null_percentage'] = (df['null_count'] / total_rows * 100).round(2)

		return df[
			['data_type', 'total_rows', 'distinct_count', 'distinct_percentage',
			 'null_count', 'null_percentage']]

	except Exception as e:
		logger.error(f"Error creating summary DataFrame: {str(e)}")
		raise


def analyze_correlations(df: pd.DataFrame,
						 method: str = 'pearson',
						 numeric_columns: Optional[List[str]] = None,
						 correlation_threshold: float = 0.7) -> Tuple[
	pd.DataFrame, pd.DataFrame]:
	"""
    Analyze correlations in the dataset and identify strong correlations.

    Args:
        df: Input DataFrame containing the data
        method: Correlation method to use
        numeric_columns: Specific numeric columns to analyze
        correlation_threshold: Threshold for strong correlations

    Returns:
        Tuple containing:
            - Complete correlation matrix
            - DataFrame of strong correlations
    """
	try:
		logger.info(f"Analyzing correlations using {method} method...")

		if numeric_columns is None:
			exclude_patterns = ['timestamp', 'hex', 'flight', 'registration',
								'type', 'category', 'squawk', 'emergency',
								'nav_modes']

			numeric_columns = df.select_dtypes(
				include=[np.number]).columns.tolist()
			numeric_columns = [col for col in numeric_columns
							   if not any(pattern in col.lower()
										  for pattern in exclude_patterns)]

		numeric_df = df[numeric_columns].copy()

		if 'alt_baro' in numeric_df.columns:
			numeric_df['alt_baro'] = pd.to_numeric(
				numeric_df['alt_baro'].replace('ground', '0'),
				errors='coerce'
			)

		correlation_matrix = numeric_df.corr(method=method).round(4)

		# Find strong correlations
		mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
		strong_corr = correlation_matrix.where(mask)
		strong_pairs = []

		for col in strong_corr.columns:
			series = strong_corr[col]
			strong = series[abs(series) >= correlation_threshold]

			for idx, value in strong.items():
				if not pd.isna(value):
					strong_pairs.append({
						'Variable 1': col,
						'Variable 2': idx,
						'Correlation': value
					})

		if not strong_pairs:
			strong_correlations = pd.DataFrame(
				columns=['Variable 1', 'Variable 2', 'Correlation']
			)
		else:
			strong_correlations = pd.DataFrame(strong_pairs)
			strong_correlations = strong_correlations.sort_values(
				'Correlation',
				key=abs,
				ascending=False
			)

		return correlation_matrix, strong_correlations

	except Exception as e:
		logger.error(f"Error in correlation analysis: {str(e)}")
		raise


################################################################################
# main function
################################################################################

def run(
	table_name: str = "postgres_db.public.aircraft_tracking",
	correlation_method: str = "pearson",
	correlation_threshold: float = 0.7,
	sample_size: int = 10000
) -> bool:
	"""
    Main execution function for aircraft tracking data analysis.

    Args:
        table_name: Full name of the table to analyze
        correlation_method: Method to use for correlation analysis
        correlation_threshold: Threshold for identifying strong correlations
        sample_size: Number of rows to use for correlation analysis

    Returns:
        bool: True if execution successful, False otherwise
    """
	try:
		# Validate environment variables
		missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
		if missing_vars:
			raise ValueError(
				f"Missing required environment variables: {missing_vars}")

		# Test connection
		if not test_duckdb_postgres_connection():
			raise ConnectionError("Failed to establish database connection")

		# Setup connection
		conn = setup_duckdb_postgres()
		logger.info("Starting data analysis...")

		# Get table statistics
		stats = get_table_stats(conn, table_name)
		summary_df = create_summary_df(stats)
		logger.info("\nColumn Summary Statistics:")
		print(summary_df)

		# Correlation analysis
		logger.info(
			f"\nFetching {sample_size:,} rows for correlation analysis...")
		sample_df = conn.execute(
			f"SELECT * FROM {table_name} LIMIT {sample_size};"
		).df()

		corr_matrix, strong_corr = analyze_correlations(
			sample_df,
			method=correlation_method,
			correlation_threshold=correlation_threshold
		)

		logger.info("\nStrong Correlations:")
		print(strong_corr)

		conn.close()
		return True

	except Exception as e:
		logger.error(f"Analysis failed: {str(e)}")
		return False


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Aircraft tracking data analysis")
	parser.add_argument(
		"--table_name",
		default="postgres_db.public.aircraft_tracking",
		help="Full name of the table to analyze"
	)
	parser.add_argument(
		"--correlation_method",
		default="pearson",
		choices=["pearson", "spearman", "kendall"],
		help="Correlation method to use"
	)
	parser.add_argument(
		"--correlation_threshold",
		type=float,
		default=0.7,
		help="Threshold for strong correlations"
	)
	parser.add_argument(
		"--sample_size",
		type=int,
		default=10000,
		help="Number of rows to use for correlation analysis"
	)

	args = parser.parse_args()
	success = run(**vars(args))
	exit(0 if success else 1)

################################################################################
# end of _02_readsb_hist_EDA.py
################################################################################