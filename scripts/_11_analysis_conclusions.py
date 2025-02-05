# Standard library imports
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extensions
import psycopg2.extras
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

# Define constants and paths
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", "."))
DATA_PATH = PROJECT_PATH / 'data'

# Input paths
CLUSTER_PATH = DATA_PATH / '_10_clustering' / 'results.parquet'

# Output paths
OUTPUT_BASE_PATH = DATA_PATH / '_11_analysis_&_conclusions'
TRAJECTORIES_PATH = OUTPUT_BASE_PATH / 'trajectories.parquet'
AUTOENCODER_TRAINING_SAMPLE_PATH = OUTPUT_BASE_PATH / 'autoencoder_training_sample.parquet'


################################################################################
# global supporting functions
################################################################################


def optimize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize numeric datatypes in DataFrame to minimize memory usage.

    Args:
        df: Input DataFrame to optimize

    Returns:
        pd.DataFrame: DataFrame with optimized datatypes for memory efficiency
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].astype(np.float64)

        if np.all(np.abs(series - series.round()) < 1e-10):
            min_val = series.min()
            max_val = series.max()

            if min_val >= 0:
                if max_val <= np.iinfo(np.uint8).max:
                    df[col] = series.astype(np.uint8)
                elif max_val <= np.iinfo(np.uint16).max:
                    df[col] = series.astype(np.uint16)
                elif max_val <= np.iinfo(np.uint32).max:
                    df[col] = series.astype(np.uint32)
                else:
                    df[col] = series.astype(np.uint64)
            else:
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    df[col] = series.astype(np.int8)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    df[col] = series.astype(np.int16)
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = series.astype(np.int32)
                else:
                    df[col] = series.astype(np.int64)
        else:
            float32_series = series.astype(np.float32)
            if np.allclose(series, float32_series, rtol=1e-7, atol=1e-14):
                df[col] = float32_series
            else:
                df[col] = series

    return df


################################################################################
# database operations
################################################################################


def create_db_connection(
    username: str = os.getenv('DB_USER'),
    password: str = os.getenv('DB_PASSWORD'),
    hostname: str = os.getenv('DB_HOST'),
    port: int = int(os.getenv('DB_PORT', '5432')),
    dbname: str = os.getenv('DB_NAME')
) -> psycopg2.extensions.connection:
    """Create a PostgreSQL database connection using psycopg2.

    Args:
        username: Database username. Defaults to DB_USER env variable
        password: Database password. Defaults to DB_PASSWORD env variable
        hostname: Database host. Defaults to DB_HOST env variable
        port: Database port. Defaults to DB_PORT env variable
        dbname: Database name. Defaults to DB_NAME env variable

    Returns:
        psycopg2.extensions.connection: Active database connection

    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        connection = psycopg2.connect(
            user=username,
            password=password,
            host=hostname,
            port=port,
            dbname=dbname
        )
        logger.debug(f"Successfully connected to database {dbname} at {hostname}")
        return connection
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise


################################################################################
# extract operations
################################################################################


def get_clustered_data() -> pd.DataFrame:
    """Read in the clustered data from the parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the clustered data

    Raises:
        FileNotFoundError: If clustered data file is not found
    """
    try:
        logger.info(f"Loading clustered data from {CLUSTER_PATH}")
        return pd.read_parquet(CLUSTER_PATH)
    except FileNotFoundError as e:
        logger.error(f"Failed to load clustered data: {str(e)}")
        raise


def get_trajectory_data(segment_ids: List[str]) -> pd.DataFrame:
    """Retrieve trajectory data for specified segment IDs from database.

    Args:
        segment_ids: List of segment IDs to query

    Returns:
        pd.DataFrame: DataFrame containing the trajectory data with optimized dtypes

    Raises:
        Exception: If query execution fails
    """
    query = """
    WITH trajectory_points AS (
        SELECT 
            segment_id,
            icao,
            start_timestamp,
            generate_subscripts(headings, 1) - 1 as point_idx,
            headings[generate_subscripts(headings, 1)] as heading,
            ground_speeds[generate_subscripts(ground_speeds, 1)] as ground_speed,
            altitudes[generate_subscripts(altitudes, 1)] as altitude,
            time_offsets[generate_subscripts(time_offsets, 1)] as time_offset
        FROM autoencoder_training_unscaled
        WHERE segment_id = ANY(%s)
    ),
    position_calc AS (
        SELECT
            segment_id,
            icao,
            start_timestamp,
            point_idx,
            heading * pi() / 180 as heading_rad,
            ground_speed,
            altitude,
            EXTRACT(EPOCH FROM (time_offset * interval '1 microsecond')) as time_sec,
            ground_speed * sin(heading * pi() / 180) as dx,
            ground_speed * cos(heading * pi() / 180) as dy
        FROM trajectory_points
    ),
    path_points AS (
        SELECT
            segment_id,
            icao,
            start_timestamp,
            point_idx,
            SUM(dx * time_sec) OVER (PARTITION BY segment_id ORDER BY point_idx) as x,
            SUM(dy * time_sec) OVER (PARTITION BY segment_id ORDER BY point_idx) as y,
            altitude as z,
            heading_rad,
            ground_speed,
            time_sec
        FROM position_calc
    )
    SELECT
        segment_id,
        icao,
        start_timestamp,
        point_idx,
        ROUND(x::numeric, 2) as x,
        ROUND(y::numeric, 2) as y,
        ROUND(z::numeric, 2) as z,
        ROUND((heading_rad * 180 / pi())::numeric, 2) as heading_deg,
        ROUND(ground_speed::numeric, 2) as ground_speed,
        ROUND(time_sec::numeric, 2) as time_sec
    FROM path_points
    ORDER BY segment_id, point_idx;
    """

    conn = None
    try:
        conn = create_db_connection()
        with conn.cursor() as cur:
            logger.info(f"Executing trajectory query for {len(segment_ids)} segments")
            cur.execute(query, (segment_ids,))
            columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()

        df = pd.DataFrame(results, columns=columns)
        logger.info(f"Retrieved {len(df)} trajectory points")
        return optimize_datatypes(df)

    except Exception as e:
        logger.error(f"Error executing trajectory query: {str(e)}")
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")


def get_autoencoder_training_sample(
    sample_size: int = 100,
    output_path: Optional[Path] = AUTOENCODER_TRAINING_SAMPLE_PATH
) -> pd.DataFrame:
    """Get a random sample from autoencoder_training_unscaled table.

    Args:
        sample_size: Number of random samples to retrieve
        output_path: Path to save parquet file. If None, skip saving

    Returns:
        pd.DataFrame: DataFrame containing the random sample

    Raises:
        Exception: If query fails or saving fails
    """
    query = f"""
        SELECT *
        FROM autoencoder_training_unscaled
        ORDER BY RANDOM()
        LIMIT {sample_size};
    """

    conn = None
    try:
        conn = create_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            logger.info(f"Retrieving {sample_size} random training samples")
            cursor.execute(query)
            results = cursor.fetchall()

        df = pd.DataFrame(results)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"Saved training sample to {output_path}")

        return df

    except Exception as e:
        logger.error(f"Error getting training sample: {str(e)}")
        raise

    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")


################################################################################
# transform operations
################################################################################


def rotate_points(group: pd.DataFrame) -> pd.DataFrame:
	"""Rotate points in a segment to align with Y-axis (90 degrees).

	Args:
		group: DataFrame containing points for a single segment

	Returns:
		pd.DataFrame: DataFrame with rotated coordinates added
	"""
	target_angle = np.pi / 2

	first_points = group.head(4)
	dx = first_points['x'].to_numpy()[-1] - first_points['x'].to_numpy()[0]
	dy = first_points['y'].to_numpy()[-1] - first_points['y'].to_numpy()[0]
	initial_angle = np.arctan2(dy, dx)

	rotation_angle = target_angle - initial_angle

	cos_theta = np.cos(rotation_angle)
	sin_theta = np.sin(rotation_angle)
	rotation_matrix = np.array([[cos_theta, -sin_theta],
								[sin_theta, cos_theta]], dtype=np.float64)

	points = np.column_stack((
		group['x'].to_numpy(dtype=np.float64),
		group['y'].to_numpy(dtype=np.float64)
	))

	rotated_points = points @ rotation_matrix.T

	result = group.copy()
	result['original_x'] = group['x'].to_numpy(dtype=np.float64)
	result['original_y'] = group['y'].to_numpy(dtype=np.float64)
	result['x'] = rotated_points[:, 0]
	result['y'] = rotated_points[:, 1]
	result['z'] = group['z'].to_numpy(dtype=np.float64)

	return result


def rotate_trajectories(df: pd.DataFrame) -> pd.DataFrame:
	"""Transform trajectory data by rotating each segment to align with Y-axis.

	Args:
		df: Input DataFrame with trajectory coordinates

	Returns:
		pd.DataFrame: DataFrame with rotated coordinates
	"""
	try:
		logger.info("Starting trajectory rotation process")
		df = df.copy()
		df['x'] = df['x'].astype(np.float64)
		df['y'] = df['y'].astype(np.float64)
		df['z'] = df['z'].astype(np.float64)

		logger.info("Rotating trajectories to align with Y-axis")
		with tqdm(total=len(df['segment_id'].unique()),
				  desc="Rotating segments") as pbar:
			rotated_df = df.groupby('segment_id', group_keys=False).apply(
				lambda g: rotate_points(g))
			pbar.update(1)

		z_unchanged = np.allclose(df['z'].to_numpy(dtype=np.float64),
								  rotated_df['z'].to_numpy(dtype=np.float64))
		logger.info(f"Z coordinates preserved: {z_unchanged}")

		return rotated_df

	except Exception as e:
		logger.error(f"Error rotating trajectories: {str(e)}")
		raise


################################################################################
# load operations
################################################################################


def merge_and_save_trajectories(
	clustered_df: pd.DataFrame,
	trajectories_df: pd.DataFrame,
	output_path: Path = TRAJECTORIES_PATH
) -> pd.DataFrame:
	"""Join clustered and trajectory data and save to parquet.

	Args:
		clustered_df: DataFrame containing clustering results
		trajectories_df: DataFrame containing trajectory data
		output_path: Path to save merged results

	Returns:
		pd.DataFrame: Merged DataFrame that was saved

	Raises:
		ValueError: If DataFrames don't contain required columns or join produces unexpected rows
	"""
	try:
		logger.info("Merging clustered and trajectory data")

		if 'segment_id' not in clustered_df.columns or 'segment_id' not in trajectories_df.columns:
			raise ValueError("Both DataFrames must contain 'segment_id' column")

		joined_df = clustered_df.merge(
			trajectories_df,
			on='segment_id',
			how='left'
		)

		expected_rows = len(clustered_df) * 50
		if len(joined_df) != expected_rows:
			raise ValueError(
				f"Expected {expected_rows} rows after join (50 per cluster), but got {len(joined_df)}")

		output_path.parent.mkdir(parents=True, exist_ok=True)
		joined_df.to_parquet(output_path, index=False)
		logger.info(f"Saved merged trajectories to {output_path}")

		return joined_df

	except Exception as e:
		logger.error(f"Error merging and saving trajectories: {str(e)}")
		raise


################################################################################
# main function
################################################################################


def run(
	test_mode: bool = False,
	sample_size: int = 100,
	output_path: Optional[Path] = TRAJECTORIES_PATH,
	autoencoder_sample_path: Optional[Path] = AUTOENCODER_TRAINING_SAMPLE_PATH
) -> bool:
	"""Main execution function for trajectory analysis and processing.

	Args:
		test_mode: If True, run with minimal data for testing
		sample_size: Number of random samples to retrieve for autoencoder training
		output_path: Path to save merged trajectory data. If None, skip saving
		autoencoder_sample_path: Path to save autoencoder training samples. If None, skip saving

	Returns:
		bool: True if execution successful, False otherwise
	"""
	try:
		if test_mode:
			logger.setLevel(logging.DEBUG)
			logger.info("Running in test mode with reduced sample size")
			sample_size = min(sample_size, 10)

		# Get clustered data
		logger.info("Starting trajectory processing pipeline")
		clustered_df = get_clustered_data()

		# Process trajectories
		logger.info("Retrieving and processing trajectory data")
		trajectories_df = get_trajectory_data(list(clustered_df['segment_id']))
		rotated_trajectories_df = rotate_trajectories(trajectories_df)

		# Save merged results
		if output_path is not None:
			merge_and_save_trajectories(
				clustered_df,
				rotated_trajectories_df,
				output_path
			)

		# Get autoencoder training samples
		logger.info("Retrieving autoencoder training samples")
		get_autoencoder_training_sample(
			sample_size=sample_size,
			output_path=autoencoder_sample_path
		)

		logger.info("Processing completed successfully")
		return True

	except Exception as e:
		logger.error(f"Processing failed: {str(e)}")
		return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Process trajectory data and prepare autoencoder training samples")

	parser.add_argument(
		"--test",
		"-t",
		action="store_true",
		help="Run in test mode with debug logging and minimal data"
	)

	parser.add_argument(
		"--sample-size",
		type=int,
		default=100,
		help="Number of random samples to retrieve for autoencoder training (default: 100)"
	)

	parser.add_argument(
		"--output-path",
		type=Path,
		default=TRAJECTORIES_PATH,
		help="Path to save merged trajectory data"
	)

	parser.add_argument(
		"--autoencoder-sample-path",
		type=Path,
		default=AUTOENCODER_TRAINING_SAMPLE_PATH,
		help="Path to save autoencoder training samples"
	)

	args = parser.parse_args()

	success = run(
		test_mode=args.test,
		sample_size=args.sample_size,
		output_path=args.output_path,
		autoencoder_sample_path=args.autoencoder_sample_path
	)

	exit(0 if success else 1)

################################################################################
# end of trajectory_processor.py
################################################################################