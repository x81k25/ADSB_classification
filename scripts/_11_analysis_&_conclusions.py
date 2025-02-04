# import Standard library imports
import argparse
import logging
import os
import pandas as pd
from pathlib import Path
import pickle
from time import perf_counter
from typing import Dict, List, Tuple, Any

# Third party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extensions
import psycopg2.extras
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List
import pandas as pd
from scipy import stats

################################################################################
# initial parameters and setup
################################################################################

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# create file paths
PROJECT_PATH = Path(os.getenv("PROJECT_PATH"))

# create read paths
cluster_path = PROJECT_PATH / 'data' / '_10_clustering' / 'results.parquet'

# create write paths
trajectories_path = PROJECT_PATH / 'data' / '_11_analysis_&_conclusions' / 'trajectories.parquet'
autoencoder_training_sample_path = PROJECT_PATH / 'data' / '_11_analysis_&_conclusions' / 'autoencoder_training_sample.parquet'


################################################################################
# global supporting functions
################################################################################

def create_db_connection(
    username: str = os.getenv('DB_USER'),
    password: str = os.getenv('DB_PASSWORD'),
    hostname: str = os.getenv('DB_HOST'),
    port: int = int(os.getenv('DB_PORT', 5432)),
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
        Database connection object

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
        logger.debug(
            f"Successfully connected to database {dbname} at {hostname}")
        return connection
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise


################################################################################
# get flight path trajectories
################################################################################

def get_clustered_data():
    """
    Read in the clustered data from the parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the clustered data
    """
    try:
        logger.info(f"Loading clustered data from {cluster_path}")
        return pd.read_parquet(cluster_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load clustered data: {str(e)}")
        raise


def get_trajectory_data(segment_ids: List[str]) -> pd.DataFrame:
    """
    Retrieves trajectory data for specified segment IDs using environment variables for DB connection
    and saves the data to a parquet file. Optimizes memory usage by assigning the most efficient
    data types to each column.

    Args:
        segment_ids (List[str]): List of segment IDs to query

    Returns:
        pd.DataFrame: DataFrame containing the trajectory data with optimized dtypes

    Raises:
        Exception: If query execution fails or if saving to parquet fails
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
        # Create connection using the helper function
        conn = create_db_connection()

        # Create a cursor and execute the query
        with conn.cursor() as cur:
            cur.execute(query, (segment_ids,))

            # Fetch column names from cursor description
            columns = [desc[0] for desc in cur.description]

            # Fetch all results
            results = cur.fetchall()

        # Create DataFrame from results
        df = pd.DataFrame(results, columns=columns)

        # Optimize data types for each column
        for col in df.columns:
            # Skip if the column is datetime or non-numeric
            if pd.api.types.is_datetime64_any_dtype(
                df[col]) or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Convert to float64 first to ensure consistent handling
            series = df[col].astype(np.float64)

            # Check if all values in the column are effectively integers
            # Using a small epsilon to account for floating point precision
            if np.all(np.abs(series - series.round()) < 1e-10):
                # If they are all integers, convert to the smallest integer type that can hold the data
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
                    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(
                        np.int8).max:
                        df[col] = series.astype(np.int8)
                    elif min_val >= np.iinfo(
                        np.int16).min and max_val <= np.iinfo(np.int16).max:
                        df[col] = series.astype(np.int16)
                    elif min_val >= np.iinfo(
                        np.int32).min and max_val <= np.iinfo(np.int32).max:
                        df[col] = series.astype(np.int32)
                    else:
                        df[col] = series.astype(np.int64)
            else:
                # If they are truly floats, use float32 if precision is sufficient, otherwise float64
                float32_series = series.astype(np.float32)
                if np.allclose(series, float32_series, rtol=1e-7, atol=1e-14):
                    df[col] = float32_series
                else:
                    df[col] = series

        return df

    except Exception as e:
        logger.error(
            f"Error executing trajectory query or saving to parquet: {str(e)}")
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")


import numpy as np
import pandas as pd


def rotate_points(group):
    """
    Rotate points in a segment to align with Y-axis (90 degrees).
    Only x and y coordinates are rotated, z remains unchanged.
    """
    # Target angle is 90 degrees (π/2 radians)
    target_angle = np.pi / 2

    # Calculate initial direction using first 4 points
    first_points = group.head(4)
    dx = first_points['x'].to_numpy()[-1] - first_points['x'].to_numpy()[0]
    dy = first_points['y'].to_numpy()[-1] - first_points['y'].to_numpy()[0]
    initial_angle = np.arctan2(dy, dx)

    # Calculate rotation angle
    rotation_angle = target_angle - initial_angle

    # Create rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]], dtype=np.float64)

    # Extract points as numpy arrays
    points = np.column_stack((
        group['x'].to_numpy(dtype=np.float64),
        group['y'].to_numpy(dtype=np.float64)
    ))

    # Perform rotation
    rotated_points = points @ rotation_matrix.T

    # Create new dataframe with rotated points
    result = group.copy()
    result['original_x'] = group['x'].to_numpy(dtype=np.float64)
    result['original_y'] = group['y'].to_numpy(dtype=np.float64)
    result['x'] = rotated_points[:, 0]
    result['y'] = rotated_points[:, 1]
    # Ensure z is proper numpy dtype but unchanged
    result['z'] = group['z'].to_numpy(dtype=np.float64)

    return result


def rotate_trajectories(df):
    """
    Transform trajectory data by rotating each segment to align with Y-axis (90 degrees).

    Parameters:
    df (pandas.DataFrame): Input DataFrame with columns: segment_id, x, y, z

    Returns:
    pandas.DataFrame: Transformed DataFrame with original and rotated coordinates
    """
    # Convert numeric columns to proper numpy dtypes
    df = df.copy()
    df['x'] = df['x'].astype(np.float64)
    df['y'] = df['y'].astype(np.float64)
    df['z'] = df['z'].astype(np.float64)

    # Apply rotation to each segment
    rotated_df = df.groupby('segment_id', group_keys=False).apply(rotate_points)

    # Verify that Z coordinates are unchanged
    z_unchanged = np.allclose(df['z'].to_numpy(dtype=np.float64),
                              rotated_df['z'].to_numpy(dtype=np.float64))
    print(f"Z coordinates preserved: {z_unchanged}")

    return rotated_df


def merge_and_save_trajectories(
    clustered_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    trajectories_path: Path = trajectories_path
):
    """
    Join clustered_df and trajectories_df on segment_id and save to parquet.
    Each row in clustered_df maps to multiple rows in trajectories_df.
    Clustering fields will be kept first in the output DataFrame.

    Args:
        clustered_df (pd.DataFrame): DataFrame containing clustering results (1 row per segment)
        trajectories_df (pd.DataFrame): DataFrame containing trajectory data (50 rows per segment)
        trajectories_path (Path): Path object specifying where to save the parquet file

    Returns:
        pd.DataFrame: The joined DataFrame that was saved
    """
    # Verify both DataFrames have segment_id
    if 'segment_id' not in clustered_df.columns or 'segment_id' not in trajectories_df.columns:
        raise ValueError("Both DataFrames must contain 'segment_id' column")

    # Join DataFrames
    # Using left join to keep order of clustered_df columns first
    joined_df = clustered_df.merge(
        trajectories_df,
        on='segment_id',
        how='left'
    )

    # Verify the expected multiplier in the joined result
    expected_rows = len(clustered_df) * 50
    if len(joined_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows after join (50 per cluster), but got {len(joined_df)}")

    # Create parent directory if it doesn't exist
    trajectories_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    joined_df.to_parquet(trajectories_path, index=False)

    return joined_df


################################################################################
# get sample autoencdoer_training_data
################################################################################

def get_autoencoder_training_sample(
    sample_size: int = 100,
    output_path: Path = autoencoder_training_sample_path
) -> pd.DataFrame:
    """
    Get a random sample from autoencoder_training_unscaled table and save as parquet.

    Args:
        sample_size: Number of random samples to retrieve
        output_path: Path to save parquet file. If None, only returns DataFrame

    Returns:
        pandas.DataFrame containing the random sample
    """
    query = f"""
        SELECT *
        FROM autoencoder_training_unscaled
        ORDER BY RANDOM()
        LIMIT {sample_size};
    """

    try:
        # Create database connection
        conn = create_db_connection()

        # Use RealDictCursor so we get results as dictionaries
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to parquet if path provided
        if output_path is not None:
            # Ensure parent directories exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)

    except Exception as e:
        raise Exception(f"Error getting training sample: {str(e)}")

    finally:
        if conn:
            conn.close()


################################################################################
# run
################################################################################

# get trajectories
clustered_df = get_clustered_data()

trajectories_df = get_trajectory_data(list(clustered_df['segment_id']))

rotated_trajectories_df = rotate_trajectories(trajectories_df)

merge_and_save = merge_and_save_trajectories(clustered_df, rotated_trajectories_df)

# get sample datasets
get_autoencoder_training_sample()

################################################################################
# main guard
################################################################################



################################################################################
# end of analysis_&_conclusions.py
################################################################################
