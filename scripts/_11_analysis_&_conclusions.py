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

# Column type definitions - primitive types for array contents
DECIMAL_COLUMNS: set = {
    'vertical_rates', 'ground_speeds', 'headings', 'altitudes',
    'vertical_accels', 'ground_accels', 'turn_rates', 'climb_descent_accels'
}

INTEGER_COLUMNS: set = {
    'point_count', 'time_offsets'
}

STRING_COLUMNS: set = {
    'icao'
}

TIMESTAMP_COLUMNS: set = {'start_timestamp'}
INTERVAL_COLUMNS: set = {'segment_duration'}

# Columns that contain single values (not arrays)
SINGULAR_COLUMNS: set = {
    'segment_id',           # BIGINT
    'icao',                 # CHAR(7)
    'start_timestamp',      # TIMESTAMPTZ
    'segment_duration',     # INTERVAL
    'point_count'           # INTEGER
}

# Columns that contain arrays
ARRAY_COLUMNS: set = {
    'vertical_rates',         # DOUBLE PRECISION[]
    'ground_speeds',          # DOUBLE PRECISION[]
    'headings',               # DOUBLE PRECISION[]
    'altitudes',              # DOUBLE PRECISION[]
    'time_offsets',           # INTEGER[]
    'vertical_accels',        # DOUBLE PRECISION[]
    'ground_accels',          # DOUBLE PRECISION[]
    'turn_rates',             # DOUBLE PRECISION[]
    'climb_descent_accels'    # DOUBLE PRECISION[]
}

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
# data extract functions to support notebooks
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
    and saves the data to a parquet file.

    Args:
        segment_ids (List[str]): List of segment IDs to query

    Returns:
        pd.DataFrame: DataFrame containing the trajectory data

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

        return df

    except Exception as e:
        logger.error(
            f"Error executing trajectory query or saving to parquet: {str(e)}")
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")


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
# run
################################################################################

clustered_df = get_clustered_data()

trajectories_df = get_trajectory_data(list(clustered_df['segment_id']))

merge_and_save = merge_and_save_trajectories(clustered_df, trajectories_df)

################################################################################
# main guard
################################################################################



################################################################################
# end of analysis_&_conclusions.py
################################################################################
