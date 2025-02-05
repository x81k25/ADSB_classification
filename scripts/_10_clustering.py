# Standard library imports
import argparse
from dotenv import load_dotenv
from functools import partial
import logging
import os
from pathlib import Path
import pickle
from typing import (Tuple)
import warnings

# Third party imports
import hdbscan
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

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

# Constants
DEFAULT_MIN_CLUSTER_SIZE: int = 35
DEFAULT_MIN_SAMPLES: int = 4
DEFAULT_EPSILON: float = 0.25
INPUT_PATH: str = 'data/reconstructed_df.pkl'
OUTPUT_PATH: str = 'data/clustered_df.pkl'

# load .env
load_dotenv()

# create file paths
PROJECT_PATH = Path(os.getenv("PROJECT_PATH"))

# create read paths
input_data_path = PROJECT_PATH / 'data' / '_09_autoencoder_training' / 'results.parquet'

# create write paths
model_path = PROJECT_PATH / 'data' / '_10_clustering' / 'model.pkl'
data_path = PROJECT_PATH / 'data' / '_10_clustering' / 'results.parquet'

# suppress deprecation warnings
warnings.simplefilter('ignore', FutureWarning)

################################################################################
# data preparation
################################################################################

def load_data(input_data_path: Path = input_data_path) -> pd.DataFrame:
    """
    Load the flight data from a parquet file.

    Args:
        input_path: Path to the input parquet file (defaults to input_data_path)

    Returns:
        pd.DataFrame: Loaded flight data

    Raises:
        FileNotFoundError: If the input file doesn't exist
    """
    try:
        logger.info(f"Loading data from {input_data_path}")
        return pd.read_parquet(input_data_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


################################################################################
# clustering functions
################################################################################

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Modified preprocessing for high-dimensional flight data.
    Returns features DataFrame, segment IDs series, and the fitted scaler.
    """
    logger.info("Preprocessing data - removing outliers and scaling")

    # Store segment_id separately and ensure it's int
    segment_ids = df['segment_id'].astype(int).copy()

    # Remove segment_id before processing
    df_features = df.drop('segment_id', axis=1)

    # More lenient outlier removal (now only on feature columns)
    z_scores = np.abs(stats.zscore(df_features))
    df_no_outliers = df_features[
        (z_scores < 4).all(axis=1)]  # Increased to 4 std deviations

    # Keep corresponding segment_ids
    valid_indices = df_features.index[
        (z_scores < 4).all(axis=1)]
    segment_ids = segment_ids[valid_indices].reset_index(drop=True)

    # Normalize the features only
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(df_no_outliers),
        columns=df_no_outliers.columns
    )

    # Return features and segment_ids separately
    return features_scaled, segment_ids, scaler


def cluster_flight_patterns(
    df: pd.DataFrame,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON
) -> Tuple[pd.DataFrame, hdbscan.HDBSCAN]:
    """
    Cluster flight patterns using HDBSCAN after preprocessing.

    Args:
        df: DataFrame containing flight feature data (without segment_ids)
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        epsilon: Cluster selection epsilon parameter

    Returns:
        Tuple containing:
            - DataFrame with feature data and cluster labels
            - Fitted HDBSCAN object
    """
    logger.info("Starting flight pattern clustering")

    # Initialize and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )

    logger.info("Fitting HDBSCAN clusterer")
    cluster_labels = clusterer.fit_predict(df)

    # Create a copy of the dataframe and add cluster labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Log clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_points = np.sum(cluster_labels == -1)
    total_points = len(cluster_labels)

    logger.info(f"Number of clusters found: {n_clusters}")
    logger.info(
        f"Percentage of points labeled as noise: {100 * noise_points / total_points:.2f}%")

    return df_with_clusters, clusterer


def analyze_clusters(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze characteristics of each cluster.

    Args:
        df_with_clusters: DataFrame with cluster labels

    Returns:
        DataFrame with cluster statistics
    """
    logger.info("Analyzing cluster characteristics")
    cluster_stats = []

    for cluster in tqdm(sorted(df_with_clusters['cluster'].unique()),
                        desc="Analyzing clusters"):
        if cluster == -1:
            continue

        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


################################################################################
# functions to identify key points
################################################################################

def find_typical_point(cluster_data: pd.DataFrame,
                       metric: str = 'euclidean') -> int:
    """
    Find the most typical point in a cluster (point with smallest average distance to all other points).

    Args:
        cluster_data: DataFrame containing just the points from one cluster
                     (should not include cluster labels or metadata columns)
        metric: Distance metric to use ('euclidean', 'manhattan', etc.)

    Returns:
        int: Index of the most typical point in the cluster
    """
    # Convert to numpy for faster computation
    data = cluster_data.values

    # Calculate pairwise distances
    distances = np.zeros(len(data))
    for i in range(len(data)):
        if metric == 'euclidean':
            dist = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
        elif metric == 'manhattan':
            dist = np.sum(np.abs(data - data[i]), axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

        # Calculate average distance to all other points
        distances[i] = np.mean(dist)

    # Return index of point with minimum average distance
    return cluster_data.index[np.argmin(distances)]


def find_extreme_point(cluster_data: pd.DataFrame,
                       metric: str = 'euclidean') -> int:
    """
    Find the most extreme point in a cluster (point with largest average distance to all other points).

    Args:
        cluster_data: DataFrame containing just the points from one cluster
                     (should not include cluster labels or metadata columns)
        metric: Distance metric to use ('euclidean', 'manhattan', etc.)

    Returns:
        int: Index of the most extreme point in the cluster
    """
    # Convert to numpy for faster computation
    data = cluster_data.values

    # Calculate pairwise distances
    distances = np.zeros(len(data))
    for i in range(len(data)):
        if metric == 'euclidean':
            dist = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
        elif metric == 'manhattan':
            dist = np.sum(np.abs(data - data[i]), axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

        # Calculate average distance to all other points
        distances[i] = np.mean(dist)

    # Return index of point with maximum average distance
    return cluster_data.index[np.argmax(distances)]


def process_cluster(cluster_id: int, df: pd.DataFrame,
                    metric: str = 'euclidean') -> tuple:
    """
    Process a single cluster to find typical and extreme points.

    Args:
        cluster_id: ID of the cluster to process
        df: DataFrame with feature data and cluster labels
        metric: Distance metric to use

    Returns:
        tuple: (cluster_id, typical_idx, extreme_idx) or None if error
    """
    try:
        # Get cluster data (excluding cluster label)
        cluster_mask = df['cluster'] == cluster_id

        # Store the original indices before any processing
        original_indices = df[cluster_mask].index.values

        # Get cluster data and reset index for local processing
        cluster_data = df[cluster_mask].drop(['cluster'], axis=1).reset_index(
            drop=True)

        if len(cluster_data) < 2:
            logger.warning(
                f"Cluster {cluster_id} has less than 2 points, skipping")
            return None

        # Find typical and extreme points (these will return local indices)
        typical_local_idx = find_typical_point(cluster_data, metric)
        extreme_local_idx = find_extreme_point(cluster_data, metric)

        # Map local indices back to original dataframe indices
        typical_idx = original_indices[
            typical_local_idx] if typical_local_idx is not None else None
        extreme_idx = original_indices[
            extreme_local_idx] if extreme_local_idx is not None else None

        return (cluster_id, typical_idx, extreme_idx)

    except Exception as e:
        logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
        return None


################################################################################
# main function
################################################################################

def run(
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON,
    metric: str = 'euclidean',
    test: bool = False,
    max_sample_size: int = None
) -> bool:
    """
    Main execution function for flight pattern clustering.
    """
    try:
        if test:
            logger.setLevel(logging.DEBUG)
            logger.debug("Running in test mode")
            min_cluster_size = 10
            min_samples = 2
            max_sample_size = 1000
            epsilon = 0.3

        # Load the data
        df = load_data()

        # Store the segment_id separately and keep track of the original index
        df = df.reset_index(drop=True)  # Ensure clean integer index
        segment_ids = df['segment_id'].copy()
        df = df.drop(columns=['segment_id'])

        # Apply max_sample_size if provided and not in test mode
        if not test and max_sample_size is not None and len(
            df) > max_sample_size:
            logger.info(
                f"Sampling {max_sample_size} rows from {len(df)} total rows")
            # Use random state for reproducibility
            df = df.sample(max_sample_size, random_state=42)
            segment_ids = segment_ids.loc[df.index]
            # Reset indices after sampling
            df = df.reset_index(drop=True)
            segment_ids = segment_ids.reset_index(drop=True)
        elif test and len(df) > max_sample_size:
            df = df.sample(max_sample_size, random_state=42)
            segment_ids = segment_ids.loc[df.index]
            df = df.reset_index(drop=True)
            segment_ids = segment_ids.reset_index(drop=True)

        # Perform clustering
        df_with_clusters, clusterer = cluster_flight_patterns(
            df,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon
        )

        # save model object
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(clusterer, f)

        # Get unique clusters (excluding noise)
        unique_clusters = sorted(
            [c for c in df_with_clusters['cluster'].unique() if c != -1])
        logger.info(f"Processing {len(unique_clusters)} clusters for exemplars")

        # Initialize parallel processing
        num_cores = multiprocessing.cpu_count()
        process_cluster_partial = partial(process_cluster,
                                          df=df_with_clusters,
                                          metric=metric)

        # Process clusters in parallel with progress bar
        with Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap(process_cluster_partial, unique_clusters),
                total=len(unique_clusters),
                desc="Finding cluster exemplars"
            ))

        # Create final DataFrame with segment_ids reattached
        final_df = pd.DataFrame({
            'segment_id': segment_ids,
            'cluster': df_with_clusters['cluster'],
            'is_most_typical': False,
            'is_most_extreme': False
        })

        # Verify segment_id integrity
        logger.info("Verifying segment_id integrity...")
        if len(final_df) != len(segment_ids):
            raise ValueError(
                "Length mismatch between final_df and original segment_ids")

        # Add verification
        if not np.issubdtype(final_df['segment_id'].dtype, np.integer):
            logger.warning(
                "segment_id was converted to float, forcing back to int")
            final_df['segment_id'] = final_df['segment_id'].astype(int)

        # Update flags based on results
        for result in results:
            if result is not None:
                cluster_id, typical_idx, extreme_idx = result

                # Set typical flag
                final_df.loc[typical_idx, 'is_most_typical'] = True

                # Set extreme flag
                final_df.loc[extreme_idx, 'is_most_extreme'] = True

        # Verify we have the correct number of exemplars
        n_typical = final_df['is_most_typical'].sum()
        n_extreme = final_df['is_most_extreme'].sum()
        logger.info(
            f"Found {n_typical} typical and {n_extreme} extreme exemplars")

        if n_typical != len(unique_clusters) or n_extreme != len(
            unique_clusters):
            logger.warning(
                "Number of exemplars doesn't match number of clusters!")

        # Save results
        logger.info(f"Saving results to {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(data_path)

        return True

    except Exception as e:
        logger.error(f"Error in clustering process: {str(e)}")
        return False

    except Exception as e:
        logger.error(f"Error in clustering process: {str(e)}")
        return False


################################################################################
# script execution
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flight pattern clustering script")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="Minimum size of clusters"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="HDBSCAN min_samples parameter"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Cluster selection epsilon parameter"
    )
    parser.add_argument(
        "--max-sample-size",
        type=int,
        default=10000, #None,
        help="Maximum number of samples to use for clustering (if None, uses all data)"
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run script in test mode with debug logging"
    )

    args = parser.parse_args()
    success = run(**vars(args))
    exit(0 if success else 1)


################################################################################
# end of _10_clustering.py
################################################################################