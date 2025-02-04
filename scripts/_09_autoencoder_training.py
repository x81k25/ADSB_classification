# Standard library imports
import argparse
import logging
import os
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

# create write paths
model_path = PROJECT_PATH / 'data' / '_09_autoencoder_training' / 'model.pth'
data_path = PROJECT_PATH / 'data' / '_09_autoencoder_training' / 'results.parquet'

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
    'icao',                # CHAR(7)
    'start_timestamp',      # TIMESTAMPTZ
    'segment_duration',     # INTERVAL
    'point_count'          # INTEGER
}

# Columns that contain arrays
ARRAY_COLUMNS: set = {
    'vertical_rates',           # DOUBLE PRECISION[]
    'ground_speeds',           # DOUBLE PRECISION[]
    'headings',               # DOUBLE PRECISION[]
    'altitudes',              # DOUBLE PRECISION[]
    'time_offsets',           # INTEGER[]
    'vertical_accels',        # DOUBLE PRECISION[]
    'ground_accels',          # DOUBLE PRECISION[]
    'turn_rates',            # DOUBLE PRECISION[]
    'climb_descent_accels'    # DOUBLE PRECISION[]
}

################################################################################
# database connection functions
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


def create_db_engine() -> Engine:
    """Create SQLAlchemy engine from environment variables.

    Returns:
        SQLAlchemy Engine instance

    Raises:
        ValueError: If any required environment variables are missing
    """
    required_vars = {
        'DB_USER': os.getenv('DB_USER'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD'),
        'DB_HOST': os.getenv('DB_HOST'),
        'DB_PORT': os.getenv('DB_PORT'),
        'DB_NAME': os.getenv('DB_NAME')
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    connection_url = (
        f"postgresql://{required_vars['DB_USER']}:{required_vars['DB_PASSWORD']}"
        f"@{required_vars['DB_HOST']}:{required_vars['DB_PORT']}/{required_vars['DB_NAME']}"
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )


################################################################################
# data loading and preparation
################################################################################

def load_autoencoder_training_data(sample_size: int | None) -> pd.DataFrame:
    """Load autoencoder training data from PostgreSQL with appropriate dtype handling.

    Args:
        sample_size: Optional number of rows to limit the query to. If None, returns all rows.
                    When provided, returns that many random rows from the table.

    Returns:
        DataFrame containing autoencoder training data with properly typed columns:
        - DECIMAL columns as numpy arrays of float64 (full precision)
        - INTEGER columns as numpy arrays of int32
        - STRING columns as strings
        - TIMESTAMP columns as pandas timestamps
        - INTERVAL columns as pandas timedelta

    Raises:
        Exception: If data loading fails
    """
    engine = None
    try:
        engine = create_db_engine()

        if sample_size is not None:
            query = f"""
                SELECT *
                FROM autoencoder_training_unscaled
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """
        else:
            query = """
                SELECT *
                FROM autoencoder_training_unscaled
                ORDER BY icao, start_timestamp
            """

        # Initial load with basic type handling
        df = pd.read_sql_query(
            sql=query,
            con=engine,
            parse_dates=['start_timestamp']
        )

        logger.info(
            f"Successfully loaded {len(df)} rows from autoencoder_training_data")

        # Convert Python lists to numpy arrays with appropriate types
        for col in df.columns:
            if col in DECIMAL_COLUMNS:
                df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))

            elif col in INTEGER_COLUMNS and col in ARRAY_COLUMNS:
                df[col] = df[col].apply(lambda x: np.array(x, dtype=np.int32))

            elif col in INTERVAL_COLUMNS:
                df[col] = pd.to_timedelta(df[col])

        return df

    except Exception as e:
        logger.error(f"Error loading autoencoder training data: {str(e)}")
        raise

    finally:
        if engine:
            engine.dispose()


def prepare_data(df: pd.DataFrame) -> Tuple[
    np.ndarray, StandardScaler, List[str], List[str]]:
    """
    Transform dataframe with nested arrays into flattened format and scale the data.
    Removes any rows containing NaN values.

    Args:
        df: Input DataFrame containing nested arrays of length 50 for flight data

    Returns:
        Tuple containing:
            - Scaled data as numpy array ready for PyTorch
            - Fitted StandardScaler
            - List of feature names in order
            - List of segment IDs in order
    """
    try:
        # Arrays to unpack in order
        array_columns = [
            'time_offsets',
            'vertical_rates',
            'ground_speeds',
            'headings',
            'altitudes',
            'vertical_accels',
            'ground_accels',
            'turn_rates',
            'climb_descent_accels'
        ]

        # Validate array lengths before processing
        for col in array_columns:
            if not all(len(arr) == 50 for arr in df[col]):
                msg = f"Not all arrays in column {col} have length 50"
                logger.error(msg)
                raise ValueError(msg)

        # Capture segment IDs
        segment_ids = df['segment_id'].tolist()

        # Pre-allocate all columns in a dictionary
        new_columns = {}
        features = []  # Track feature names in order

        # Iterate through time points first, then features
        for i in range(50):
            for col in array_columns:
                base_name = col.rstrip('s')
                new_col = f"{base_name}_{i}"
                new_columns[new_col] = df[col].apply(lambda x: x[i])
                features.append(new_col)

        # Create new dataframe all at once
        result_df = pd.DataFrame(new_columns)

        # Remove rows with any NaN values
        initial_rows = len(result_df)
        result_df = result_df.dropna()
        removed_rows = initial_rows - len(result_df)

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows containing NaN values")
            logger.info(f"Retained {len(result_df)} valid rows")
            # Update segment_ids to match removed rows
            segment_ids = segment_ids[:len(result_df)]

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(result_df)

        # Final validation check
        if np.isnan(scaled_data).any():
            msg = "NaN values detected after scaling"
            logger.error(msg)
            raise ValueError(msg)

        return scaled_data, scaler, features, segment_ids

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

################################################################################
# model definition
################################################################################

class FlightDataset(Dataset):
    """Custom Dataset for flight data."""

    def __init__(self, data: Any) -> None:
        self.data = torch.FloatTensor(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class Autoencoder(nn.Module):
   """Autoencoder neural network architecture."""

   def __init__(self, input_dim: int, encoding_dim: int = 32) -> None:
       super(Autoencoder, self).__init__()

       self.encoder = nn.Sequential(
           nn.Linear(input_dim, 256),
           nn.ReLU(),
           nn.Linear(256, 128),
           nn.ReLU(),
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Linear(64, encoding_dim),
           nn.ReLU()
       )

       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, 64),
           nn.ReLU(),
           nn.Linear(64, 128),
           nn.ReLU(),
           nn.Linear(128, 256),
           nn.ReLU(),
           nn.Linear(256, input_dim)
       )

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       encoded = self.encoder(x)
       decoded = self.decoder(encoded)
       return decoded

   def encode(self, x: torch.Tensor) -> torch.Tensor:
       return self.encoder(x)


################################################################################
# training functions
################################################################################

def train_autoencoder(
    data: Any,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[Autoencoder, Dict[str, List[float]]]:
    """Train the autoencoder model."""
    dataset = FlightDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = data.shape[1]
    model = Autoencoder(input_dim).to(device)

    # Debug: Check input data statistics
    logger.debug(f"Input data shape: {data.shape}")
    logger.debug(f"Input data range: [{data.min():.4f}, {data.max():.4f}]")
    logger.debug(
        f"Input data contains NaN: {torch.isnan(torch.tensor(data)).any()}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    history = {'train_loss': []}

    start_time = perf_counter()
    with tqdm(total=epochs, desc='Training Progress') as pbar:
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)

                # Debug: Monitor batch statistics
                if batch_idx == 0:  # Only for first batch of each epoch
                    logger.debug(
                        f"Epoch {epoch} - Batch range: [{batch.min():.4f}, {batch.max():.4f}]")
                    logger.debug(
                        f"Batch contains NaN: {torch.isnan(batch).any()}")

                output = model(batch)

                # Debug: Monitor model output before loss calculation
                if batch_idx == 0:  # Only for first batch of each epoch
                    logger.debug(
                        f"Model output range: [{output.min():.4f}, {output.max():.4f}]")
                    logger.debug(
                        f"Output contains NaN: {torch.isnan(output).any()}")
                    logger.debug(
                        f"Output contains inf: {torch.isinf(output).any()}")

                loss = criterion(output, batch)

                # Debug: Monitor loss value
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"NaN/Inf detected in loss at epoch {epoch}, batch {batch_idx}")
                    logger.error(f"Loss value: {loss.item()}")
                    # Optional: you might want to break here or raise an exception

                optimizer.zero_grad()
                loss.backward()

                # Debug: Monitor gradients
                if batch_idx == 0:  # Only for first batch of each epoch
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_nan = torch.isnan(param.grad).any()
                            grad_inf = torch.isinf(param.grad).any()
                            if grad_nan or grad_inf:
                                logger.error(
                                    f"NaN/Inf detected in gradients for {name}")

                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloader)
            history['train_loss'].append(avg_train_loss)

            elapsed = perf_counter() - start_time
            pbar.set_postfix({
                'loss': f'{avg_train_loss:.4f}',
                'best_loss': f'{min(history["train_loss"]):.4f}',
                'elapsed': f'{elapsed:.1f}s'
            })
            pbar.update(1)

    final_time = perf_counter() - start_time
    logger.info(f'Training completed in {final_time:.1f}s')
    logger.info(f'Final loss: {avg_train_loss:.4f}')
    logger.info(f'Best loss: {min(history["train_loss"]):.4f}')

    return model, history


def save_artifacts(
    model: Autoencoder,
    reconstructed_df: pd.DataFrame,
    model_path: Path,
    data_path: Path
) -> None:
    """Save trained model and reconstructed data to files.

    Args:
        model: Trained Autoencoder model
        reconstructed_df: DataFrame with reconstructed data
        model_path: Path to save model file
        data_path: Path to save reconstructed data file
    """

    torch.save(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # When saving to parquet, the index will be preserved by default
    # But to make it explicit and ensure it's treated as a proper column:
    reconstructed_df.to_parquet(data_path, index=False)
    logger.info(f"Saved reconstructed data to {data_path}")


################################################################################
# main function
################################################################################

def run(
    test_mode: bool = False,
    max_sample_size: int = None,
    epochs: int = 50,
    batch_size: int = 32,
    model_path: Path = model_path,
    data_path: Path = data_path,
    device: str = None
) -> bool:
    """Main execution function.

    Args:
        test_mode: Whether to run in test mode
        max_sample_size: Number of rows to sample from the database
        epochs: Number of training epochs
        batch_size: Training batch size
        model_path: Path to save model file
        data_path: Path to save reconstructed data file
        device: Device to use for training (None for auto-detection)

    Returns:
        bool: True if execution successful, False otherwise
    """
    try:
        if test_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Note: outputs are still saved in test mode")
            logger.info("Running in test mode")
            epochs = 2
            batch_size = 16
            max_sample_size = 10000

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Load and prepare data
        df = load_autoencoder_training_data(max_sample_size)
        if test_mode:
            df = df.head(10)

        start_time = perf_counter()
        scaled_data, scaler, features, segment_ids = prepare_data(df)

        # Train model
        model, history = train_autoencoder(
            scaled_data,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        # Generate reconstructions
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(scaled_data).to(device)
            reconstructed_data = model(data_tensor).cpu().numpy()

        # Create reconstructed DataFrame with the original index
        reconstructed_df = pd.DataFrame(
            scaler.inverse_transform(reconstructed_data),
            columns=features
        )

        reconstructed_df['segment_id'] = segment_ids

        # Save outputs
        save_artifacts(model, reconstructed_df, model_path, data_path)

        execution_time = perf_counter() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

        return True

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train autoencoder on flight data')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with debug logging')
    parser.add_argument('--max_sample_size', type=int, default=100000,
                        help='Run in test mode with debug logging')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Output directory for saved model')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='Output directory for saved data')
    parser.add_argument('--device', type=str,
                        help='Device to use for training (cuda/cpu)')

    args = parser.parse_args()
    success = run(
        test_mode=args.test,
        max_sample_size=args.max_sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
        data_path=args.data_path,
        device=args.device
    )

    exit(0 if success else 1)

################################################################################
# end of _09_autoencoder_training.py
################################################################################