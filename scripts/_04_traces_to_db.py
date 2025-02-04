# Internal/built-in libraries
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import gzip
import json
import logging
import multiprocessing
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third-party libraries
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import requests
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
BATCH_SIZE: int = 50000
MAX_WORKERS: int = multiprocessing.cpu_count() * 16
MAX_PROCESS_WORKERS: int = max(1, multiprocessing.cpu_count() - 1)
CHUNK_SIZE: int = 100
MAX_CONCURRENT_REQUESTS: int = 50

# Database configuration
DB_PARAMS: Dict[str, str] = {
    'dbname': os.getenv('DB_NAME', ''),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', ''),
    'port': os.getenv('DB_PORT', '')
}


################################################################################
# validation functions
################################################################################

def validate_hex_range(start_hex: str, end_hex: str) -> None:
    """
    Validate hex inputs are valid 2-character hex strings in correct order.

    Args:
        start_hex: Starting hex value
        end_hex: Ending hex value

    Raises:
        ValueError: If hex values are invalid or in wrong order
    """
    if not (len(start_hex) == 2 and len(end_hex) == 2):
        raise ValueError("Start and end hex must be 2 characters")

    try:
        start_val = int(start_hex, 16)
        end_val = int(end_hex, 16)
    except ValueError:
        raise ValueError("Invalid hex values provided")

    if start_val > end_val:
        raise ValueError("Start hex must be less than or equal to end hex")


################################################################################
# database operations
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


def insert_batch(conn: psycopg2.extensions.connection,
                 records: List[Tuple]) -> None:
    """
    Insert a batch of records into the database.

    Args:
        conn: Database connection
        records: List of record tuples to insert

    Raises:
        Exception: If insertion fails
    """
    if not records:
        return

    deduplicated_records = deduplicate_records(records)
    cursor = conn.cursor()

    try:
        insert_query = """
            INSERT INTO aircraft_traces (
                icao, actual_timestamp, base_timestamp, seconds_offset,
                latitude, longitude, altitude_baro, ground_speed, track,
                flags, vertical_rate_baro, flight, squawk, category,
                nav_qnh, nav_altitude_mcp, nav_heading, emergency,
                nic, rc, version, nic_baro, nac_p, nac_v, sil,
                sil_type, gva, sda, source_type, altitude_geom,
                vertical_rate_geom, indicated_airspeed, roll_angle,
                source_file_path
            ) VALUES %s
            ON CONFLICT (icao, actual_timestamp) DO UPDATE SET
                base_timestamp = EXCLUDED.base_timestamp,
                seconds_offset = EXCLUDED.seconds_offset,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                altitude_baro = EXCLUDED.altitude_baro,
                ground_speed = EXCLUDED.ground_speed,
                track = EXCLUDED.track,
                flags = EXCLUDED.flags,
                vertical_rate_baro = EXCLUDED.vertical_rate_baro,
                flight = EXCLUDED.flight,
                squawk = EXCLUDED.squawk,
                category = EXCLUDED.category,
                nav_qnh = EXCLUDED.nav_qnh,
                nav_altitude_mcp = EXCLUDED.nav_altitude_mcp,
                nav_heading = EXCLUDED.nav_heading,
                emergency = EXCLUDED.emergency,
                nic = EXCLUDED.nic,
                rc = EXCLUDED.rc,
                version = EXCLUDED.version,
                nic_baro = EXCLUDED.nic_baro,
                nac_p = EXCLUDED.nac_p,
                nac_v = EXCLUDED.nac_v,
                sil = EXCLUDED.sil,
                sil_type = EXCLUDED.sil_type,
                gva = EXCLUDED.gva,
                sda = EXCLUDED.sda,
                source_type = EXCLUDED.source_type,
                altitude_geom = EXCLUDED.altitude_geom,
                vertical_rate_geom = EXCLUDED.vertical_rate_geom,
                indicated_airspeed = EXCLUDED.indicated_airspeed,
                roll_angle = EXCLUDED.roll_angle,
                source_file_path = EXCLUDED.source_file_path
        """

        psycopg2.extras.execute_values(
            cursor,
            insert_query,
            deduplicated_records,
            template=None,
            page_size=1000
        )
        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert batch: {str(e)}")
        raise
    finally:
        cursor.close()


################################################################################
# network operations
################################################################################

def construct_url(date: datetime, dir_hex: str, file_hex: str) -> str:
    """
    Construct URL for trace file.

    Args:
        date: Date for the trace file
        dir_hex: Directory hex value
        file_hex: File hex value

    Returns:
        str: Constructed URL
    """
    return (f"https://samples.adsbexchange.com/traces/"
            f"{date.year:04d}/{date.month:02d}/{date.day:02d}/"
            f"{dir_hex}/trace_full_{file_hex}.json")


def get_valid_files_in_directory(date: datetime, dir_hex: str) -> List[str]:
    """
    Get list of actual files that exist in a directory by scraping the directory listing.

    Args:
        date: Date to check
        dir_hex: Directory hex value

    Returns:
        List[str]: List of valid file hex values
    """
    url = f"https://samples.adsbexchange.com/traces/{date.year:04d}/{date.month:02d}/{date.day:02d}/{dir_hex}/"

    try:
        with requests.Session() as session:
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            response = session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            files = []

            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith('trace_full_') and href.endswith(
                    '.json'):
                    hex_part = href[11:-5]
                    hex_part = hex_part.replace('~', '')
                    files.append(hex_part)

            return files

    except Exception as e:
        logger.error(
            f"Error fetching directory listing for {dir_hex}: {str(e)}")
        return []


async def fetch_url_content(session: aiohttp.ClientSession, url: str) -> \
Optional[bytes]:
    """
    Fetch content from URL using aiohttp.

    Args:
        session: aiohttp session
        url: URL to fetch

    Returns:
        Optional[bytes]: Content if successful, None if failed
    """
    try:
        async with session.get(url) as response:
            if response.status == 404:
                return None
            return await response.read()
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


################################################################################
# data processing
################################################################################

def process_content(content: Optional[bytes], url: str) -> List[Tuple]:
    """
    Process the content from a URL.

    Args:
        content: Raw content bytes
        url: Source URL

    Returns:
        List[Tuple]: Processed records
    """
    if content is None:
        return []

    try:
        if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
            bio = BytesIO(content)
            with gzip.GzipFile(fileobj=bio) as gz:
                data = json.loads(gz.read())
        else:
            data = json.loads(content)

        return list(process_trace_data(data, url))
    except Exception as e:
        logger.error(f"Error processing content from {url}: {str(e)}")
        return []


def process_trace_data(trace_data: Dict[str, Any], source_file: str) -> \
Iterator[Tuple]:
    """
    Process trace data into database records.

    Args:
        trace_data: Raw trace data
        source_file: Source file path

    Yields:
        Tuple: Processed record
    """
    base_timestamp = trace_data['timestamp']
    icao = trace_data['icao']

    for trace_point in trace_data['trace']:
        try:
            actual_timestamp = datetime.fromtimestamp(
                base_timestamp + trace_point[0],
                tz=timezone.utc
            )

            altitude_baro = trace_point[3]
            if isinstance(altitude_baro,
                          str) and altitude_baro.lower() == 'ground':
                altitude_baro = 0

            record = [
                icao,
                actual_timestamp,
                int(base_timestamp),
                trace_point[0],
                trace_point[1],
                trace_point[2],
                altitude_baro,
                trace_point[4],
                trace_point[5],
                trace_point[6],
                trace_point[7],
            ]

            metadata = trace_point[8] if len(trace_point) > 8 else None
            if metadata:
                record.extend([
                    metadata.get('flight'),
                    metadata.get('squawk'),
                    metadata.get('category'),
                    metadata.get('nav_qnh'),
                    metadata.get('nav_altitude_mcp'),
                    metadata.get('nav_heading'),
                    metadata.get('emergency'),
                    metadata.get('nic'),
                    metadata.get('rc'),
                    metadata.get('version'),
                    metadata.get('nic_baro'),
                    metadata.get('nac_p'),
                    metadata.get('nac_v'),
                    metadata.get('sil'),
                    metadata.get('sil_type'),
                    metadata.get('gva'),
                    metadata.get('sda')
                ])
            else:
                record.extend([None] * 17)

            record.extend([
                trace_point[9] if len(trace_point) > 9 else None,
                trace_point[10] if len(trace_point) > 10 else None,
                trace_point[11] if len(trace_point) > 11 else None,
                trace_point[12] if len(trace_point) > 12 else None,
                trace_point[13] if len(trace_point) > 13 else None,
                source_file
            ])

            yield tuple(record)

        except Exception as e:
            logger.error(
                f"Error processing trace point in {source_file}: {str(e)}")
            continue


def count_non_none_fields(record: Tuple) -> int:
    """
    Count number of non-None fields in a record.

    Args:
        record: Database record tuple

    Returns:
        int: Count of non-None fields
    """
    return sum(1 for field in record if field is not None)


def deduplicate_records(records: List[Tuple]) -> List[Tuple]:
    """
    Deduplicate records keeping the ones with more data.

    Args:
        records: List of record tuples

    Returns:
        List[Tuple]: Deduplicated records
    """
    seen = {}
    total_dups = 0

    for record in records:
        key = (record[0], record[1])
        if key in seen:
            total_dups += 1
            if count_non_none_fields(record) > count_non_none_fields(seen[key]):
                seen[key] = record
        else:
            seen[key] = record

    return list(seen.values())


async def process_url_chunk(
    urls: List[str],
    session: aiohttp.ClientSession,
    executor: ProcessPoolExecutor
) -> List[Tuple]:
    """
    Process a chunk of URLs concurrently.

    Args:
        urls: List of URLs to process
        session: Shared aiohttp ClientSession for HTTP requests
        executor: Shared ProcessPoolExecutor for content processing

    Returns:
        List[Tuple]: Processed records from all URLs
    """
    tasks = [fetch_url_content(session, url) for url in urls]
    contents = await asyncio.gather(*tasks)

    futures = [executor.submit(process_content, content, url)
               for content, url in zip(contents, urls) if content]

    results = []
    for future in as_completed(futures):
        try:
            results.extend(future.result())
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")

    return results


def remove_duplicates(records: List[tuple]) -> List[tuple]:
    """
    Remove duplicate records while keeping track of how many were removed.

    Args:
        records: List of record tuples

    Returns:
        List of unique records
    """
    seen = set()
    unique_records = []

    for record in records:
        if record not in seen:
            seen.add(record)
            unique_records.append(record)

    return unique_records


################################################################################
# main function
################################################################################

async def process_hex_range(
    date: datetime,
    start_hex: str,
    end_hex: str,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE
) -> bool:
    """
    Process all files in the hex range for the given date.

    Args:
        date: Date to process
        start_hex: Starting hex value
        end_hex: Ending hex value
        batch_size: Size of database insertion batches
        chunk_size: Size of URL processing chunks

    Returns:
        bool: True if successful, False if failed

    Raises:
        ValueError: If hex range is invalid
    """
    validate_hex_range(start_hex, end_hex)

    start_val = int(start_hex, 16)
    end_val = int(end_hex, 16)

    conn = None
    try:
        conn = get_db_connection()
        current_batch = []
        batch_count = 0
        files_processed = 0
        successful_files = 0
        duplicate_counts = {}  # Track duplicates per hex directory
        start_time = time.time()  # Track overall processing time

        # Calculate total directories for progress bar
        total_dirs = end_val - start_val + 1

        async with aiohttp.ClientSession() as session:
            with ProcessPoolExecutor(
                max_workers=MAX_PROCESS_WORKERS) as executor:
                with tqdm(total=total_dirs,
                          desc="Processing directories") as pbar:
                    for dir_val in range(start_val, end_val + 1):
                        dir_hex = f"{dir_val:02x}"
                        dir_start_time = time.time()  # Track time per directory
                        logger.info(f"Processing directory {dir_hex}")

                        valid_files = get_valid_files_in_directory(date,
                                                                   dir_hex)
                        logger.info(f"Found {len(valid_files)} files in directory {dir_hex}")

                        dir_duplicates = 0  # Track duplicates for current directory
                        dir_records_processed = 0  # Track records for current directory
                        first_records_shown = False  # Flag to control showing first 10 records

                        for i in range(0, len(valid_files), chunk_size):
                            chunk = valid_files[i:i + chunk_size]
                            urls = [construct_url(date, dir_hex, file_hex) for
                                    file_hex
                                    in chunk]

                            chunk_records = await process_url_chunk(urls,
                                                                    session,
                                                                    executor)

                            if chunk_records:
                                # Check for duplicates before extending
                                original_len = len(chunk_records)
                                unique_records = remove_duplicates(
                                    chunk_records)
                                duplicates = original_len - len(unique_records)
                                dir_duplicates += duplicates

                                current_batch.extend(unique_records)
                                successful_files += len(unique_records)
                                dir_records_processed += len(unique_records)

                                # Only show first 10 records once per hex directory
                                if not first_records_shown and unique_records:
                                    first_records_shown = True

                                while len(current_batch) >= batch_size:
                                    batch_to_insert = current_batch[:batch_size]
                                    current_batch = current_batch[batch_size:]
                                    insert_batch(conn, batch_to_insert)
                                    batch_count += 1

                            files_processed += len(chunk)
                            if files_processed % 100 == 0:
                                logger.info(
                                    f"Progress: Processed {files_processed} files, "
                                    f"found data in {successful_files} files"
                                )

                        # Store duplicate count for this directory
                        duplicate_counts[dir_hex] = dir_duplicates

                        # Log directory completion time and stats
                        dir_time = time.time() - dir_start_time
                        logger.info(
                            f"Directory {dir_hex} completed in {dir_time:.2f} seconds. "
                            f"Duplicates found: {dir_duplicates}, "
                            f"Records processed: {dir_records_processed}"
                        )
                        pbar.update(1)

                # Insert final batch if any records remain
                if current_batch:
                    logger.info(f"Processing final batch {batch_count + 1}")
                    insert_batch(conn, current_batch)

        total_time = time.time() - start_time
        logger.info(
            f"Final statistics:\n"
            f"Total files processed: {files_processed}\n"
            f"Files with data: {successful_files}\n"
            f"Total batches inserted: {batch_count + 1}\n"
            f"Total processing time: {total_time:.2f} seconds\n"
            f"Duplicate counts by directory: {json.dumps(duplicate_counts, indent=2)}"
        )
        return True

    except Exception as e:
        logger.error(f"Error processing hex range: {str(e)}")
        return False

    finally:
        if conn:
            conn.close()


async def run_traces_to_db(
    date: datetime = datetime(2024, 8, 1),
    start_hex: str = "00",
    end_hex: str = "ff",
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE
) -> bool:
    """
    Main execution function for processing aircraft trace data.
    Function names breaks "run" convention to avoid conflict with built-in run function.

    Args:
        date: Date to process
        start_hex: Starting hex value (default: "00")
        end_hex: Ending hex value (default: "ff")
        batch_size: Size of database insertion batches (default: BATCH_SIZE)
        chunk_size: Size of URL processing chunks (default: CHUNK_SIZE)

    Returns:
        bool: True if successful, False if failed
    """
    logger.info(
        f"Starting extraction for {date.date()} from {start_hex} to {end_hex}"
    )

    logger.debug("BATCH_SIZE: %s", BATCH_SIZE)
    logger.debug("MAX_WORKERS: %s", MAX_WORKERS)
    logger.debug("MAX_PROCESS_WORKERS: %s", MAX_PROCESS_WORKERS)
    logger.debug("MAX_CONCURRENT_REQUESTS: %s", MAX_CONCURRENT_REQUESTS)
    logger.debug("CHUNK_SIZE: %s", CHUNK_SIZE)

    try:
        return await process_hex_range(
            date,
            start_hex,
            end_hex,
            batch_size,
            chunk_size
        )
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process aircraft trace data")
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2024, 8, 1),
        help="Date to process (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--start-hex",
        type=str,
        default="00",
        help="Starting hex value (2 characters)"
    )
    parser.add_argument(
        "--end-hex",
        type=str,
        default="ff",
        help="Ending hex value (2 characters)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Database insertion batch size"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="URL processing chunk size"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test function"
    )

    args = parser.parse_args()

    # Set logging level based on mode
    if args.test:
        logger.setLevel(logging.DEBUG)
        logger.info("Running test mode with DEBUG logging enabled")
        test_args = {
            "date": datetime(2024, 8, 1),
            "start_hex": "00",
            "end_hex": "00",
            "batch_size": 10000,
            "chunk_size": 50000
        }
        success = asyncio.run(run_traces_to_db(
            test_args["date"],
            test_args["start_hex"],
            test_args["end_hex"],
            test_args["batch_size"],
            test_args["chunk_size"]
        ))
    else:
        success = asyncio.run(run_traces_to_db(
            args.date,
            args.start_hex,
            args.end_hex,
            args.batch_size,
            args.chunk_size
        ))

        exit(0 if success else 1)

################################################################################
# end of _04_traces_to_db.py
################################################################################