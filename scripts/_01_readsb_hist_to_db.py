# Standard library imports
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import gc
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Generator

# Third party imports
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import psutil
import psycopg2
from psycopg2.extras import execute_values


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

# Database configuration
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')


################################################################################
# global supporting functions
################################################################################

def get_db_connection() -> psycopg2.extensions.connection:
	"""
	Create and return a database connection.

	Returns:
		psycopg2.extensions.connection: Database connection object

	Raises:
		psycopg2.Error: If connection fails
	"""
	try:
		return psycopg2.connect(
			dbname=DB_NAME,
			user=DB_USER,
			password=DB_PASSWORD,
			host=DB_HOST,
			port=DB_PORT
		)
	except psycopg2.Error as e:
		logger.error(f"Failed to connect to database: {str(e)}")
		raise


################################################################################
# extract functions
################################################################################

def download_adsb_file(params: Tuple[int, int, int, str]) -> Optional[
	Dict[str, Any]]:
	"""
	Download and parse a single ADSB Exchange data file.

	Args:
		params: Tuple containing (year, month, day, time_str)

	Returns:
		Optional[Dict[str, Any]]: Parsed JSON data if successful, None if failed

	Raises:
		requests.RequestException: If download fails
		json.JSONDecodeError: If JSON parsing fails
	"""
	year, month, day, time_str = params
	base_url = "https://samples.adsbexchange.com/readsb-hist"
	file_url = f"{base_url}/{year:04d}/{month:02d}/{day:02d}/{time_str}.json.gz"

	try:
		response = requests.get(file_url, stream=True)
		response.raise_for_status()
		return json.loads(response.content)
	except (requests.RequestException, json.JSONDecodeError) as e:
		logger.error(f"Failed to download/parse {file_url}: {str(e)}")
		return None


def _generate_file_params(begin_dt: datetime, end_dt: datetime) -> Generator[
	Tuple[int, int, int, str], None, None]:
	"""
	Generate parameters for file downloads based on time range.

	Args:
		begin_dt: Start datetime
		end_dt: End datetime

	Yields:
		Tuple[int, int, int, str]: Parameters for file download (year, month, day, time_str)
	"""
	current_dt = begin_dt
	while current_dt <= end_dt:
		yield (
			current_dt.year,
			current_dt.month,
			current_dt.day,
			current_dt.strftime('%H%M%SZ')
		)
		current_dt += timedelta(minutes=5)


def extract_adsb_data(begin_dt: datetime, end_dt: datetime,
					  max_workers: int = 15) -> List[Dict[str, Any]]:
	"""
	Extract ADSB Exchange data between two timestamps using parallel processing.

	Args:
		begin_dt: Start datetime
		end_dt: End datetime
		max_workers: Maximum number of parallel workers

	Returns:
		List[Dict[str, Any]]: List of transformed aircraft records
	"""
	all_records: List[Dict[str, Any]] = []
	file_params = list(_generate_file_params(begin_dt, end_dt))

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(download_and_process_file, param) for param
				   in file_params]

		for future in tqdm(futures, desc="Downloading and processing files"):
			try:
				result = future.result()
				all_records.extend(result)
			except Exception as e:
				logger.error(f"Failed to process file: {str(e)}")
				continue

	return all_records


################################################################################
# transform functions
################################################################################

def transform_aircraft_record(aircraft: Dict[str, Any], file_timestamp: int,
							  snapshot_messages: int) -> Dict[str, Any]:
	"""
	Transform a single aircraft record from raw JSON format to database schema format.

	Args:
		aircraft: Raw aircraft data
		file_timestamp: Timestamp of the data file
		snapshot_messages: Total messages in the snapshot

	Returns:
		Dict[str, Any]: Transformed aircraft record
	"""
	return {
		'snapshot_timestamp': file_timestamp,
		'hex': aircraft.get('hex', ''),
		'actual_timestamp': file_timestamp - aircraft.get('seen', 0),
		'snapshot_messages': snapshot_messages,
		'flight': aircraft.get('flight', '').strip(),
		'registration': aircraft.get('r', ''),
		'type': aircraft.get('t', ''),
		'aircraft_type': aircraft.get('type', ''),
		'alt_baro': 0 if aircraft.get('alt_baro') == 'ground' else aircraft.get(
			'alt_baro'),
		'alt_geom': aircraft.get('alt_geom'),
		'ground_speed': aircraft.get('gs'),
		'track': aircraft.get('track'),
		'baro_rate': aircraft.get('baro_rate'),
		'squawk': aircraft.get('squawk', ''),
		'emergency': str(aircraft.get('emergency', '')),
		'category': aircraft.get('category', ''),
		'latitude': aircraft.get('lat'),
		'longitude': aircraft.get('lon'),
		'nic': aircraft.get('nic'),
		'rc': aircraft.get('rc'),
		'seen_pos': aircraft.get('seen_pos'),
		'version': aircraft.get('version'),
		'nav_qnh': aircraft.get('nav_qnh'),
		'nav_altitude_mcp': aircraft.get('nav_altitude_mcp'),
		'nav_heading': aircraft.get('nav_heading'),
		'nav_modes': aircraft.get('nav_modes', []),
		'messages': aircraft.get('messages', 0),
		'seen': aircraft.get('seen'),
		'rssi': aircraft.get('rssi'),
		'mlat': aircraft.get('mlat', []),
		'tisb': aircraft.get('tisb', [])
	}


def download_and_process_file(params: Tuple[int, int, int, str]) -> List[
	Dict[str, Any]]:
	"""
	Download and process a single ADSB data file.

	Args:
		params: Tuple containing (year, month, day, time_str)

	Returns:
		List[Dict[str, Any]]: List of transformed aircraft records
	"""
	data = download_adsb_file(params)
	if not data:
		return []

	file_timestamp = data.get('now', 0)
	snapshot_messages = data.get('messages', 0)
	_, _, _, time_str = params

	records = [
		transform_aircraft_record(aircraft, file_timestamp, snapshot_messages)
		for aircraft in data.get('aircraft', [])
	]

	logger.debug(f"Processed {len(records)} records from {time_str}")
	return records


################################################################################
# load functions
################################################################################

def load_records_to_db(records: List[Dict[str, Any]]) -> None:
	"""
	Load transformed aircraft records into PostgreSQL database.

	Args:
		records: List of transformed aircraft records

	Raises:
		psycopg2.Error: If database operation fails
	"""
	if not records:
		logger.info("No records to load")
		return

	insert_query = """
        INSERT INTO aircraft_tracking (
            snapshot_timestamp, hex, actual_timestamp, snapshot_messages,
            flight, registration, type, aircraft_type, alt_baro, alt_geom,
            ground_speed, track, baro_rate, squawk, emergency, category,
            latitude, longitude, nic, rc, seen_pos, version, nav_qnh,
            nav_altitude_mcp, nav_heading, nav_modes, messages, seen,
            rssi, mlat, tisb
        ) VALUES %s
        ON CONFLICT (snapshot_timestamp, hex) DO NOTHING
    """

	values = [(
		record['snapshot_timestamp'], record['hex'], record['actual_timestamp'],
		record['snapshot_messages'], record['flight'], record['registration'],
		record['type'], record['aircraft_type'], record['alt_baro'],
		record['alt_geom'], record['ground_speed'], record['track'],
		record['baro_rate'], record['squawk'], record['emergency'],
		record['category'], record['latitude'], record['longitude'],
		record['nic'], record['rc'], record['seen_pos'], record['version'],
		record['nav_qnh'], record['nav_altitude_mcp'], record['nav_heading'],
		record['nav_modes'], record['messages'], record['seen'],
		record['rssi'], record['mlat'], record['tisb']
	) for record in records]

	conn = None
	try:
		conn = get_db_connection()
		with conn.cursor() as cur:
			execute_values(cur, insert_query, values)
			conn.commit()
			logger.info(
				f"Successfully loaded {len(records)} records to database")
	except Exception as e:
		if conn:
			conn.rollback()
		logger.error(f"Failed to load records to database: {str(e)}")
		raise
	finally:
		if conn:
			conn.close()


################################################################################
# run function
################################################################################

def run(
	start_time: datetime = datetime(2024, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
	end_time: datetime = datetime(2024, 8, 2, 0, 0, 0, tzinfo=timezone.utc),
	chunk_size: int = 500,
	max_workers: int = 32
) -> bool:
	"""
	Main execution function for ADSB data processing.

	Args:
		start_time: Start time for data processing
		end_time: End time for data processing
		chunk_size: Number of files to process in each chunk
		max_workers: Maximum number of parallel workers

	Returns:
		bool: True if processing successful, False otherwise
	"""
	try:
		logger.info(f"Starting processing from {start_time} to {end_time}")

		total_intervals = int((end_time - start_time).total_seconds() / 300)
		current_time = start_time

		for chunk_start_idx in tqdm(range(0, total_intervals, chunk_size),
									desc="Processing chunks"):
			chunk_end_idx = min(chunk_start_idx + chunk_size, total_intervals)
			chunk_end = start_time + timedelta(minutes=5 * chunk_end_idx)

			logger.info(f"Processing chunk: {current_time} to {chunk_end}")

			try:
				chunk_records = extract_adsb_data(current_time, chunk_end,
												  max_workers)

				if chunk_records:
					load_records_to_db(chunk_records)
					del chunk_records
					gc.collect()
					logger.debug("Cleared chunk from memory")
				else:
					logger.warning(
						f"No records found for chunk {current_time} to {chunk_end}")

			except Exception as e:
				logger.error(
					f"Failed to process chunk {current_time} to {chunk_end}: {str(e)}")
				continue

			current_time = chunk_end

		return True

	except Exception as e:
		logger.error(f"Error processing ADSB data: {str(e)}")
		return False


################################################################################
# test run function
################################################################################

def test_run(
	start_time: datetime = datetime(2024, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
	end_time: datetime = datetime(2024, 8, 1, 0, 15, 0, tzinfo=timezone.utc),
	chunk_size: int = 50,
	max_workers: int = 32
) -> bool:
	"""
	Main execution function for ADSB data processing.

	Args:
	   start_time: Start time for data processing
	   end_time: End time for data processing
	   chunk_size: Number of files to process in each chunk
	   max_workers: Maximum number of parallel workers

	Returns:
	   bool: True if processing successful, False otherwise
	"""
	try:
		logger.info(
			f"Starting processing with parameters: start_time={start_time}, "
			f"end_time={end_time}, chunk_size={chunk_size}, max_workers={max_workers}")

		total_intervals = int((end_time - start_time).total_seconds() / 300)
		logger.info(f"Calculated {total_intervals} total intervals to process")

		current_time = start_time
		process_start_time = time.time()
		successful_chunks = 0
		failed_chunks = 0
		total_records_processed = 0

		for chunk_start_idx in tqdm(range(0, total_intervals, chunk_size),
									desc="Processing chunks"):
			chunk_end_idx = min(chunk_start_idx + chunk_size, total_intervals)
			chunk_end = start_time + timedelta(minutes=5 * chunk_end_idx)

			chunk_start_time = time.time()
			logger.info(
				f"Starting chunk {chunk_start_idx // chunk_size + 1} of "
				f"{(total_intervals + chunk_size - 1) // chunk_size}")
			logger.info(f"Processing time range: {current_time} to {chunk_end}")

			try:
				logger.debug(f"Memory usage before extraction: "
							 f"{psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

				chunk_records = extract_adsb_data(current_time, chunk_end,
												  max_workers)

				if chunk_records:
					record_count = len(chunk_records)
					logger.info(f"Extracted {record_count} records from chunk")
					total_records_processed += record_count

					load_start_time = time.time()
					load_records_to_db(chunk_records)
					logger.info(f"Database load completed in "
								f"{time.time() - load_start_time:.2f} seconds")

					del chunk_records
					gc.collect()
					logger.debug(f"Memory usage after cleanup: "
								 f"{psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
					successful_chunks += 1

				else:
					logger.warning(
						f"No records found for chunk {current_time} to {chunk_end}")
					failed_chunks += 1

				logger.info(f"Chunk processing completed in "
							f"{time.time() - chunk_start_time:.2f} seconds")

			except Exception as e:
				failed_chunks += 1
				logger.error(
					f"Failed to process chunk {current_time} to {chunk_end}: {str(e)}")
				logger.debug(f"Stack trace: ", exc_info=True)
				continue

			current_time = chunk_end

		total_time = time.time() - process_start_time
		logger.info(f"Processing completed in {total_time:.2f} seconds")
		logger.info(f"Results summary:")
		logger.info(f"- Total records processed: {total_records_processed}")
		logger.info(f"- Successful chunks: {successful_chunks}")
		logger.info(f"- Failed chunks: {failed_chunks}")
		logger.info(f"- Average processing time per chunk: "
					f"{total_time / (successful_chunks + failed_chunks):.2f} seconds")

		return True

	except Exception as e:
		logger.error(f"Critical error processing ADSB data: {str(e)}")
		logger.error("Stack trace: ", exc_info=True)
		return False


################################################################################
# main guard
################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process ADSB Exchange data')
	parser.add_argument(
		'--start-time',
		type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H').replace(
			tzinfo=timezone.utc),
		default=datetime(2024, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
		help='Start time (YYYY-MM-DD-HH)'
	)
	parser.add_argument(
		'--end-time',
		type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H').replace(
			tzinfo=timezone.utc),
		default=datetime(2024, 8, 2, 0, 0, 0, tzinfo=timezone.utc),
		help='End time (YYYY-MM-DD-HH)'
	)
	parser.add_argument(
		'--chunk-size',
		type=int,
		default=100,
		help='Number of files to process in each chunk'
	)
	parser.add_argument(
		'--max-workers',
		type=int,
		default=32,
		help='Maximum number of parallel workers'
	)

	args = parser.parse_args()
	success = run(**vars(args))
	exit(0 if success else 1)

################################################################################
# end of adsb_etl.py
################################################################################