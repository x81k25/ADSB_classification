import psycopg2
import itertools
import string
import logging
from datetime import datetime
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os
import sys

################################################################################
# initial setup and parameters
################################################################################

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

################################################################################
# supporting functions
################################################################################

def generate_hex_prefixes(start_hex='00', end_hex='ff'):
	"""
	Generate hex combinations between start_hex and end_hex inclusive.
	All values are handled as lowercase.
	"""
	start_hex = start_hex.lower()
	end_hex = end_hex.lower()

	# Validate hex inputs
	if not all(c in string.hexdigits for c in start_hex + end_hex):
		raise ValueError("Invalid hex value provided")
	if len(start_hex) != 2 or len(end_hex) != 2:
		raise ValueError("Hex values must be exactly 2 characters")

	# Convert hex to numbers for comparison
	start_num = int(start_hex, 16)
	end_num = int(end_hex, 16)

	if start_num > end_num:
		raise ValueError(
			f"Start hex {start_hex} is greater than end hex {end_hex}")

	return [format(i, '02x') for i in range(start_num, end_num + 1)]


def check_table_dependencies(cursor, table_name):
	"""Check if table has any dependencies before dropping."""
	cursor.execute("""
        SELECT DISTINCT dependent_ns.nspname as dependent_schema,
                      dependent_view.relname as dependent_view
        FROM pg_depend 
        JOIN pg_rewrite ON pg_depend.objid = pg_rewrite.oid 
        JOIN pg_class as dependent_view ON pg_rewrite.ev_class = dependent_view.oid 
        JOIN pg_class as source_table ON pg_depend.refobjid = source_table.oid 
        JOIN pg_namespace dependent_ns ON dependent_view.relnamespace = dependent_ns.oid 
        JOIN pg_namespace source_ns ON source_table.relnamespace = source_ns.oid 
        WHERE source_table.relname = %s
    """, (table_name,))
	return cursor.fetchall()


def create_sharded_table(cursor, hex_prefix):
	"""Create a new table for the given hex prefix."""
	table_name = f'aircraft_traces_{hex_prefix}'

	logger.info(f"Creating table {table_name}")

	# Drop table if exists
	deps = check_table_dependencies(cursor, table_name)
	if deps:
		logger.warning(f"Table {table_name} has dependencies: {deps}")

	cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

	# Create new table
	cursor.execute(f"""
        CREATE TABLE {table_name} (
            icao CHAR(6) NOT NULL,
            actual_timestamp TIMESTAMPTZ NOT NULL,
            base_timestamp BIGINT NOT NULL,
            seconds_offset DECIMAL(10,2) NOT NULL,
            latitude DECIMAL(9,6),
            longitude DECIMAL(9,6),
            altitude_baro INTEGER,
            ground_speed DECIMAL(6,1),
            track DECIMAL(6,2),
            flags INTEGER,
            vertical_rate_baro INTEGER,
            altitude_geom INTEGER,
            vertical_rate_geom INTEGER,
            indicated_airspeed INTEGER,
            roll_angle DECIMAL(6,2),
            source_type VARCHAR(20),
            source_file_path TEXT,
            flight VARCHAR(20),
            squawk VARCHAR(4),
            category VARCHAR(2),
            nav_qnh DECIMAL(6,1),
            nav_altitude_mcp INTEGER,
            nav_heading DECIMAL(6,2),
            emergency VARCHAR(20),
            nic INTEGER,
            rc INTEGER,
            version INTEGER,
            nic_baro INTEGER,
            nac_p INTEGER,
            nac_v INTEGER,
            sil INTEGER,
            sil_type VARCHAR(20),
            gva INTEGER,
            sda INTEGER,
            PRIMARY KEY (icao, actual_timestamp)
        )
    """)

	return table_name


def transfer_data(cursor, hex_prefix, batch_size=100000):
	"""Transfer data for the given hex prefix in batches."""
	source_table = 'aircraft_traces'
	target_table = f'aircraft_traces_{hex_prefix}'

	# Get total count for this prefix
	cursor.execute(f"""
        SELECT COUNT(*) 
        FROM {source_table} 
        WHERE icao LIKE '{hex_prefix}%'
        AND icao NOT LIKE '~%'
    """)
	total_rows = cursor.fetchone()[0]
	logger.info(f"Total rows to transfer for {hex_prefix}: {total_rows}")

	# Transfer in batches
	offset = 0
	while True:
		logger.info(f"Transferring batch for {hex_prefix} at offset {offset}")
		cursor.execute(f"""
            INSERT INTO {target_table}
            SELECT * FROM (
                SELECT 
                    SUBSTRING(icao FROM 1 FOR 6) as icao,
                    actual_timestamp,
                    base_timestamp,
                    seconds_offset,
                    latitude,
                    longitude,
                    altitude_baro,
                    ground_speed,
                    track,
                    flags,
                    vertical_rate_baro,
                    altitude_geom,
                    vertical_rate_geom,
                    indicated_airspeed,
                    roll_angle,
                    source_type,
                    source_file_path,
                    flight,
                    squawk,
                    category,
                    nav_qnh,
                    nav_altitude_mcp,
                    nav_heading,
                    emergency,
                    nic,
                    rc,
                    version,
                    nic_baro,
                    nac_p,
                    nac_v,
                    sil,
                    sil_type,
                    gva,
                    sda
                FROM {source_table}
                WHERE icao LIKE '{hex_prefix}%'
                AND icao NOT LIKE '~%'
                ORDER BY actual_timestamp
                LIMIT {batch_size}
                OFFSET {offset}
            ) sub
        """)

		rows_inserted = cursor.rowcount
		if rows_inserted == 0:
			break

		offset += batch_size
		logger.info(f"Inserted {rows_inserted} rows for {hex_prefix}")

	return total_rows


def create_indexes(cursor, table_name):
	"""Create indexes on the new table."""
	logger.info(f"Creating indexes for {table_name}")
	cursor.execute(f"""
        CREATE INDEX idx_traces_timestamp_{table_name.split('_')[-1]} 
        ON {table_name} (actual_timestamp)
    """)
	cursor.execute(f"""
        CREATE INDEX idx_traces_icao_{table_name.split('_')[-1]} 
        ON {table_name} (icao)
    """)


def vacuum_table(conn, table_name):
	"""VACUUM the table to reclaim space."""
	# Need to be in autocommit mode for VACUUM
	old_isolation_level = conn.isolation_level
	conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

	cursor = conn.cursor()
	logger.info(f"VACUUMing table {table_name}")
	cursor.execute(f"VACUUM {table_name}")

	conn.set_isolation_level(old_isolation_level)

################################################################################
# main function
################################################################################

def main(start_hex='00', end_hex='ff'):
	# Load environment variables
	load_dotenv()

	# Database connection parameters from environment
	db_params = {
		'dbname': os.getenv('DB_NAME'),
		'user': os.getenv('DB_USER'),
		'password': os.getenv('DB_PASSWORD'),
		'host': os.getenv('DB_HOST'),
		'port': os.getenv('DB_PORT')
	}

	# Validate all required environment variables are present
	missing_vars = [k for k, v in db_params.items() if not v]
	if missing_vars:
		logger.error(
			f"Missing required environment variables: {', '.join(missing_vars)}")
		sys.exit(1)

	start_time = datetime.now()
	logger.info(
		f"Starting table sharding from {start_hex} to {end_hex} at {start_time}")

	try:
		conn = psycopg2.connect(**db_params)

		# Set schema if provided
		if os.getenv('DB_SCHEMA'):
			with conn.cursor() as cursor:
				cursor.execute(f"SET search_path TO {os.getenv('DB_SCHEMA')}")
				conn.commit()

		for hex_prefix in generate_hex_prefixes(start_hex, end_hex):
			logger.info(f"\nProcessing prefix {hex_prefix}")
			prefix_start_time = datetime.now()

			try:
				with conn.cursor() as cursor:
					# Create new table
					table_name = create_sharded_table(cursor, hex_prefix)
					conn.commit()

					# Transfer data
					total_rows = transfer_data(cursor, hex_prefix)
					conn.commit()

					# Create indexes
					create_indexes(cursor, table_name)
					conn.commit()

					# VACUUM
					vacuum_table(conn, table_name)

					prefix_end_time = datetime.now()
					duration = prefix_end_time - prefix_start_time
					logger.info(
						f"Completed {hex_prefix} in {duration}. Rows: {total_rows}")

			except Exception as e:
				logger.error(f"Error processing {hex_prefix}: {str(e)}")
				conn.rollback()
				continue

	except Exception as e:
		logger.error(f"Database connection error: {str(e)}")
		sys.exit(1)
	finally:
		if 'conn' in locals():
			conn.close()

	end_time = datetime.now()
	total_duration = end_time - start_time
	logger.info(f"Completed table sharding in {total_duration}")

if __name__ == "__main__":
	if len(sys.argv) == 3:
		main(sys.argv[1], sys.argv[2])
	elif len(sys.argv) == 1:
		main()  # Use default range 00-ff
	else:
		print("Usage: python script.py [start_hex end_hex]")
		print("Example: python script.py 00 0f")
		print("If no arguments provided, processes full range (00-ff)")
		sys.exit(1)

################################################################################
# end of _06_traces_db_partitioning.py
################################################################################