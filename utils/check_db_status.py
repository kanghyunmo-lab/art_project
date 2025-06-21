# utils/check_db_status.py
import os
import sys
import logging
from datetime import datetime

# --- Project Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project-specific Imports ---
try:
    from config.config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_PARAMS
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.exceptions import InfluxDBError
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_latest_funding_rate():
    """Checks and prints the latest funding rate entry in InfluxDB."""
    if not all([INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG]):
        logging.error("InfluxDB connection details are not configured. Please check your .env or config.py.")
        return

    bucket = INFLUXDB_PARAMS.get('funding_rate_bucket')
    measurement = INFLUXDB_PARAMS.get('funding_rate_measurement')

    if not bucket or not measurement:
        logging.error("InfluxDB bucket or measurement for funding rates is not configured in config.py.")
        return

    try:
        with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            query_api = client.query_api()
            
            flux_query = f'''
            from(bucket: "{bucket}")
              |> range(start: 0)
              |> filter(fn: (r) => r._measurement == "{measurement}")
              |> last()
            '''
            
            logging.info(f"Querying InfluxDB for the latest record in bucket '{bucket}'...")
            tables = query_api.query(flux_query)

            if not tables:
                logging.warning(f"No data found in measurement '{measurement}' of bucket '{bucket}'.")
                return

            for table in tables:
                for record in table.records:
                    latest_time = record.get_time()
                    latest_value = record.get_value()
                    symbol = record.values.get('symbol', 'N/A')
                    logging.info("--- Latest Funding Rate Entry ---")
                    logging.info(f"  Symbol: {symbol}")
                    logging.info(f"  Time: {latest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    logging.info(f"  Rate: {latest_value}")
                    logging.info("---------------------------------")
                    return # Exit after printing the first (and only) result

    except InfluxDBError as e:
        logging.error(f"An error occurred with InfluxDB: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_latest_funding_rate()
