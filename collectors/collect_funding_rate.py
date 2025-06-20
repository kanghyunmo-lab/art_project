# Standard library imports
import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone # timedelta, timezone 추가 (historical_funding_rate 함수에서 사용)
import argparse

# Load .env file first
from dotenv import load_dotenv

# --- Project Setup ---
# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# sys.path 수정 및 디버깅
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Add to the beginning for priority

# .env 파일 로드 (PROJECT_ROOT 정의 후, config 임포트 전)
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# --- Project-specific Imports ---
try:
    from utils.decorators import retry_on_request_exception
except ImportError as e:
    # Define a dummy decorator if import fails, to prevent NameError later, or re-raise
    def retry_on_request_exception(func): return func # Dummy placeholder
    # raise # Uncomment to halt execution if this import is critical and fails

try:
    from config.config import (
        INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, # Should be loaded from .env by config.py
        INFLUXDB_PARAMS, DATA_PARAMS, RETRY_CONFIG, FUNDING_RATE_COLLECTOR_PARAMS
    )
    # Verify if InfluxDB credentials from .env were loaded into these variables via config.py
except ImportError as e:
    # Define placeholders or raise error if config is critical
    INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG = None, None, None
    INFLUXDB_PARAMS, DATA_PARAMS, RETRY_CONFIG, FUNDING_RATE_COLLECTOR_PARAMS = {}, {}, {}, {}
    # raise # Uncomment to halt execution if config import is critical and fails

# --- Third-party Library Imports ---
import requests
from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.rest import ApiException

# --- Custom Exception Definition ---
# This was added in a previous step to this file. Ideally, move to utils/exceptions.py
class BinanceApiLogicalError(Exception):
    """바이낸스 API가 논리적 오류를 반환할 때 발생하는 사용자 정의 예외입니다."""
    pass


# 로깅 설정
log_path = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'funding_rate_collector.log')),
        logging.StreamHandler()
    ]
)

# .env 파일에서 환경 변수 로드
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# InfluxDB 연결 정보
INFLUXDB_URL = os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')

# 설정 파일에서 펀딩비 수집기 파라미터 로드
from config.config import FUNDING_RATE_COLLECTOR_PARAMS

# Binance API base URL (선물)
BASE_URL = "https://fapi.binance.com"
FUNDING_RATE_HISTORY_ENDPOINT = "/fapi/v1/fundingRate"

class BinanceApiLogicalError(requests.exceptions.RequestException):
    """Custom exception for Binance API logical errors (e.g., rate limits with 200 OK but error in body)."""
    def __init__(self, message, code=None, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.code = code

@retry_on_request_exception
def get_funding_rate_history(symbol, start_time=None, end_time=None, limit=FUNDING_RATE_COLLECTOR_PARAMS['default_limit']):
    params = {
        'symbol': symbol,
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    try:
        response = requests.get(BASE_URL + FUNDING_RATE_HISTORY_ENDPOINT, params=params)
        response.raise_for_status()  # HTTP 4xx/5xx 오류 발생 시 예외 발생
        
        data = response.json()

        # Binance API의 경우, HTTP 200 OK를 반환하면서도 내용에 오류 코드를 포함할 수 있음
        # 예: { "code": -1121, "msg": "Invalid symbol." }
        # 예: { "code": -4003, "msg": "Quantity less than zero." } -> 이런건 재시도 의미 없음
        # 예: { "code": -1003, "msg": "Too many requests." } -> 이런건 재시도 필요
        if isinstance(data, dict) and 'code' in data and data['code'] != 0:
            error_code = data.get('code')
            error_msg = data.get('msg', 'Unknown Binance API error')
            logging.error(f"Binance API returned an error for {symbol}: Code {error_code}, Msg: {error_msg}")
            if error_code == -1121: # Invalid symbol, 재시도 불필요
                raise ValueError(f"Invalid symbol: {symbol}. Binance API error: {error_msg} (Code: {error_code})")
            # 다른 특정 재시도 불필요 코드가 있다면 여기에 추가
            # 그 외의 논리적 오류는 BinanceApiLogicalError로 감싸서 데코레이터가 재시도하도록 함
            raise BinanceApiLogicalError(f"Binance API logical error: {error_msg} (Code: {error_code})", code=error_code, response=response)

        if not isinstance(data, list):
            logging.error(f"Unexpected data format from Binance API for {symbol}. Expected list, got {type(data)}. Data: {data}")
            # 이 경우도 재시도 대상이 될 수 있도록 BinanceApiLogicalError 발생
            raise BinanceApiLogicalError(f"Unexpected data format from Binance API for {symbol}. Expected list, got {type(data)}.", response=response)

        logging.info(f"{symbol} 펀딩 비율 API 호출 성공. {len(data)}개 데이터 수신.")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"{symbol} 펀딩 비율 API 요청 중 오류 발생: {e}")
        return None

@retry_on_request_exception
def write_to_influxdb(data, symbol):
    if not INFLUXDB_URL or not INFLUXDB_TOKEN or not INFLUXDB_ORG:
        logging.error("InfluxDB 연결 정보가 .env 파일에 설정되지 않았거나 config에서 로드되지 않았습니다.")
        # 이 경우 재시도해도 소용없으므로, 심각한 설정 오류로 간주하고 예외 발생 또는 특정 값 반환
        raise ConnectionError("InfluxDB configuration is missing.")

    bucket = INFLUXDB_PARAMS['funding_rate_bucket']
    measurement = INFLUXDB_PARAMS['funding_rate_measurement']
    points_written_count = 0

    if not data:
        logging.info(f"No data to write for {symbol}.")
        return points_written_count

    points = []
    for record in data:
        if not all(k in record for k in ["fundingRate", "fundingTime"]):
            logging.warning(f"Skipping record due to missing essential fields: {record} for symbol {symbol}")
            continue
        try:
            funding_rate = float(record["fundingRate"])
            funding_time_ms = int(record["fundingTime"])
            # markPrice는 선택적일 수 있으므로, 없으면 0.0 또는 None으로 처리
            mark_price_str = record.get("markPrice")
            mark_price = float(mark_price_str) if mark_price_str is not None else 0.0 
        except (ValueError, TypeError) as ve:
            logging.warning(f"Skipping record due to data conversion error: {ve} - {record} for symbol {symbol}")
            continue

        point = (
            Point(measurement)
            .tag("symbol", symbol)
            .field("fundingRate", funding_rate)
            .field("markPrice", mark_price) # markPrice 필드가 없을 수도 있음을 인지
            .time(funding_time_ms, WritePrecision.MS)
        )
        points.append(point)

    if not points:
        logging.info(f"No valid points to write for {symbol} after filtering.")
        return points_written_count

    try:
        # InfluxDBClient는 with 문으로 관리하는 것이 좋으나, 데코레이터와 함께 사용 시 복잡해질 수 있음.
        # 여기서는 함수 호출 시마다 client를 생성하는 대신, 모듈 레벨에서 client를 관리하거나,
        # main 함수 등에서 한 번만 초기화하고 넘겨주는 방식을 고려할 수 있음.
        # 현재 코드는 모듈 레벨에 client가 없으므로, 함수 내에서 생성.
        # 하지만 기존 코드(view_file 결과)에서는 with InfluxDBClient(...)를 사용하고 있었음.
        # 데코레이터와 함께 쓰려면, client를 인자로 받거나, 데코레이터가 client 생성을 관리해야 함.
        # 여기서는 기존 구조를 최대한 유지하며, 예외 발생 시 데코레이터가 잡도록 함.
        # 주의: 이 방식은 매번 연결을 생성/해제하므로 비효율적일 수 있음.
        # TODO: InfluxDBClient 인스턴스 관리 최적화 필요.
        with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            # 버킷 존재 확인 및 생성 로직은 애플리케이션 시작 시 한 번만 수행하는 것이 효율적.
            # 매번 write_to_influxdb 호출 시마다 확인하는 것은 비효율적.
            # 여기서는 일단 기존 로직을 유지하되, 개선 필요성을 주석으로 남김.
            # TODO: 버킷 확인/생성 로직을 애플리케이션 초기화 단계로 이동.
            buckets_api = client.buckets_api()
            try:
                found_bucket = buckets_api.find_bucket_by_name(bucket_name=bucket)
                if not found_bucket:
                    logging.info(f"InfluxDB 버킷 '{bucket}'이(가) '{INFLUXDB_ORG}' 조직 내에 존재하지 않습니다. 새로 생성합니다.")
                    retention_rules = BucketRetentionRules(type="expire", every_seconds=365 * 24 * 60 * 60) # 365일 보존
                    buckets_api.create_bucket(bucket_name=bucket, org=INFLUXDB_ORG, retention_rules=retention_rules)
                    logging.info(f"InfluxDB 버킷 '{bucket}'이(가) '{INFLUXDB_ORG}' 조직 내에 성공적으로 생성되었습니다.")
            except ApiException as e:
                if e.status == 422 and "name conflicts with an existing bucket" in str(e.body).lower():
                    logging.warning(f"InfluxDB 버킷 '{bucket}' 생성 시도 중 이미 존재(422). 계속 진행.")
                elif e.status == 404:
                    logging.error(f"InfluxDB 버킷 '{bucket}' 작업 중 조직/버킷 찾을 수 없음 (404): {e.body}. INFLUXDB_ORG ('{INFLUXDB_ORG}') 확인.")
                    raise requests.exceptions.ConnectionError(f"InfluxDB org/bucket not found: {e.body}") # 재시도 유도
                else:
                    logging.error(f"InfluxDB 버킷 확인/생성 API 오류: {e.body if e.body else e}")
                    raise requests.exceptions.ConnectionError(f"InfluxDB API error during bucket check/create: {e.body if e.body else e}") # 재시도 유도
            except Exception as e:
                logging.error(f"InfluxDB 버킷 확인/생성 일반 오류: {e}")
                raise requests.exceptions.ConnectionError(f"InfluxDB general error during bucket check/create: {e}") # 재시도 유도

            write_api = client.write_api(write_options=SYNCHRONOUS)
            write_api.write(bucket=bucket, org=INFLUXDB_ORG, record=points)
            points_written_count = len(points)
            logging.info(f"InfluxDB에 {points_written_count}개 {symbol} 펀딩 비율 데이터 쓰기 완료. 버킷: {bucket}")
            return points_written_count

    except (ApiException, InfluxDBError) as e: # InfluxDB 관련 주요 예외
        # ApiException은 HTTP 오류 정보를 포함할 수 있음
        status_code = e.status if hasattr(e, 'status') else None
        if status_code and status_code in RETRY_CONFIG.get("retry_http_status_codes", []):
             logging.warning(f"Retriable InfluxDB API 에러 (status {status_code}) 발생: {e}. 데코레이터가 재시도합니다.")
             # HTTPError로 변환하여 데코레이터가 HTTP 상태 코드 기반 재시도 로직을 타도록 함
             http_error = requests.exceptions.HTTPError(str(e), response=requests.Response())
             http_error.response.status_code = status_code # status_code 설정
             raise http_error
        else:
            logging.error(f"InfluxDB 쓰기 중 심각한 오류 발생 (status {status_code if status_code else 'N/A'}): {e}")
            # 재시도 불가능한 InfluxDB 오류는 ConnectionError로 감싸서 데코레이터가 일반 RequestException으로 처리하도록 함
            # 또는, 특정 오류는 그대로 raise하여 상위에서 처리하도록 할 수도 있음
            raise requests.exceptions.ConnectionError(f"Non-retriable InfluxDB error: {e}")
    except Exception as e: # 그 외 일반 예외 (예: 네트워크 문제 등)
        logging.error(f"InfluxDB 쓰기 중 예기치 않은 오류 발생: {e}")
        # 일반 예외도 ConnectionError로 감싸서 재시도 유도
        raise requests.exceptions.ConnectionError(f"Unexpected error during InfluxDB write: {e}")

def collect_historical_funding_rate(symbol, start_date_str, end_date_str):
    logging.info(f"과거 펀딩 비율 데이터 수집 시작: {symbol} ({start_date_str} ~ {end_date_str})")
    try:
        # 날짜 문자열을 datetime 객체로 변환 (UTC 기준)
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError as e:
        logging.error(f"날짜 형식 오류 ({start_date_str}, {end_date_str}): {e}. 'YYYY-MM-DD' 형식을 사용하세요.")
        return 0

    current_start_dt = start_dt
    # end_dt는 해당 날짜의 시작이므로, 실제로는 end_dt + 1일까지의 데이터를 포함해야 함
    # API는 endTime을 exclusive하게 처리하므로, end_dt의 다음 날 00:00:00 이전까지 수집
    effective_end_dt = end_dt + timedelta(days=1)
    
    total_records_collected_session = 0
    api_call_delay = FUNDING_RATE_COLLECTOR_PARAMS.get("api_call_delay_seconds", 0.5)
    historical_api_limit = FUNDING_RATE_COLLECTOR_PARAMS.get("historical_limit", 1000)
    max_consecutive_fetch_failures = 3 # 연속 API/DB 실패 시 해당 시간대를 건너뛰기 위한 임계값
    consecutive_failures = 0

    while current_start_dt < effective_end_dt:
        # API 호출 전 지연 (필수)
        time.sleep(api_call_delay)

        # API는 밀리초 타임스탬프를 사용
        current_start_timestamp_ms = int(current_start_dt.timestamp() * 1000)
        
        # endTime은 다음 루프의 시작점이 되거나, effective_end_dt를 넘지 않도록 설정
        # 바이낸스 API는 한 번에 최대 1000개까지 가져오므로, endTime을 명시하는 것이 좋음
        # 여기서는 endTime을 명시하지 않고, limit에 의존하여 다음 current_start_dt를 계산
        # 이는 get_funding_rate_history가 start_time만 받고 limit만큼 가져오는 현재 구조에 맞춤
        # 만약 get_funding_rate_history가 endTime도 받는다면, 더 정교한 endTime 설정 가능

        logging.info(f"Fetching data for {symbol} starting from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} (Timestamp: {current_start_timestamp_ms})")
        
        data = None
        try:
            data = get_funding_rate_history(
                symbol=symbol,
                start_time=current_start_timestamp_ms, # datetime 객체가 아닌 ms timestamp 전달
                # end_time=current_end_timestamp_ms, # 필요하다면 endTime도 계산하여 전달
                limit=historical_api_limit
            )
            # API 호출 성공 시 연속 실패 카운트 초기화
            # (get_funding_rate_history 내부에서 모든 재시도 성공 시)
            # 만약 get_funding_rate_history가 예외를 던지면 이 부분은 실행되지 않음

            if data:
                points_written = write_to_influxdb(data, symbol)
                if points_written > 0:
                    total_records_collected_session += points_written
                    # 다음 시작 시간 설정: 마지막으로 가져온 데이터의 시간 + 1ms
                    last_funding_time_ms = int(data[-1]["fundingTime"])
                    current_start_dt = datetime.fromtimestamp(last_funding_time_ms / 1000, tz=timezone.utc) + timedelta(milliseconds=1)
                    consecutive_failures = 0 # 데이터 수집 및 쓰기 성공 시 초기화
                else:
                    # DB 쓰기 실패 (모든 재시도 후) 또는 유효한 데이터 없음
                    logging.warning(f"No data written to DB for {symbol} for period starting {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}. Data was fetched but not written.")
                    # 이 경우, current_start_dt를 업데이트하지 않아 다음 루프에서 동일 시간대 재시도.
                    # 하지만 무한 루프를 피하기 위해 consecutive_failures를 증가시키고, 임계값 도달 시 current_start_dt를 강제 이동.
                    consecutive_failures += 1

                if len(data) < historical_api_limit:
                    logging.info(f"Fetched less than limit ({len(data)} < {historical_api_limit}) for {symbol}. Assuming end of available data for this period or reaching current time.")
                    # 다음 루프에서 current_start_dt (이미 업데이트됨)으로 계속 진행
            else:
                # API가 빈 데이터를 반환 (정상적일 수 있음, 해당 기간 데이터 없음)
                logging.info(f"No data returned by API for {symbol} for period starting {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}. Advancing to next possible time slot.")
                # 다음 가능한 펀딩 시간으로 이동 (보통 8시간 간격). 여기서는 1시간씩 증가시켜 다음 시도.
                current_start_dt += timedelta(hours=1)
                consecutive_failures = 0 # 빈 데이터는 오류가 아니므로 실패 카운트 초기화
        
        except (requests.exceptions.RequestException, BinanceApiLogicalError, ValueError, ConnectionError) as e:
            # get_funding_rate_history 또는 write_to_influxdb에서 모든 재시도 실패 후 발생한 예외
            logging.error(f"Failed to collect or write data for {symbol} starting {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} after all retries: {type(e).__name__}: {e}")
            consecutive_failures += 1
        
        finally:
            if consecutive_failures >= max_consecutive_fetch_failures:
                logging.error(f"Max consecutive failures ({max_consecutive_fetch_failures}) reached for {symbol} at {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}. Skipping this 8-hour window.")
                current_start_dt += timedelta(hours=8) # 8시간 건너뛰기
                consecutive_failures = 0 # 카운터 초기화
            elif consecutive_failures > 0 and not data : # API 호출 실패 후 (data is None or empty) 다음 시간으로 이동
                 # write_to_influxdb 실패 시에는 current_start_dt가 업데이트되지 않아 동일 시간 재시도 유도 (위에서 처리)
                 # get_funding_rate_history 실패 시에는 current_start_dt를 증가시켜야 함
                logging.info(f"Advancing current_start_dt by 1 hour for {symbol} due to error. Consecutive failures: {consecutive_failures}")
                current_start_dt += timedelta(hours=1)

        if current_start_dt >= effective_end_dt:
            logging.info(f"Historical collection for {symbol} reached or passed end_date ({effective_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
            break

    logging.info(f"{symbol}에 대한 과거 데이터 수집 완료. 총 {total_records_collected_session}개의 데이터 포인트 수집됨.")
    return total_records_collected_session


def main():
    parser = argparse.ArgumentParser(description="Binance Funding Rate Collector for InfluxDB.")

    parser.add_argument(
        '--symbol',
        type=str,
        default=FUNDING_RATE_COLLECTOR_PARAMS['default_symbol'],
        help=f"Target symbol to collect (e.g., BTCUSDT). Default: {FUNDING_RATE_COLLECTOR_PARAMS['default_symbol']}"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=FUNDING_RATE_COLLECTOR_PARAMS['default_mode'],
        choices=['recent', 'historical'],
        help="Collector mode: 'recent' for latest data, 'historical' for backfilling. Default: 'recent'"
    )
    parser.add_argument(
        '--start_date',
        type=str,
        default=None,
        help="Start date for historical collection (YYYY-MM-DD). Required if mode is 'historical'."
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default=None,
        help="End date for historical collection (YYYY-MM-DD). Required if mode is 'historical'."
    )
    parser.add_argument(
        '--limit',
        type=int,
        # 'default_limit' 대신 'recent_limit' 사용 또는 argparse에서 기본값을 설정하고 config 값은 내부 로직에서 참조
        default=FUNDING_RATE_COLLECTOR_PARAMS.get('recent_limit', 10), # config.py에 recent_limit이 없을 경우 대비
        help=f"Number of recent records to fetch in 'recent' mode. Default from config: {FUNDING_RATE_COLLECTOR_PARAMS.get('recent_limit', 10)}"
    )

    args = parser.parse_args()

    symbol_to_fetch = args.symbol.upper() # 심볼은 대문자로 통일

    logging.info(f"Funding rate collection process started for symbol: {symbol_to_fetch}, mode: {args.mode}")

    if args.mode == 'recent':
        logging.info(f"Collecting recent {args.limit} funding rate(s) for {symbol_to_fetch}...")
        try:
            # recent 모드에서는 start_time, end_time을 None으로 전달
            funding_history = get_funding_rate_history(symbol_to_fetch, start_time=None, end_time=None, limit=args.limit)
            if funding_history:
                logging.info(f"Fetched {len(funding_history)} records for recent mode.")
                points_written = write_to_influxdb(funding_history, symbol_to_fetch)
                if points_written > 0:
                    logging.info(f"Successfully wrote {points_written} recent funding rates for {symbol_to_fetch} to InfluxDB.")
                else:
                    logging.warning(f"No points written to InfluxDB for recent funding rates of {symbol_to_fetch}. This might be due to write failures or no valid data.")
            else:
                # get_funding_rate_history가 None을 반환하거나 빈 리스트를 반환하는 경우 (모든 재시도 실패 또는 데이터 없음)
                logging.warning(f"Failed to fetch recent funding rates for {symbol_to_fetch} after all retries, or no data available.")
        except (requests.exceptions.RequestException, BinanceApiLogicalError, ValueError, ConnectionError) as e:
            logging.error(f"Critical error during recent data collection for {symbol_to_fetch}: {type(e).__name__}: {e}")

    elif args.mode == 'historical':
        if not (args.start_date and args.end_date):
            logging.error("Error: --start_date and --end_date are required for historical mode.")
            parser.print_help()
            return
        logging.info(f"Collecting historical funding rates for {symbol_to_fetch} from {args.start_date} to {args.end_date}...")
        collect_historical_funding_rate(symbol_to_fetch, args.start_date, args.end_date)
    
    logging.info(f"Funding rate collection process finished for symbol: {symbol_to_fetch}, mode: {args.mode}")

if __name__ == "__main__":
    main()
