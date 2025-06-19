print("--- LOADING LATEST collector.py ---")
import logging
logger = logging.getLogger(__name__)
logger.info("--- collector.py 스크립트 시작됨 ---")
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import time

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TelegramNotifier 임포트
from notifications.telegram_bot import TelegramNotifier

# API 키는 config 파일에서 로드 (존재하지 않을 경우를 대비한 예외 처리 포함)
try:
    from config.credentials import (
        BINANCE_API_KEY, BINANCE_API_SECRET, 
        INFLUXDB_TOKEN, INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET
    )
except ImportError:
    print("Warning: credentials.py not found. Using dummy keys. Please create config/credentials.py")
    BINANCE_API_KEY = "YOUR_API_KEY"
    BINANCE_API_SECRET = "YOUR_API_SECRET"
    INFLUXDB_TOKEN = "YOUR_INFLUX_TOKEN"
    INFLUXDB_URL = "http://localhost:8086"
    INFLUXDB_ORG = "your-org"
    INFLUXDB_BUCKET = "crypto-data" # 기본값

class BinanceDataCollector:
    """
    바이낸스로부터 과거 및 실시간 데이터를 수집하고 InfluxDB에 저장합니다.
    """
    def __init__(self):
        """
        Binance 및 InfluxDB 클라이언트를 초기화합니다.
        """
        self.binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        try:
            self.influx_client = influxdb_client.InfluxDBClient(
                url=INFLUXDB_URL, 
                token=INFLUXDB_TOKEN, 
                org=INFLUXDB_ORG,
                timeout=30_000  # 타임아웃을 30초로 설정
            )
            self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.influx_bucket = INFLUXDB_BUCKET
            # InfluxDB 연결 테스트 (선택 사항, 서버 다운 시 빠른 실패를 위해)
            self.influx_client.ping()
            logger.info("InfluxDB에 성공적으로 연결되었습니다.")
        except Exception as e:
            logger.error(f"InfluxDB 연결 오류: {e}. InfluxDB 관련 작업이 비활성화됩니다.")
            self.influx_client = None
            self.influx_write_api = None
            self.influx_bucket = None

    def fetch_historical_data(self, symbol, interval, start_str, end_str=None):
        """
        지정된 기간의 과거 OHLCV 데이터를 바이낸스에서 가져옵니다.
        (에러 핸들링 강화)
        :param symbol: (str) 거래 쌍 (예: 'BTCUSDT')
        :param interval: (str) 데이터 간격 (예: '1h', '4h', '1d')
        :param start_str: (str) 시작 날짜 (예: '1 Jan, 2020', '2022-01-01')
        :param end_str: (str, optional) 종료 날짜. 기본값은 None (현재까지).
        :return: (pd.DataFrame or None) OHLCV 데이터 또는 실패 시 None
        """
        print(f"Fetching historical data for {symbol} from {start_str} to {end_str or 'now'}...")
        try:
            klines = self.binance_client.get_historical_klines(symbol, interval, start_str, end_str)
            if not klines:
                logger.warning(f"No data returned from Binance for {symbol} with interval {interval}.")
                return pd.DataFrame() # 빈 데이터프레임 반환

        except BinanceAPIException as e:
            logger.error(f"Binance API 에러 발생 (symbol={symbol}, interval={interval}): {e}")
            # 흔한 에러에 대한 사용자 친화적 메시지 추가
            if e.code == -1121:
                logger.error("API 에러 코드 -1121: 잘못된 심볼(symbol)일 수 있습니다. 확인해주세요.")
            elif e.code == -1103:
                 logger.error("API 에러 코드 -1103: 너무 많은 파라미터가 전송되었습니다. 코드 확인이 필요합니다.")
            return None
        except Exception as e:
            logger.error(f"데이터 수집 중 예기치 않은 에러 발생: {e}", exc_info=True)
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol  # Add symbol column
        print(f"Successfully fetched {len(df)} records.")
        return df[['open', 'high', 'low', 'close', 'volume', 'symbol']]

    def save_to_influxdb(self, df, measurement_name, time_precision='s', chunk_size=5000):
        """
        데이터프레임을 InfluxDB에 청크 단위로 저장합니다.

        :param df: (pd.DataFrame) 저장할 데이터. 'symbol' 컬럼을 포함해야 합니다.
        :param measurement_name: (str) InfluxDB의 measurement 이름
        :param time_precision: (str) 시간 정밀도 ('s', 'ms', 'us', 'ns')
        :param chunk_size: (int) 한 번에 저장할 데이터 행 수
        :return: (bool) 저장 성공 여부
        """
        if not self.influx_write_api:
            print("InfluxDB client not available. Skipping save operation.")
            return False

        if 'symbol' not in df.columns:
            print("Error: 'symbol' column not found in DataFrame. Cannot save to InfluxDB with symbol tag.")
            return False

        num_chunks = (len(df) - 1) // chunk_size + 1
        print(f"Saving {len(df)} records to InfluxDB in {num_chunks} chunk(s) of size {chunk_size}... (Bucket: {self.influx_bucket})")

        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            df_chunk = df.iloc[start_index:end_index]

            print(f"  - Saving chunk {i + 1}/{num_chunks} ({len(df_chunk)} records)...")
            try:
                self.influx_write_api.write(
                    bucket=self.influx_bucket,
                    record=df_chunk,
                    data_frame_measurement_name=measurement_name,
                    data_frame_tag_columns=['symbol'],
                    time_precision=time_precision
                )
            except Exception as e:
                print(f"Error saving chunk {i + 1} to InfluxDB: {e}")
                # 실패 시 관련 정보와 함께 False 반환
                return False

        print("All chunks saved to InfluxDB successfully.")
        return True

    def start_websocket_stream(self, symbol, on_message_callback):
        """
        (구조 예시) 실시간 데이터 스트림을 시작합니다.
        실제 구현에서는 python-binance의 `BinanceSocketManager`를 사용합니다.
        
        :param symbol: (str) 구독할 심볼 (소문자로)
        :param on_message_callback: (function) 메시지 수신 시 호출될 콜백 함수
        """
        print(f"(Example) Starting WebSocket stream for {symbol}...")
        print("This is a placeholder. In a real scenario, a WebSocket client would run here.")
        # 예시: from binance import BinanceSocketManager
        # bsm = BinanceSocketManager(self.binance_client)
        # conn_key = bsm.start_trade_socket(symbol, on_message_callback)
        # bsm.start()
        pass

    def query_data_from_influxdb(self, bucket, measurement, symbol, start_time=None, stop_time=None):
        """
        InfluxDB에서 시계열 데이터를 조회하여 Pandas DataFrame으로 반환합니다.

        :param bucket: (str) 데이터를 조회할 버킷 이름
        :param measurement: (str) 조회할 measurement
        :param symbol: (str) 필터링할 'symbol' 태그 값 (예: 'BTCUSDT')
        :param start_time: (str 또는 datetime) 조회 시작 시간 (예: "-30d", "2023-01-01T00:00:00Z")
        :param stop_time: (str 또는 datetime) 조회 종료 시간 (예: "now()", "2023-12-31T23:59:59Z")
        :return: (pd.DataFrame) 조회된 데이터. OHLCV 컬럼과 datetime 인덱스를 가집니다.
        """
        if not self.influx_client:
            logger.error("InfluxDB client is not available.")
            return pd.DataFrame()

        # 파라미터 기본값 설정
        start_param = start_time if start_time is not None else "-30d"
        stop_param = stop_time if stop_time is not None else "now()"

        # datetime 객체가 전달된 경우 ISO 형식으로 변환
        if hasattr(start_param, 'isoformat'):
            start_param = f"{start_param.isoformat()}Z"
        if hasattr(stop_param, 'isoformat'):
            stop_param = f"{stop_param.isoformat()}Z"

        logger.info(f"Querying {measurement} data for {symbol} from {start_param} to {stop_param}")
        
        query_api = self.influx_client.query_api()

        flux_query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_param}, stop: {stop_param})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => r.symbol == "{symbol}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
          |> sort(columns: ["_time"], desc: false)
        '''

        logger.debug(f"Flux query: {flux_query}")

        try:
            logger.info(f"Executing query for {symbol}...")
            result_df = query_api.query_data_frame(query=flux_query)
            
            if result_df.empty:
                logger.warning(f"No data returned from InfluxDB for {symbol} in {measurement}")
                return pd.DataFrame()


            # 데이터프레임 정리
            if '_time' in result_df.columns:
                result_df.rename(columns={'_time': 'timestamp'}, inplace=True)
                result_df.set_index('timestamp', inplace=True)
                # 시간대 정보가 있으면 제거 (backtrader 호환성을 위해)
                if hasattr(result_df.index, 'tz') and result_df.index.tz is not None:
                    result_df.index = result_df.index.tz_convert(None)
                
                # 필요한 컬럼만 선택
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                result_df = result_df[[col for col in required_columns if col in result_df.columns]]
                
                logger.info(f"Successfully retrieved {len(result_df)} records for {symbol} from {result_df.index.min()} to {result_df.index.max()}")
                return result_df
            else:
                logger.error("'timestamp' column not found in query results")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error querying or processing data from InfluxDB: {str(e)}", exc_info=True)
            return pd.DataFrame()

if __name__ == '__main__':
    # --- 초기화 ---
    print("--- 데이터 파이프라인 실행 --- ")
    notifier = TelegramNotifier()
    collector = BinanceDataCollector()

    try:
        # --- 데이터 수집 설정 ---
        symbol_to_collect = 'BTCUSDT'
        timeframes_to_collect = ['1h', '4h', '1d']
        start_date_to_collect = '1 Jan, 2020' # 데이터 수집 시작일
        # end_date_to_collect = None # None으로 설정 시 현재까지 데이터를 가져옴

        print(f"\n--- 과거 데이터 수집 시작: {symbol_to_collect} ---")
        for tf in timeframes_to_collect:
            print(f"  - 타임프레임: {tf}")
            measurement_name = f"{symbol_to_collect}_{tf}"

            historical_data = collector.fetch_historical_data(
                symbol_to_collect, 
                tf, 
                start_date_to_collect, 
                end_str=None # 현재까지 데이터 수집
            )
            
            if historical_data is not None and not historical_data.empty:
                print(f"    성공적으로 {len(historical_data)}개의 과거 데이터를 가져왔습니다.")
                # InfluxDB에 저장 (save_to_influxdb 메소드 사용)
                if collector.save_to_influxdb(historical_data, measurement_name):
                    print(f"    {measurement_name}에 데이터 저장 성공.")
                    notifier.send_message(f"{symbol_to_collect} {tf} 데이터 {len(historical_data)}개 InfluxDB 저장 완료")
                else:
                    print(f"    {measurement_name}에 데이터 저장 실패.")
                    notifier.send_message(f"[오류] {symbol_to_collect} {tf} 데이터 InfluxDB 저장 실패")
            else:
                print(f"    {symbol_to_collect} ({tf}) 과거 데이터 수집에 실패했거나 데이터가 없습니다.")
                notifier.send_message(f"[경고] {symbol_to_collect} ({tf}) 데이터 수집 실패 또는 데이터 없음")

        print("\n--- 모든 과거 데이터 수집 및 저장 작업 완료 ---")
        notifier.send_message("모든 과거 데이터 수집 및 저장 작업이 완료되었습니다.")

    except Exception as e:
        print(f"\n--- 에러 발생 ---")
        error_message = f"❌ [에러] Project ART: 데이터 처리 중 예외가 발생했습니다: {e}"
        print(error_message)
        notifier.send_message(error_message)

    print("\n--- 데이터 파이프라인 테스트 종료 ---")

    # --- 실시간 데이터 스트림 예제 (참고용) ---
    # def handle_message(msg):
    #     # 실제로는 여기서 메시지를 처리하여 SignalEvent를 생성하거나 DB에 저장
    #     print(f"Received WebSocket message: {msg}")
    #
    # collector.start_websocket_stream('btcusdt', handle_message)
