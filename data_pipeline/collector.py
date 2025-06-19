print("--- LOADING LATEST collector.py ---")
import logging
logger = logging.getLogger(__name__)
logger.info("--- collector.py 스크립트 시작됨 ---")
import pandas as pd
from binance.client import Client
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

        :param symbol: (str) 거래 쌍 (예: 'BTCUSDT')
        :param interval: (str) 데이터 간격 (예: '1h', '4h', '1d')
        :param start_str: (str) 시작 날짜 (예: '1 Jan, 2020')
        :param end_str: (str, optional) 종료 날짜. 기본값은 None (현재까지).
        :return: (pd.DataFrame) OHLCV 데이터
        """
        print(f"Fetching historical data for {symbol} from {start_str} to {end_str or 'now'}...")
        klines = self.binance_client.get_historical_klines(symbol, interval, start_str, end_str)
        
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

    def save_to_influxdb(self, df, measurement_name, chunk_size=5000):
        """
        데이터프레임을 InfluxDB에 청크 단위로 저장합니다.

        :param df: (pd.DataFrame) 저장할 데이터. 'symbol' 컬럼을 포함해야 합니다.
        :param measurement_name: (str) InfluxDB의 measurement 이름
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
                    data_frame_tag_columns=['symbol']
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

if __name__ == '__main__':
    # --- 초기화 ---
    print("--- 데이터 파이프라인 테스트 시작 ---")
    notifier = TelegramNotifier()
    collector = BinanceDataCollector()

    # InfluxDB 연결 상태 확인
    if not collector.influx_write_api:
        message = "Project ART: InfluxDB 클라이언트가 초기화되지 않았습니다. 데이터 저장 및 알림을 건너뜁니다."
        print(message)
        # notifier.send_message(message) # InfluxDB 연결 실패도 알릴 수 있음
    else:
        # --- 과거 데이터 수집 및 저장 예제 ---
        try:
            # 1. BTC/USDT 1시간 봉 데이터를 2023년 1월 1일부터 가져오기
            print("\n[1/3] 바이낸스에서 과거 데이터 수집 시작...")
            btc_df = collector.fetch_historical_data('BTCUSDT', '1h', '1 Jan, 2023')
            
            if not btc_df.empty:
                print("\n--- 수집된 데이터 샘플 ---")
                print(btc_df.head())

                # 2. InfluxDB에 저장
                print("\n[2/3] InfluxDB에 데이터 저장 시작...")
                save_successful = collector.save_to_influxdb(btc_df, 'crypto_prices_hourly')

                # 3. 텔레그램 알림
                print("\n[3/3] 결과 텔레그램 알림 발송...")
                if save_successful:
                    message = f"✅ [성공] Project ART: BTCUSDT 데이터 {len(btc_df)}건을 InfluxDB에 성공적으로 저장했습니다."
                    notifier.send_message(message)
                else:
                    message = f"❌ [실패] Project ART: BTCUSDT 데이터를 InfluxDB에 저장하는 데 실패했습니다. 로그를 확인해주세요."
                    notifier.send_message(message)
                print("알림 발송 완료.")
            else:
                print("수집된 데이터가 없습니다. 작업을 중단합니다.")

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
