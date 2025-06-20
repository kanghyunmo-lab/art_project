# -*- coding: utf-8 -*-
"""
InfluxDB에서 시계열 데이터를 읽어오는 모듈.
"""
import os
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

import logging

# 로거 설정
logger = logging.getLogger(__name__)

class InfluxReader:
    def __init__(self):
        """
        InfluxDB 클라이언트 초기화 및 연결 설정.
        환경 변수에서 연결 정보를 로드합니다.
        """
        load_dotenv()
        self.url = os.getenv('INFLUXDB_URL')
        self.token = os.getenv('INFLUXDB_TOKEN')
        self.org = os.getenv('INFLUXDB_ORG')

        if not all([self.url, self.token, self.org]):
            logger.error("InfluxDB 연결 정보(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG)가 환경 변수에 올바르게 설정되지 않았습니다.")
            raise SystemExit("InfluxDB 설정 오류로 프로그램을 종료합니다.")

        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=30_000) # 30초 타임아웃
            # 연결 테스트 (선택 사항, 하지만 시작 시점에 문제를 파악하는 데 도움됨)
            if not self.client.ping():
                logger.error(f"InfluxDB 서버 ({self.url})에 연결할 수 없습니다. PING 실패.")
                raise SystemExit("InfluxDB 연결 실패로 프로그램을 종료합니다.")
            logger.info(f"InfluxDB ({self.url})에 성공적으로 연결되었습니다.")
            self.query_api = self.client.query_api()
        except Exception as e:
            logger.error(f"InfluxDB 클라이언트 초기화 중 오류 발생: {e}")
            raise SystemExit("InfluxDB 클라이언트 초기화 실패로 프로그램을 종료합니다.")

    def get_data(self, bucket_name: str, start: str, end: str, symbol: str, timeframe: str, data_type: str = 'ohlcv'):
        """
        지정된 bucket에서 데이터를 조회하여 DataFrame으로 반환합니다.
        measurement 이름은 symbol과 timeframe을 조합하여 동적으로 생성됩니다.

        :param bucket_name: (str) 조회할 bucket 이름
        :param start: (str) 조회 시작 시간 (Flux 쿼리 형식, 예: '-30d', '2022-01-01T00:00:00Z')
        :param end: (str) 조회 종료 시간 (Flux 쿼리 형식)
        :param symbol: (str) 필터링할 'symbol' (예: 'BTC/USDT')
        :param timeframe: (str) 조회할 시간 프레임 (예: '1h', '4h', '1d')
        :param data_type: (str) 조회할 데이터 종류 ('ohlcv' 또는 'funding_rate')
        :return: (pd.DataFrame) 조회된 데이터
        """
        logger = logging.getLogger(self.__class__.__name__)

        flux_query_parts = [
            f'from(bucket: "{bucket_name}")',
            f'  |> range(start: {start}, stop: {end})',
        ]

        if data_type == 'ohlcv':
            # OHLCV 데이터: measurement 이름을 동적으로 생성
            dynamic_measurement_name = f"{symbol.replace('/', '')}_{timeframe}"
            flux_query_parts.extend([
                f'  |> filter(fn: (r) => r._measurement == "{dynamic_measurement_name}")',
                '  |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")',
                '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")',
                '  |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])'
            ])
        elif data_type == 'funding_rate':
            # 펀딩비 데이터: 환경변수에서 measurement 이름을 가져오고, symbol 태그로 필터링
            funding_rate_measurement = os.getenv('INFLUXDB_MEASUREMENT_FUNDING', 'funding_rate')
            flux_query_parts.extend([
                f'  |> filter(fn: (r) => r._measurement == "{funding_rate_measurement}")',
                f'  |> filter(fn: (r) => r.symbol == "{symbol.replace("/", "")}")',
                '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")',
            ])

        flux_query = '\n'.join(flux_query_parts)
        logger.debug(f"Executing Flux query:\n{flux_query}")

        try:
            query_api = self.client.query_api()
            df = query_api.query_data_frame(query=flux_query)
            
            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()

            if df.empty:
                logger.warning(f"Query returned an empty DataFrame for {symbol} ({data_type} / {timeframe}) in bucket {bucket_name}.")
                return pd.DataFrame()

            if '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'])
                df.set_index('_time', inplace=True)
                df.index.name = 'timestamp'
            
            for col in ['result', 'table']:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)

            # 데이터 유형별 후처리
            if data_type == 'ohlcv':
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            elif data_type == 'funding_rate':
                if 'fundingRate' in df.columns:
                    df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
                    df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
                else:
                    logger.warning("'fundingRate' column not found in funding rate data.")

            df.sort_index(inplace=True)
            logger.info(f"Successfully loaded {len(df)} rows for {symbol} ({data_type} / {timeframe}) from {bucket_name}.")
            return df

        except Exception as e:
            logger.error(f"Error executing Flux query or processing data: {e}\nQuery:\n{flux_query}")
            traceback.print_exc()
            return pd.DataFrame()

if __name__ == '__main__':
    # --- 예제 사용법 (환경 변수 및 config.py 설정 필요) ---
    # 이 스크립트를 직접 실행하기 전에, config.py나 유사한 설정 파일에서
    # INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG 와 필요한 버킷/측정값 이름을 정의해야 합니다.
    # 예: from config import INFLUXDB_PARAMS, DATA_PARAMS


    logger.info("InfluxReader 예제 실행 시작")

    try:
        reader = InfluxReader()

        # config.py에서 설정값 가져오기 (실제 프로젝트에서는 config 모듈을 임포트)
        # 여기서는 예시를 위해 하드코딩된 값을 사용합니다.
        # 실제 사용 시에는 from config import INFLUXDB_PARAMS, DATA_PARAMS 와 같이 사용하세요.
        ohlcv_bucket = os.getenv('INFLUXDB_BUCKET_OHLCV', 'crypto_data') # .env 또는 기본값
        ohlcv_measurement_template = '{symbol}_{timeframe}' # 예: BTCUSDT_1h
        funding_bucket = os.getenv('INFLUXDB_BUCKET_FUNDING', 'funding_rates')
        funding_measurement = os.getenv('INFLUXDB_MEASUREMENT_FUNDING', 'funding_rate_history')
        test_symbol_ohlcv = 'BTCUSDT'
        test_symbol_funding = 'BTCUSDT' # API에서 사용하는 심볼 (예: BTCUSDT)
        
        # OHLCV 데이터 조회 예제
        logger.info(f"--- {test_symbol_ohlcv} 1h OHLCV 데이터 조회 (최근 3일) ---")
        measurement_1h = ohlcv_measurement_template.format(symbol=test_symbol_ohlcv.replace('/', ''), timeframe='1h')
        df_1h = reader.get_data(
            bucket_name=ohlcv_bucket, 
            measurement_name=measurement_1h, 
            symbol=test_symbol_ohlcv.replace('/', ''), # InfluxDB 태그는 보통 / 없음
            start='-3d',
            data_type='ohlcv'
        )
        if not df_1h.empty:
            print(df_1h.head())
            print(df_1h.tail())
            logger.info(f"Shape: {df_1h.shape}")
        else:
            logger.warning(f"{measurement_1h} 데이터 조회 실패 또는 데이터 없음")

        # 펀딩비 데이터 조회 예제
        logger.info(f"--- {test_symbol_funding} 펀딩비 데이터 조회 (최근 7일) ---")
        df_funding = reader.get_data(
            bucket_name=funding_bucket,
            measurement_name=funding_measurement,
            symbol=test_symbol_funding, # 펀딩비 저장 시 사용된 symbol 태그 값
            start='-7d',
            data_type='funding_rate'
        )
        if not df_funding.empty:
            print(df_funding.head())
            print(df_funding.tail())
            logger.info(f"Shape: {df_funding.shape}")
        else:
            logger.warning(f"{funding_measurement} ({test_symbol_funding}) 데이터 조회 실패 또는 데이터 없음")

    except SystemExit as e:
        logger.error(f"프로그램 실행 중단: {e}")
    except Exception as e:
        logger.error(f"예제 실행 중 예기치 않은 오류 발생: {e}", exc_info=True)

    logger.info("InfluxReader 예제 실행 완료")

