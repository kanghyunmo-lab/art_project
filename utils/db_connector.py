# -*- coding: utf-8 -*-
"""
InfluxDB와의 연결 및 데이터 조회를 담당하는 클래스를 정의합니다.
"""
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
import pandas as pd
import logging

# .env 파일에서 환경 변수 로드
# 이 모듈을 임포트하는 최상위 스크립트에서 load_dotenv()를 호출하는 것을 권장하지만,
# 직접 실행될 경우를 대비하여 여기서도 호출합니다.
load_dotenv()
logger = logging.getLogger(__name__)

class InfluxReader:
    """InfluxDB에서 데이터를 읽어오는 역할을 담당합니다."""
    def __init__(self):
        """InfluxDB 클라이언트를 초기화합니다."""
        try:
            # config.py에서 InfluxDB 설정 읽기
            from config import INFLUXDB_PARAMS
            
            # InfluxDB 설정 로그 출력
            logger.info(f"Connecting to InfluxDB at {INFLUXDB_PARAMS.get('url')} with org {INFLUXDB_PARAMS.get('org')}")
            
            # InfluxDB 클라이언트 초기화
            self.client = InfluxDBClient(
                url=INFLUXDB_PARAMS.get('url'),
                token=INFLUXDB_PARAMS.get('token'),
                org=INFLUXDB_PARAMS.get('org'),
                timeout=30_000  # 타임아웃을 30초로 설정
            )
            
            # 버킷 설정
            self.bucket = INFLUXDB_PARAMS.get('ohlcv_bucket')
            if not self.bucket:
                raise ValueError("InfluxDB bucket name is missing in config.py")
                
            # org 변수 초기화
            self.org = INFLUXDB_PARAMS.get('org')
            if not self.org:
                raise ValueError("InfluxDB org is missing in config.py")

            # 연결 테스트
            try:
                self.client.health()
                logger.info("Successfully connected to InfluxDB")
            except Exception as e:
                logger.error(f"Failed to connect to InfluxDB: {e}")
                raise

            # Query API 초기화
            self.query_api = self.client.query_api()
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise

    def get_data(self, query: str) -> pd.DataFrame:
        """주어진 Flux 쿼리를 실행하고 결과를 Pandas DataFrame으로 반환합니다."""
        try:
            logger.debug(f"Executing Flux query: \n{query}")
            result_df = self.query_api.query_data_frame(query=query, org=self.org)
            logger.debug(f"Query returned {len(result_df)} rows.")
            return result_df
        except Exception as e:
            logger.error(f"Error executing InfluxDB query: {e}")
            return pd.DataFrame() # 오류 발생 시 빈 데이터프레임 반환

    def close(self):
        """InfluxDB 클라이언트 연결을 닫습니다."""
        if self.client:
            self.client.close()
            logger.info("InfluxDB connection closed.")

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트 코드
    logging.basicConfig(level=logging.INFO)
    try:
        reader = InfluxReader()
        
        # 간단한 테스트 쿼리 (최근 10분 데이터)
        test_query = f'''
        from(bucket: "{reader.bucket}")
        |> range(start: -10m)
        |> filter(fn: (r) => r["_measurement"] == "ohlcv")
        |> limit(n: 5)
        '''
        
        print("--- Testing InfluxDB Connection with query ---")
        df = reader.get_data(test_query)
        
        if not df.empty:
            print("Successfully retrieved data:")
            print(df.head())
        else:
            print("Query returned no data or an error occurred.")
            
        reader.close()

    except Exception as e:
        print(f"An error occurred during the test: {e}")
