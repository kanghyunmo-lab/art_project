# -*- coding: utf-8 -*-
"""
InfluxDB에서 시계열 데이터를 읽어오는 모듈.
"""
import os
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

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
        self.bucket = os.getenv('INFLUXDB_BUCKET')
        
        if not all([self.url, self.token, self.org, self.bucket]):
            raise ValueError("InfluxDB 연결 정보가 환경 변수에 올바르게 설정되지 않았습니다.")

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()

    def get_data(self, measurement_name, start='-30d', end='now()', symbol=None):
        """
        지정된 measurement에서 데이터를 조회하여 DataFrame으로 반환합니다.

        :param measurement_name: (str) 조회할 measurement 이름
        :param start: (str) 조회 시작 시간 (Flux 쿼리 형식, 예: '-30d', '2022-01-01T00:00:00Z')
        :param end: (str) 조회 종료 시간 (Flux 쿼리 형식)
        :param symbol: (str, optional) 필터링할 'symbol' 태그 값
        :return: (pd.DataFrame) 조회된 데이터
        """
        flux_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {end})
          |> filter(fn: (r) => r._measurement == "{measurement_name}")
        '''
        
        if symbol:
            flux_query += f'  |> filter(fn: (r) => r.symbol == "{symbol}")\n'

        flux_query += '''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "open", "high", "low", "close", "volume", "symbol"])
        '''
        
        print(f"Executing Flux query for {measurement_name}...")
        try:
            df = self.query_api.query_data_frame(query=flux_query, org=self.org)
            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True)

            if df.empty:
                print(f"Warning: No data returned for {measurement_name}")
                return pd.DataFrame()

            # 데이터 타입 정리
            df.rename(columns={'_time': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['result', 'table'], inplace=True, errors='ignore')

            # InfluxDB에서 정수형으로 저장된 값을 float으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            print(f"Error querying InfluxDB: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # --- 예제 사용법 ---
    reader = InfluxReader()
    
    # 1시간봉 데이터 조회
    df_1h = reader.get_data('BTCUSDT_1h', symbol='BTCUSDT', start='-7d')
    print("--- BTCUSDT 1h Data (last 7 days) ---")
    print(df_1h.head())
    print(df_1h.tail())
    print(f"Shape: {df_1h.shape}")
