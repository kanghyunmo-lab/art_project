#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
바이낸스에서 데이터를 수집하여 InfluxDB에 저장하는 스크립트
"""
import os
import sys
from datetime import datetime, timedelta

# 프로젝트 루트 경로를 시스템 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_pipeline.collector import BinanceDataCollector

def main():
    print("=== 바이낸스 데이터 수집기 시작 ===")
    
    # 데이터 수집기 초기화
    collector = BinanceDataCollector()
    
    # 수집할 기간 설정 (최근 7일)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    print(f"{start_time}부터 {end_time}까지의 데이터를 수집합니다...")
    
    # BTC/USDT 1시간봉 데이터 수집
    symbol = 'BTCUSDT'
    interval = '1h'
    
    print(f"\n{symbol}의 {interval} 간격으로 데이터를 수집 중...")
    df = collector.fetch_historical_data(
        symbol=symbol,
        interval=interval,
        start_str=start_time.strftime('%d %b, %Y %H:%M:%S'),
        end_str=end_time.strftime('%d %b, %Y %H:%M:%S')
    )
    
    if not df.empty:
        print(f"\n{len(df)}개의 레코드를 성공적으로 수집했습니다.")
        print("\n처음 5개 레코드:")
        print(df.head())
        
        # InfluxDB에 저장
        measurement = 'crypto_prices_hourly'
        print(f"\nInfluxDB에 {measurement} 측정값으로 저장 중...")
        
        success = collector.save_to_influxdb(df, measurement)
        if success:
            print("\n데이터가 성공적으로 InfluxDB에 저장되었습니다!")
        else:
            print("\nInfluxDB에 데이터를 저장하는 데 실패했습니다.")
    else:
        print("\n수집된 데이터가 없습니다.")

if __name__ == '__main__':
    main()
