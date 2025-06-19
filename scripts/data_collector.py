#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
바이낸스에서 과거 OHLCV 데이터를 수집하여 InfluxDB에 저장하는 스크립트

[실행 예시]
# 5년치 BTCUSDT 1시간봉 데이터 수집 (기본값)
python scripts/data_collector.py

# 2022년 1월 1일부터 ETHUSDT 1일봉 데이터 수집
python scripts/data_collector.py --symbol ETHUSDT --interval 1d --start 2022-01-01

# 30일 전부터 XRPUSDT 4시간봉 데이터 수집
python scripts/data_collector.py --symbol XRPUSDT --interval 4h --days 30
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# 프로젝트 루트 경로를 시스템 경로에 추가
# os.path.dirname(__file__)는 현재 파일(data_collector.py)의 디렉토리 (scripts)
# os.path.join(..., '..')는 상위 디렉토리 (art_project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.collector import BinanceDataCollector

def main(args):
    print("=== 바이낸스 과거 데이터 수집기 시작 ===")
    
    collector = BinanceDataCollector()
    
    # 수집 기간 설정
    end_time = datetime.utcnow()
    if args.days:
        start_time = end_time - timedelta(days=args.days)
    else:
        start_time = datetime.strptime(args.start, '%Y-%m-%d')

    print(f"수집 기간: {start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}")
    print(f"대상: {args.symbol}, 간격: {args.interval}")
    
    df = collector.fetch_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start_str=start_time.strftime('%Y-%m-%d'),
        end_str=end_time.strftime('%Y-%m-%d')
    )
    
    if df is not None and not df.empty:
        print(f"\n총 {len(df)}개의 레코드를 성공적으로 수집했습니다.")
        print("\n최근 5개 레코드:")
        print(df.tail())
        
        # InfluxDB 저장
        # 예: 'BTCUSDT_1h'
        measurement = f"{args.symbol}_{args.interval}"
        print(f"\nInfluxDB '{measurement}' 테이블에 저장 중...")
        
        success = collector.save_to_influxdb(df, measurement, time_precision='s')
        if success:
            print("\n데이터가 성공적으로 InfluxDB에 저장되었습니다!")
        else:
            print("\nInfluxDB에 데이터를 저장하는 데 실패했습니다.")
    else:
        print("\n수집된 데이터가 없거나 오류가 발생했습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='바이낸스 과거 데이터 수집 스크립트')
    
    # 기본 시작 날짜: 5년 전 오늘
    default_start_date = (datetime.utcnow() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='수집할 암호화폐 심볼 (예: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='데이터 간격 (예: 1m, 5m, 1h, 1d)')
    parser.add_argument('--start', type=str, default=default_start_date, help='수집 시작일 (YYYY-MM-DD 형식)')
    parser.add_argument('--days', type=int, help='오늘로부터 몇일 전까지 수집할지 지정 (start 대신 사용)')

    args = parser.parse_args()
    main(args)
