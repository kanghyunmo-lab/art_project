# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import logging
import traceback

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- 필요한 모듈 임포트 ---
from config.config import PATH_PARAMS, DATA_PARAMS, TRADING_PARAMS
from data_pipeline.collector import BinanceDataCollector
import pandas_ta as ta

def add_all_features(df):
    """데이터프레임에 기술적 지표와 시간 피처를 추가합니다."""
    logger.info("기술적 지표 및 시간 피처 추가 중...")
    # Pandas TA를 사용하여 다양한 기술적 지표 추가
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.adx(append=True)
    df.ta.obv(append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    
    # 시간 관련 피처 추가
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    df.dropna(inplace=True)
    logger.info("피처 추가 완료.")
    return df

def get_triple_barrier_labels(df, pt_sl, target_vol, period):
    """삼중 장벽 레이블링을 계산합니다."""
    logger.info("삼중 장벽 레이블링 계산 중...")
    # 이 부분은 단순화를 위해 가격 변화율로 대체합니다.
    # 실제 삼중 장벽 로직은 더 복잡합니다.
    daily_vol = df['close'].pct_change().rolling(period).std().mean()
    upper_barrier = df['close'] * (1 + daily_vol * pt_sl[0])
    lower_barrier = df['close'] * (1 - daily_vol * pt_sl[1])
    
    price_change = df['close'].shift(-period) / df['close'] - 1
    
    conditions = [
        (price_change > daily_vol * pt_sl[0]), # 상단 장벽 터치 (매수)
        (price_change < -daily_vol * pt_sl[1]), # 하단 장벽 터치 (매도)
    ]
    choices = [1, -1]
    df['label'] = np.select(conditions, choices, default=0)
    logger.info("레이블링 완료.")
    return df

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """데이터셋 생성 파이프라인 메인 함수"""
    try:
        logger.info("--- 데이터셋 생성 파이프라인 시작 ---")

        # 1. 데이터 로드
        # InfluxDB에서 전체 기간의 데이터를 가져옵니다.
        collector = BinanceDataCollector()
        all_data_start = DATA_PARAMS['train_start']
        all_data_end = DATA_PARAMS['test_end']
        logger.info(f"{all_data_start} 부터 {all_data_end} 까지의 데이터를 InfluxDB에서 로드합니다.")
        df = collector.query_data_from_influxdb(
            measurement='crypto_prices_hourly',
            symbol='BTCUSDT',
            start_time=all_data_start,
            end_time=all_data_end
        )
        if df.empty:
            logger.error("데이터 로드 실패. InfluxDB 연결 및 데이터 존재 여부를 확인하세요.")
            return
        logger.info(f"데이터 로드 완료. 총 {len(df)}개의 레코드.")

        # 2. 피처 엔지니어링 및 레이블링
        df_features = add_all_features(df.copy())
        df_labeled = get_triple_barrier_labels(df_features, pt_sl=TRADING_PARAMS['pt_sl_multipliers'], target_vol=0.01, period=24)
        logger.info("레이블링 완료.")

        # 4. 학습/테스트 데이터 분리
        train_start_dt = pd.to_datetime(DATA_PARAMS['train_start'])
        train_end_dt = pd.to_datetime(DATA_PARAMS['train_end'])
        test_start_dt = pd.to_datetime(DATA_PARAMS['test_start'])
        test_end_dt = pd.to_datetime(DATA_PARAMS['test_end'])

        df_train = df_labeled.loc[train_start_dt:train_end_dt]
        df_test = df_labeled.loc[test_start_dt:test_end_dt]

        logger.info(f"학습 데이터: {len(df_train)}개, 테스트 데이터: {len(df_test)}개")

        # 5. 데이터 저장
        output_dir = os.path.join(PATH_PARAMS['data_path'], 'processed')
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, 'btcusdt_labeled_features_train.parquet')
        test_path = os.path.join(output_dir, 'btcusdt_labeled_features_test.parquet')

        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

        logger.info(f"학습 데이터 저장 완료: {train_path}")
        logger.info(f"테스트 데이터 저장 완료: {test_path}")
        logger.info("--- 데이터셋 생성 파이프라인 성공적으로 완료 ---")

    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
