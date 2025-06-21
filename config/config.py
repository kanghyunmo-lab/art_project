# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# .env 파일을 로드하여 환경 변수를 설정합니다.
# config 파일을 임포트하는 모든 스크립트에서 환경 변수를 사용할 수 있습니다.
load_dotenv()

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 경로 설정 ---
PATH_PARAMS = {
    'root_path': ROOT_DIR,
    'data_path': os.path.join(ROOT_DIR, 'data'),
    'log_path': os.path.join(ROOT_DIR, 'logs'),
    'model_path': os.path.join(ROOT_DIR, 'models', 'saved_models'),
}

# --- InfluxDB 설정 ---
INFLUXDB_PARAMS = {
    'url': os.getenv('INFLUXDB_URL'),
    'token': os.getenv('INFLUXDB_TOKEN'),
    'org': os.getenv('INFLUXDB_ORG'),
    'funding_rate_bucket': 'funding_rates',
    'funding_rate_measurement': 'funding_rate_history',
    'ohlcv_bucket': 'art_project',
    'ohlcv_measurement': 'ohlcv',
}

# --- 데이터 및 피처 설정 ---
DATA_PARAMS = {
    'symbol': 'BTC/USDT',
    'timeframes': ['1h', '4h', '1d'],
    'main_timeframe': '1h',  # 주 분석 기준 타임프레임
    'fetch_start_date': '2017-01-01T00:00:00Z',
    'train_start_date': '2017-01-01T00:00:00Z',
    'train_end_date': '2023-12-31T23:59:59Z',
    'test_start_date': '2024-01-01T00:00:00Z',
    'test_end_date': '2025-12-31T23:59:59Z',
}

# --- 백테스터 설정 ---
BACKTESTER_PARAMS = {
    'initial_cash': 10000,  # 초기 자본금 (USDT)
    'commission': 0.001,    # 거래 수수료 (0.1%)
    'data_path': 'data/processed/btcusdt_labeled_features_test.parquet',  # 백테스트용 데이터 경로
    'model_path': 'models/saved_models/model_final.joblib'  # 모델 파일 경로
}

# --- 펀딩 비율 수집기 설정 ---
FUNDING_RATE_COLLECTOR_PARAMS = {
    'default_symbol': 'BTCUSDT',
    'default_mode': 'historical',
    'default_limit': 10,
    'historical_batch_limit': 1000,
    'api_call_delay_seconds': 0.5
}

# --- 트레이딩 전략 및 레이블링 설정 ---
TRADING_PARAMS = {
    # Triple-Barrier Method 파라미터
    'pt_sl_multipliers': [1.1, 0.95],  # [수익 실현, 손실 제한] multiplier (이익실현 1.1배, 손절 0.95배)
    'holding_period': 8,  # 최대 보유 기간 (1h 캔들 기준, 8시간으로 단축)
    'num_candles_max_hold': 8,  # 최대 보유 캔들 수 (8시간으로 단축)
    # 트렌드/변동성/신호 관련 파라미터
    'fast_ma_period': 5,  # 더 빠른 추세 전환 감지
    'slow_ma_period': 21,  # 장기 추세 유지
    'adx_period': 14,  # ADX 기간
    'adx_threshold': 8,  # 더 낮은 임계값으로 더 많은 신호 확보
    'atr_period_for_trgt': 14,
    # 이벤트 샘플링 파라미터
    'volatility_lookback': 50,  # 변동성 계산 기간
    'volatility_target': 0.005, # CUSUM 필터의 일일 변동성 목표
}

# --- 머신러닝 모델 설정 ---
MODEL_PARAMS = {
    'xgboost': {
        # 기본 파라미터
        'objective': 'multi:softprob',  # 다중 분류
        'num_class': 3,                # 클래스 개수 (-1, 0, 1)
        'eval_metric': 'mlogloss',     # 평가 지표
        'random_state': 42,            # 재현성을 위한 시드
        'use_label_encoder': False,     # 레이블 인코더 사용 안 함
        
        # 학습 파라미터
        'n_estimators': 1000,          # 트리 개수 (early stopping으로 조정됨)
        'learning_rate': 0.01,         # 학습률 (더 낮은 학습률로 더 안정적인 학습)
        'max_depth': 6,                # 트리 최대 깊이 (과적합 방지)
        'min_child_weight': 1,         # 자식 노드의 최소 가중치 합 (과적합 조절)
        'gamma': 0.1,                  # 리프 노드 추가 분할을 위한 최소 손실 감소
        'subsample': 0.8,              # 훈련 데이터 샘플링 비율 (과적합 방지)
        'colsample_bytree': 0.8,       # 특성 샘플링 비율 (과적합 방지)
        'colsample_bylevel': 0.8,      # 레벨별 특성 샘플링 비율
        'reg_alpha': 0.1,              # L1 정규화 (과적합 방지)
        'reg_lambda': 1.0,             # L2 정규화 (과적합 방지)
        'scale_pos_weight': 1,         # 클래스 가중치 (불균형 데이터셋 대응)
        
        # 성능 최적화
        'n_jobs': -1,                  # 사용할 CPU 코어 수 (-1: 모든 코어 사용)
        'tree_method': 'hist',         # 히스토그램 기반 트리 구성 (메모리 효율적)
        'grow_policy': 'depthwise',    # 트리 성장 정책 (depthwise: 균형 잡힌 트리)
        'max_bin': 256,                # 히스토그램의 최대 bin 수
        'early_stopping_rounds': 50,    # 조기 종료를 위한 라운드 수
        'verbose_eval': 10,            # 학습 진행 상황 출력 주기 (10번마다 출력)
    },
    
    # 조기 종료 설정
    'early_stopping_rounds': 50,        # 검증 점수가 개선되지 않을 때 기다릴 라운드 수
    'verbose_eval': 10                  # 학습 진행 상황 출력 주기
}

# --- 리스크 관리 설정 ---
RISK_PARAMS = {
    'initial_capital': 10000, # USD
    'leverage_default': 5,
    'max_risk_per_trade': 0.01, # 전체 자본 대비 거래당 최대 손실률 (1%)
    'max_drawdown_limit': 0.20, # 전체 자본 대비 최대 낙폭 한도 (20%)
    'max_open_positions': 5,
}

# --- 로깅 설정 ---
LOGGING_PARAMS = {
    'level': 'INFO', # 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
}

# --- API 호출 재시도 설정 ---
RETRY_CONFIG = {
    "max_retry_attempts": 5,
    "initial_backoff_seconds": 1,
    "max_backoff_seconds": 60,
    "jitter": True,
    "retry_http_status_codes": [500, 502, 503, 504],
}
