# -*- coding: utf-8 -*-
"""
전체 프로젝트에서 사용되는 주요 설정 변수들을 정의합니다.
"""
import os

# --- 경로 설정 ---
# 프로젝트의 루트 디렉토리를 기준으로 절대 경로를 생성합니다.
# 이 스크립트(config.py)는 'c:\art_project\config'에 위치하므로,
# 루트 디렉토리는 이 파일 위치의 상위 디렉토리입니다.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_PARAMS = {
    'root_path': ROOT_DIR,
    'data_path': os.path.join(ROOT_DIR, 'data'),
    'log_path': os.path.join(ROOT_DIR, 'logs'),
    'model_path': os.path.join(ROOT_DIR, 'models', 'saved_models'),
}

# --- 데이터 설정 ---
DATA_PARAMS = {
    'symbol': 'BTC/USDT',
    'timeframes': ['1h', '4h', '1d'],
    # 데이터 수집 및 학습/테스트 기간 설정
    'fetch_start_date': '2017-01-01T00:00:00Z',
    'train_start_date': '2017-01-01T00:00:00Z',
    'train_end_date': '2023-12-31T23:59:59Z',
    'test_start_date': '2024-01-01T00:00:00Z',
    'test_end_date': '2025-12-31T23:59:59Z',
}

# --- 트레이딩 전략 설정 ---
TRADING_PARAMS = {
    # Triple-Barrier Method 파라미터
    'pt_sl_multipliers': [1.5, 1.5],  # [수익 실현(profit taking) multiplier, 손실 제한(stop loss) multiplier]
    'holding_period': 24,  # 최대 보유 기간 (단위: 1h 캔들)
    
    # 이벤트 샘플링 파라미터
    'volatility_lookback': 50,  # 변동성 계산 기간 (단위: 1h 캔들)
    'volatility_target': 0.005, # CUSUM 필터의 일일 변동성 목표
}

# --- 머신러닝 모델 설정 ---
MODEL_PARAMS = {
    'model_name': 'XGBoost',
    'params': {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'use_label_encoder': False
    },
    # 클래스 불균형 처리를 위한 가중치. (음성 클래스 수 / 양성 클래스 수)
    # 이 값은 train_model.py에서 동적으로 계산되어 덮어쓰여야 합니다.
    'scale_pos_weight': 1.0 
}

# --- 리스크 관리 설정 ---
RISK_PARAMS = {
    'max_drawdown': 0.20,  # 최대 허용 낙폭 (20%)
    'max_risk_per_trade': 0.02,  # 거래당 최대 리스크 (2%)
    'max_open_positions': 5,  # 최대 동시 보유 포지션 수
    'leverage': 1.0, # 레버리지
}

# --- 로깅 설정 ---
LOGGING_PARAMS = {
    'level': 'INFO', # 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
}

"""
전체 프로젝트에서 사용되는 주요 설정 변수들을 정의합니다.
"""
import os

# --- 경로 설정 ---
# 프로젝트의 루트 디렉토리를 기준으로 절대 경로를 생성합니다.
# 이 스크립트(config.py)는 'c:\art_project\config'에 위치하므로,
# 루트 디렉토리는 이 파일 위치의 상위 디렉토리입니다.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_PARAMS = {
    'root_path': ROOT_DIR,
    'data_path': os.path.join(ROOT_DIR, 'data'),
    'log_path': os.path.join(ROOT_DIR, 'logs'),
    'model_path': os.path.join(ROOT_DIR, 'models', 'saved_models'),
}

# --- 데이터 설정 ---
DATA_PARAMS = {
    'symbol': 'BTC/USDT',
    'timeframes': ['1h', '4h', '1d'],
    # 데이터 수집 및 학습/테스트 기간 설정
    'fetch_start_date': '2017-01-01T00:00:00Z',
    'train_start_date': '2017-01-01T00:00:00Z',
    'train_end_date': '2023-12-31T23:59:59Z',
    'test_start_date': '2024-01-01T00:00:00Z',
    'test_end_date': '2025-12-31T23:59:59Z',
}

# --- 트레이딩 전략 설정 ---
TRADING_PARAMS = {
    # Triple-Barrier Method 파라미터
    'pt_sl_multipliers': [1.5, 1.5],  # [수익 실현(profit taking) multiplier, 손실 제한(stop loss) multiplier]
    'holding_period': 24,  # 최대 보유 기간 (단위: 1h 캔들)
    
    # 이벤트 샘플링 파라미터
    'volatility_lookback': 50,  # 변동성 계산 기간 (단위: 1h 캔들)
    'volatility_target': 0.005, # CUSUM 필터의 일일 변동성 목표
}

# --- 머신러닝 모델 설정 ---
MODEL_PARAMS = {
    'model_name': 'XGBoost',
    'params': {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'use_label_encoder': False
    },
    # 클래스 불균형 처리를 위한 가중치. (음성 클래스 수 / 양성 클래스 수)
    # 이 값은 train_model.py에서 동적으로 계산되어 덮어쓰여야 합니다.
    'scale_pos_weight': 1.0 
}

# --- 리스크 관리 설정 ---
RISK_PARAMS = {
    'max_drawdown': 0.20,  # 최대 허용 낙폭 (20%)
    'max_risk_per_trade': 0.02,  # 거래당 최대 리스크 (2%)
    'max_open_positions': 5,  # 최대 동시 보유 포지션 수
    'leverage': 1.0, # 레버리지
}

# --- 로깅 설정 ---
LOGGING_PARAMS = {
    'level': 'INFO', # 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
}

import os
# from dotenv import load_dotenv # .env 로드는 호출하는 스크립트(예: collect_funding_rate.py)에서 수행하는 것을 가정

# InfluxDB Credentials from environment variables
INFLUXDB_URL = os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')

# Optional: Check if essential variables are loaded and print a warning if not
# This helps in diagnosing if .env wasn't loaded correctly by the calling script
# or if the variables are missing from .env
if not all([INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG]):
    print("WARNING: One or more InfluxDB environment variables (INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG) are not set.")
    print("Please ensure they are defined in your .env file and that the .env file is loaded prior to importing this config.")


# ==============================================================================
# PATH CONFIGURATIONS
# ==============================================================================
PATH_PARAMS = {
    'data_path': 'c:/art_project/data/',
    'model_path': 'c:/art_project/models/',
    'log_path': 'c:/art_project/logs/'
}

# ==============================================================================
# DATA & FEATURE CONFIGURATIONS
# ==============================================================================
DATA_PARAMS = {
    'symbol': 'BTC/USDT',
    'timeframes': ['1h', '4h', '1d'],
    'data_type': 'ohlcv',
    'train_start_date': '2017-01-01T00:00:00Z',
    'train_end_date': '2023-12-31T23:59:59Z',
    'test_start_date': '2024-01-01T00:00:00Z',
    'test_end_date': '2025-06-21T23:59:59Z',
    'main_timeframe': '1h', # 주 분석 및 신호 생성 기준 타임프레임
}

# ==============================================================================
# FUNDING RATE COLLECTOR CONFIGURATIONS
# ==============================================================================
FUNDING_RATE_COLLECTOR_PARAMS = {
    'default_symbol': 'BTCUSDT',        # 데이터 수집 기본 심볼
    'default_mode': 'recent',          # 기본 수집 모드 ('recent' 또는 'historical')
    'default_limit': 10,               # 'recent' 모드일 때 가져올 최근 데이터 개수 (API 기본값은 100, 최대 1000)
    'historical_batch_limit': 1000,    # 'historical' 모드일 때 API 호출당 가져올 데이터 개수 (최대 1000)
    'api_call_delay_seconds': 0.5      # API 호출 간 지연 시간 (초)
}

# ==============================================================================
# TRADING STRATEGY CONFIGURATIONS (for labeling.py & models)
# ==============================================================================
TRADING_PARAMS = {
    # Trend Definition
    'fast_ma_period': 10,  # Experiment 4.2: further shortened
    'slow_ma_period': 20,  # Experiment 4.2: further shortened,
    'adx_period': 14,
    'adx_threshold': 5,  # Experiment 4.1: Effectively remove ADX filter (ADX 추세 강도 임계값)

    # Volatility & Barrier Definition
    'atr_period_for_trgt': 20, # trgt 계산용 ATR 기간
    'num_candles_max_hold': 48, # 최대 보유 캔들 수 (수직 장벽, 48시간)
    'pt_sl_multipliers': [3.0, 1.0], # [profit_take_multiplier, stop_loss_multiplier]
}

# ==============================================================================
# RETRY CONFIGURATIONS
# ==============================================================================
RETRY_CONFIG = {
    "max_retry_attempts": 5,        # 최대 재시도 횟수
    "initial_backoff_seconds": 1,   # 초기 대기 시간 (초)
    "max_backoff_seconds": 60,      # 최대 대기 시간 (초)
    "jitter": True,                   # 대기 시간에 무작위성 추가 여부
    # 재시도할 HTTP 상태 코드 목록 (주로 5xx 서버 오류)
    "retry_http_status_codes": [500, 502, 503, 504],
}

# ==============================================================================
# INFLUXDB CONFIGURATIONS
# ==============================================================================
INFLUXDB_PARAMS = {
    'funding_rate_bucket': 'funding_rates',
    'funding_rate_measurement': 'funding_rate_history',
    # 기존 collector.py에서 사용하는 버킷/measurement와 충돌 방지
    'ohlcv_bucket': 'art_project', # 사용자의 실제 OHLCV 데이터 버킷
    'ohlcv_measurement': 'ohlcv',
}

# ==============================================================================
# RISK & CAPITAL MANAGEMENT CONFIGURATIONS
# ==============================================================================
MODEL_PARAMS = {
    'xgboost': {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'use_label_encoder': False
    }
}

RISK_PARAMS = {
    'initial_capital': 10000, # USD
    'leverage_default': 5, # 기본 레버리지
    
    # Capital Allocation by Market Regime
    'capital_allocation_aggressive': 0.50, # 50%
    'capital_allocation_balanced': 0.30,   # 30%
    'capital_allocation_conservative': 0.10, # 10%
    
    # Portfolio-level Risk
    'max_risk_per_trade': 0.01, # 전체 자본 대비 거래당 최대 손실률 (1%)
    'max_drawdown_limit': 0.20, # 전체 자본 대비 최대 낙폭 한도 (20%)
}
