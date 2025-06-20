# c:\art_project\config.py
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
    'timeframes': ['1h', '4h', '1d'], # 분석에 사용할 모든 타임프레임
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
    'fast_ma_period': 20,
    'slow_ma_period': 60,
    'adx_period': 14,
    'adx_threshold': 20,  # lowered from 25 for Experiment 1 (ADX 추세 강도 임계값)

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
