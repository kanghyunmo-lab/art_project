# c:\art_project\config.py

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
# TRADING STRATEGY CONFIGURATIONS (for labeling.py & models)
# ==============================================================================
TRADING_PARAMS = {
    # Trend Definition
    'fast_ma_period': 20,
    'slow_ma_period': 60,
    'adx_period': 14,
    'adx_threshold': 25, # ADX 추세 강도 임계값

    # Volatility & Barrier Definition
    'atr_period_for_trgt': 20, # trgt 계산용 ATR 기간
    'num_candles_max_hold': 48, # 최대 보유 캔들 수 (수직 장벽, 48시간)
    'pt_sl_multipliers': [3.0, 1.0], # [profit_take_multiplier, stop_loss_multiplier]
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
