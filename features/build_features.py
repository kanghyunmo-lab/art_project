# -*- coding: utf-8 -*-
"""
다중 타임프레임 데이터를 기반으로 피처를 생성, 통합하고 저장합니다.
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가하여 모듈 임포트 문제 해결
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 모듈 임포트는 여기서 한번만 수행합니다 ---
from data_pipeline.influx_reader import InfluxReader
from config.config import INFLUXDB_PARAMS, DATA_PARAMS # 설정 파일에서 파라미터 가져오기
import logging # 로깅 추가

def add_technical_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal
    df['bollinger_mavg_20d'] = df['close'].rolling(window=20).mean()
    df['bollinger_std_20d'] = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mavg_20d'] + (df['bollinger_std_20d'] * 2)
    df['bollinger_lower'] = df['bollinger_mavg_20d'] - (df['bollinger_std_20d'] * 2)
    return df

# --- 1. 데이터 로딩 함수들 ---
def load_ohlcv_data_for_timeframe(reader: InfluxReader, symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    """지정된 타임프레임의 OHLCV 데이터를 InfluxDB에서 로드합니다."""
    logger.debug(f"Loading {timeframe} OHLCV data for {symbol} from {start_date} to {end_date}")
    
    df = reader.get_data(
        bucket_name=INFLUXDB_PARAMS['ohlcv_bucket'],
        start=start_date,
        end=end_date,
        symbol=symbol,
        timeframe=timeframe,
        data_type='ohlcv'
    )

    if df.empty:
        logger.warning(f"No {timeframe} data for {symbol} found in bucket '{INFLUXDB_PARAMS['ohlcv_bucket']}'. Returning empty DataFrame.")
        return pd.DataFrame()

    # InfluxReader는 'timestamp_influx'를 인덱스로 설정. 이를 'timestamp'로 변경하여 기존 코드 호환성 유지.
    if df.index.name == 'timestamp_influx':
        df.index.name = 'timestamp'
    df = df.sort_index() # 시간순 정렬
    return df

def load_funding_rate_data(reader: InfluxReader, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """InfluxDB에서 펀딩비 데이터를 로드합니다."""
    logging.info(f"Loading funding rate data for {symbol} from bucket {INFLUXDB_PARAMS.get('funding_rate_bucket', 'default_bucket')}...")
    
    funding_symbol = symbol.replace('/', '')

    # get_data는 timeframe 인자를 요구하지만, 펀딩비 조회 시에는 사용되지 않음. 플레이스홀더로 'N/A' 전달.
    funding_df = reader.get_data(
        bucket_name=INFLUXDB_PARAMS['funding_rate_bucket'],
        start=start_date, 
        end=end_date, 
        symbol=funding_symbol, 
        timeframe='N/A', # API 시그니처 준수를 위한 플레이스홀더
        data_type='funding_rate'
    )

    if funding_df.empty:
        logging.warning(f"No funding rate data found for {symbol} (query symbol: {funding_symbol}). Returning empty DataFrame.")
        return pd.DataFrame()

    # influx_reader에서 이미 'funding_rate'로 컬럼명이 변경되었으므로, 여기서는 해당 컬럼 존재 여부만 확인.
    if 'funding_rate' in funding_df.columns:
        # 필요한 컬럼만 선택하여 반환
        return funding_df[['funding_rate']]
    else:
        logging.error(f"'funding_rate' column not found in data returned for {symbol} (query symbol: {funding_symbol}).")
        return pd.DataFrame()

# --- 2. 데이터 병합 함수 ---
def merge_ohlcv_funding_data(ohlcv_df, funding_df):
    """OHLCV와 펀딩비 데이터프레임을 병합합니다."""
    if ohlcv_df.empty or funding_df.empty:
        logging.warning("OHLCV or Funding Rate data is empty, skipping merge.")
        return ohlcv_df # 또는 빈 데이터프레임 반환

    logging.info("Merging OHLCV and funding rate data...")
    # merge_asof를 위해 양쪽 데이터프레임의 인덱스(시간)가 정렬되어 있어야 함
    merged_df = pd.merge_asof(
        left=ohlcv_df.sort_index(),
        right=funding_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward' # 이전 값으로 채우기 (펀딩비는 다음 캔들 시작 전까지 유효)
    )
    # 펀딩비 데이터가 없는 초기 구간은 NaN이 될 수 있으므로 처리
    if 'funding_rate' in merged_df.columns:
        merged_df['funding_rate'].fillna(0, inplace=True) # 초기값 0으로 채움, 추후 전략 변경 가능
    return merged_df

# --- 3. 피처 생성 함수들 ---
# 기존 add_technical_indicators 함수는 유지

def generate_funding_rate_features(df):
    """펀딩비 관련 피처를 생성합니다."""
    if 'funding_rate' not in df.columns:
        logging.warning("'fundingRate' column not found. Skipping funding rate feature generation.")
        return df

    logging.info("Generating funding rate features...")
    df_copy = df.copy()
    df_copy['funding_rate_ma_3'] = df_copy['funding_rate'].rolling(window=3).mean()
    df_copy['funding_rate_ma_8'] = df_copy['funding_rate'].rolling(window=8).mean()
    df_copy['funding_rate_roc_1'] = df_copy['funding_rate'].diff(1) # 이전 펀딩비와의 차이
    df_copy['funding_rate_std_8'] = df_copy['funding_rate'].rolling(window=8).std()
    
    # 생성된 피처들의 초기 NaN 값 처리 (bfill 후 ffill 또는 특정 값으로 채우기)
    # 여기서는 bfill 후 ffill을 사용하나, 데이터 특성에 맞게 조정 필요
    cols_to_fill = ['funding_rate_ma_3', 'funding_rate_ma_8', 'funding_rate_roc_1', 'funding_rate_std_8']
    for col in cols_to_fill:
        if col in df_copy.columns:
            df_copy[col].fillna(method='bfill', inplace=True)
            df_copy[col].fillna(method='ffill', inplace=True) # bfill 후 남은 NaN은 ffill
            df_copy[col].fillna(0, inplace=True) # 그래도 남은 NaN은 0으로 (예: 전체 데이터가 window보다 작을 때)

    return df_copy

# 기존 get_daily_volatility 함수는 일단 유지 (사용 여부 확인 필요)
def get_daily_volatility(close, lookback=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]).astype(int)
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=lookback).std()
    return df0

# --- 4. 메인 피처 빌드 함수 ---
def build_feature_matrix(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    reader = InfluxReader() # InfluxReader는 내부적으로 .env에서 URL/Token/Org를 로드
    
    # 1. 1시간 OHLCV 데이터 로드 (베이스)
    # load_ohlcv_data_for_timeframe 함수가 내부적으로 config에서 measurement_name을 사용하므로, 여기서는 호출 시 measurement_name을 전달할 필요가 없음
    # 해당 함수의 measurement_name 파라미터는 이제 사용되지 않으므로, 호출부에서 제거하거나 함수 정의에서 제거 필요.
    # 우선 호출부에서 제거하고, 함수 정의는 그대로 둠 (다른 곳에서 사용될 가능성 고려)
    base_ohlcv_df = load_ohlcv_data_for_timeframe(reader, symbol, start_date, end_date, '1h')
    if base_ohlcv_df.empty:
        logging.error("Critical: 1h base OHLCV data is missing. Cannot build feature matrix.")
        return pd.DataFrame()

    # 2. 펀딩비 데이터 로드 및 1시간 OHLCV와 병합
    funding_rate_df = load_funding_rate_data(reader, symbol, start_date, end_date)
    merged_df = merge_ohlcv_funding_data(base_ohlcv_df, funding_rate_df)

    # 3. 기술적 지표 생성 (병합된 데이터프레임 기준, 단 OHLCV 컬럼만 사용)
    # add_technical_indicators는 'close' 등 특정 컬럼명을 사용하므로, 컬럼명 충돌 방지 위해
    # 원본 OHLCV 컬럼(open, high, low, close, volume)에 대해서만 TA 지표 계산
    # 또는, add_technical_indicators 함수가 접미사 없이 컬럼을 생성하도록 수정 필요
    # 여기서는 merged_df를 그대로 전달하고, add_technical_indicators가 'close' 등을 사용한다고 가정
    # 만약 add_technical_indicators가 이미 _1h 같은 접미사를 붙인다면, 그 전에 병합해야 함.
    # 현재 add_technical_indicators는 접미사를 붙이지 않음.
    
    # TA 지표는 원본 OHLCV에만 적용 후, 펀딩비 피처와 결합하는 것이 더 명확할 수 있음
    # 현재 구조: (OHLCV + FundingRate) -> TA Features -> FundingRate Features
    # 대안 구조: OHLCV -> TA Features. FundingRate -> FundingRate Features. (TA_OHLCV + FR_Features) merge.
    # 사용자 설계안은 전자를 따르므로, 그대로 진행.
    logging.info("Generating technical indicators for merged data...")
    merged_with_ta_df = add_technical_indicators(merged_df.copy()) # 원본 보존 위해 copy

    # 4. 펀딩비 피처 생성
    final_df_with_funding_features = generate_funding_rate_features(merged_with_ta_df)

    # 5. 상위 타임프레임 OHLCV 피처 로드 및 병합
    # 기존 feature_dfs 로직을 수정하여 상위 타임프레임 처리
    higher_timeframes = ['4h', '1d']
    current_base_df = final_df_with_funding_features.copy()

    print("\n--- Merging higher timeframe OHLCV features onto 1h base (with funding features) ---")
    for tf in higher_timeframes:
        df_higher_tf_ohlcv = load_ohlcv_data_for_timeframe(reader, symbol, start_date, end_date, tf)
        if df_higher_tf_ohlcv.empty:
            continue
        
        # 상위 타임프레임 데이터에도 TA 지표 추가
        df_higher_tf_features = add_technical_indicators(df_higher_tf_ohlcv)
        
        # 병합할 피처 컬럼들 선택 (OHLCV 및 생성된 TA 지표, 이미 _tf 접미사가 붙어있지 않음)
        # 따라서 여기서 접미사를 붙여줘야 함
        cols_to_rename = {c: f"{c}_{tf}" for c in df_higher_tf_features.columns if c not in ['symbol']} # 'symbol'은 제외
        df_higher_tf_features_renamed = df_higher_tf_features.rename(columns=cols_to_rename)
        
        # 실제 병합할 컬럼들 (원본 OHLCV + TA 지표, 모두 _tf 접미사가 붙음)
        feature_cols_to_merge = [c for c in df_higher_tf_features_renamed.columns if c.endswith(f'_{tf}')]
        temp_higher_tf_features_to_merge = df_higher_tf_features_renamed[feature_cols_to_merge]

        logging.info(f"Merging {tf} features using merge_asof (backward fill)...")
        current_base_df = pd.merge_asof(
            left=current_base_df.sort_index(), 
            right=temp_higher_tf_features_to_merge.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

    # 6. 최종 NaN 처리
    # ffill 후 bfill 또는 특정 전략 사용. 현재 코드에서는 ffill 후 dropna.
    # 펀딩비 피처 생성 시 bfill->ffill->0 처리했으므로, 여기서는 ffill 후 dropna 유지
    current_base_df.fillna(method='ffill', inplace=True)
    current_base_df.dropna(inplace=True)

    logging.info("\n--- Final Feature Matrix --- ")
    # 기존의 루프는 build_feature_matrix 함수 내의 상위 타임프레임 처리 로직과 중복/혼선되므로 제거.
    # 상위 타임프레임 처리는 아래의 higher_timeframes 루프에서 이미 수행됨.
    # for tf in timeframes: # 이 부분은 주석 처리 또는 삭제
    # 아래의 current_base_df가 최종 결과가 됨.

    # 최종 컬럼 순서 정리 (선택 사항, 일관성을 위해)
    if not current_base_df.empty:
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        funding_cols = [col for col in current_base_df.columns if 'funding' in col]
        ta_cols = [col for col in current_base_df.columns if col not in ohlcv_cols and col not in funding_cols and not any(tf_suffix in col for tf_suffix in [f'_{htf}' for htf in higher_timeframes]) and col != 'symbol']
        higher_tf_cols_ordered = []
        for htf in higher_timeframes:
            higher_tf_cols_ordered.extend(sorted([col for col in current_base_df.columns if f'_{htf}' in col]))
        
        final_column_order = ohlcv_cols + sorted(funding_cols) + sorted(ta_cols) + higher_tf_cols_ordered
        # current_base_df에 없는 컬럼이 final_column_order에 있을 수 있으므로, 있는 컬럼만 선택
        final_column_order = [col for col in final_column_order if col in current_base_df.columns]
        current_base_df = current_base_df[final_column_order]


    print(current_base_df.head())
    print(f"Shape: {current_base_df.shape}")
    return current_base_df

if __name__ == '__main__':
    # 로깅 설정 (DEBUG 레벨로 변경하여 상세 로그 확인)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s') # 포맷에 모듈명과 라인번호 추가하면 더 유용
    try:
        logging.info("--- Starting feature engineering script ---")
        SYMBOL = DATA_PARAMS.get('symbol', 'BTC/USDT') # config.py에서 심볼 가져오기
        START_DATE = '2024-05-01T00:00:00Z'
        END_DATE = '2024-05-07T23:59:59Z' # 테스트를 위해 7일 전체 포함
        final_features = build_feature_matrix(SYMBOL, START_DATE, END_DATE)
        if final_features is not None and not final_features.empty:
            output_dir = 'data/processed'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{SYMBOL.lower()}_feature_matrix.parquet')
            
            final_features.to_parquet(output_path)
            logging.info(f"\nFeature matrix saved successfully to: {output_path}")
        else:
            logging.warning("\nFeature matrix generation failed or resulted in an empty dataframe.")

    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
