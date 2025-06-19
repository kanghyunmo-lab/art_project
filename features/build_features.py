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

def get_daily_volatility(close, lookback=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]).astype(int)
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=lookback).std()
    return df0

def build_feature_matrix(symbol, start_date, end_date):
    reader = InfluxReader()
    timeframes = ['1h', '4h', '1d']
    feature_dfs = {}
    for tf in timeframes:
        print(f"--- Processing {tf} data for {symbol} ---")
        measurement = f"{symbol}_{tf}"
        df = reader.get_data(measurement, start=start_date, end=end_date, symbol=symbol)
        if df.empty:
            print(f"Warning: No data for {tf}, skipping.")
            continue
        df_features = add_technical_indicators(df)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        df_features = df_features.rename(columns={
            c: f"{c}_{tf}" for c in df_features.columns if c not in ohlcv_cols
        })
        feature_dfs[tf] = df_features
    if '1h' not in feature_dfs:
        print("Error: 1h base data is missing. Cannot build feature matrix.")
        return pd.DataFrame()
    base_df = feature_dfs['1h'].sort_index() # merge_asof를 위해 시간 인덱스 기준으로 정렬
    print("\n--- Merging timeframe features onto 1h base ---")
    for tf in ['4h', '1d']:
        if tf in feature_dfs:
            print(f"Merging {tf} features using merge_asof (backward fill)...")
            # 상위 타임프레임 데이터도 시간 인덱스 기준으로 정렬
            df_higher_tf = feature_dfs[tf].sort_index()
            
            # 병합할 피처 컬럼들 선택 (예: rsi_14d_4h, macd_4h 등)
            feature_cols_to_merge = [c for c in df_higher_tf.columns if c.endswith(f'_{tf}')]
            temp_higher_tf_features = df_higher_tf[feature_cols_to_merge]

            # base_df는 이미 정렬된 상태임
            # merge_asof를 사용하여 look-ahead bias 없이 이전 시점의 데이터를 병합
            base_df = pd.merge_asof(
                left=base_df, 
                right=temp_higher_tf_features, # 선택된 피처 컬럼만 가진 DataFrame
                left_index=True,
                right_index=True,
                direction='backward' # 이전 시간의 값으로 채움
            )
    base_df.fillna(method='ffill', inplace=True)
    base_df.dropna(inplace=True)
    print("\n--- Final Feature Matrix ---")
    print(base_df.head())
    print(f"Shape: {base_df.shape}")
    return base_df

if __name__ == '__main__':
    try:
        print("--- Starting feature engineering script ---")
        SYMBOL = 'BTCUSDT'
        START_DATE = '2023-01-01T00:00:00Z'
        END_DATE = '2024-01-01T00:00:00Z'
        final_features = build_feature_matrix(SYMBOL, START_DATE, END_DATE)
        if not final_features.empty:
            output_dir = 'data/processed'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{SYMBOL.lower()}_feature_matrix.parquet')
            
            final_features.to_parquet(output_path)
            print(f"\nFeature matrix saved successfully to: {output_path}")
        else:
            print("\nFeature matrix generation failed or resulted in an empty dataframe.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
