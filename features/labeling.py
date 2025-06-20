import pandas as pd
import numpy as np
import pandas_ta as ta # TA-Lib 대신 pandas-ta 임포트
import os
from config.config import TRADING_PARAMS, DATA_PARAMS, PATH_PARAMS # config.py에서 설정값 가져오기

def generate_sample_data():
    # This function is now a stub and not used for the main path.
    # It's kept to avoid breaking any potential utility uses, though unlikely.
    print("INFO: generate_sample_data is a stub and returns an empty DataFrame.")
    return pd.DataFrame()


# get_triple_barrier_labels 함수 (기존 코드 유지)
def get_triple_barrier_labels(close, events, pt_sl, molecule):
    # pt_sl[0]: 이익 실현을 위한 trgt 대비 배수 (예: 2 -> trgt의 2배)
    # pt_sl[1]: 손절을 위한 trgt 대비 배수 (예: 1 -> trgt의 1배)
    store = []
    for i in molecule: # molecule은 이벤트 발생 시점의 인덱스 리스트
        trgt_val = events.loc[i, 'trgt'] # 'trgt'는 해당 시점의 변동성 또는 목표 가격 단위

        # 이익 실현 장벽 가격 계산
        profit_target_distance = trgt_val * pt_sl[0]
        profit_target_price = close[i] + profit_target_distance

        # 손절 장벽 가격 계산 (수정된 부분)
        stop_loss_distance = trgt_val * pt_sl[1] # pt_sl[1]은 양수여야 함
        stop_loss_price = close[i] - stop_loss_distance # 진입 가격에서 손절 거리만큼 차감

        # 가격 경로 (진입 시점부터 수직 장벽까지)
        path = close[i:events.loc[i, 't1']] # 't1'은 수직 장벽의 시간

        # 각 장벽 터치 시간 찾기
        upper_barrier_touch_time = path[path > profit_target_price].index.min()
        lower_barrier_touch_time = path[path < stop_loss_price].index.min() # 수정된 조건

        # 가장 먼저 도달한 장벽 시간 결정
        # 수직 장벽 시간(events.loc[i, 't1'])도 고려
        earliest_touch_time = pd.Series([
            lower_barrier_touch_time,
            upper_barrier_touch_time,
            events.loc[i, 't1']
        ]).dropna().min()

        # 레이블 결정
        if pd.isna(earliest_touch_time): # 어떤 장벽에도 닿지 않은 경우 (이론상 t1에 의해 발생 안함)
            bin_label = 0
            ret = 0 # 또는 path.loc[events.loc[i, 't1']] / close[i] - 1
        elif earliest_touch_time == upper_barrier_touch_time:
            bin_label = 1
            ret = path.loc[earliest_touch_time] / close[i] - 1
        elif earliest_touch_time == lower_barrier_touch_time:
            bin_label = -1
            ret = path.loc[earliest_touch_time] / close[i] - 1
        else: # 수직 장벽에 먼저 도달한 경우
            bin_label = 0
            ret = path.loc[earliest_touch_time] / close[i] - 1

        out = pd.DataFrame({
            'ret': ret,
            'bin': bin_label,
            't_event_end': earliest_touch_time # 이벤트 종료 시점 추가 (분석용)
        }, index=[i])
        store.append(out)

    if not store: # 만약 molecule이 비어있거나 어떤 이벤트도 처리되지 않았다면
        return pd.DataFrame(columns=['ret', 'bin', 't_event_end'])

    return pd.concat(store)

def get_trend_signals(df_ohlcv):
    """
    이동평균선 교차와 ADX를 사용하여 추세 기반 진입 신호(t0)를 생성합니다.
    config.py의 TRADING_PARAMS를 사용합니다.
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV 데이터프레임 (시간 인덱스 포함, 'high', 'low', 'close' 컬럼 필요)
        
    Returns:
        pd.Series: 진입 신호가 발생한 시점의 인덱스 (t0)
    """
    fast_ma_period = TRADING_PARAMS['fast_ma_period']
    slow_ma_period = TRADING_PARAMS['slow_ma_period']
    adx_period = TRADING_PARAMS['adx_period']
    adx_threshold = TRADING_PARAMS['adx_threshold']

    # 이동평균선 및 ADX 계산 (pandas-ta 사용)
    df_ohlcv['fast_ma'] = df_ohlcv.ta.sma(length=fast_ma_period)
    df_ohlcv['slow_ma'] = df_ohlcv.ta.sma(length=slow_ma_period)
    
    # pandas-ta의 adx는 ADX, DMP, DMN 값을 포함하는 데이터프레임을 반환합니다.
    adx_df = df_ohlcv.ta.adx(length=adx_period)
    df_ohlcv['adx'] = adx_df[f'ADX_{adx_period}']

    # 골든크로스: 단기 MA가 장기 MA를 상향 돌파
    golden_cross = (df_ohlcv['fast_ma'].shift(1) < df_ohlcv['slow_ma'].shift(1)) & \
                   (df_ohlcv['fast_ma'] > df_ohlcv['slow_ma'])
                   
    # 데드크로스: 단기 MA가 장기 MA를 하향 돌파 (숏 포지션 진입 신호로 활용 가능하나, 우선 롱만 고려)
    # dead_cross = (df_ohlcv['fast_ma'].shift(1) > df_ohlcv['slow_ma'].shift(1)) & \
    #              (df_ohlcv['fast_ma'] < df_ohlcv['slow_ma'])
                 
    # ADX 추세 강도 조건
    strong_trend = df_ohlcv['adx'] > adx_threshold
    
    # 최종 진입 신호 (골든크로스 & 강한 추세)
    # 현재는 롱 포지션 진입 신호만 생성합니다. 숏 포지션은 추후 확장 가능합니다.
    long_entry_signals = df_ohlcv[golden_cross & strong_trend].index
    
    return long_entry_signals

def calculate_daily_volatility(df_ohlcv):
    """
    일별 변동성(ATR)을 계산합니다. 'trgt' 계산에 사용됩니다.
    config.py의 TRADING_PARAMS를 사용합니다.
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV 데이터프레임 ('high', 'low', 'close' 컬럼 필요)
        
    Returns:
        pd.Series: 각 시점의 ATR 값 (trgt)
    """
    atr_period = TRADING_PARAMS['atr_period_for_trgt']
    
    # pandas-ta의 atr은 ATR 값을 포함하는 Series를 반환합니다.
    atr = df_ohlcv.ta.atr(length=atr_period)
    return atr

def add_vertical_barrier(t_events, df_ohlcv, num_candles_max_hold):
    """
    수직 장벽(t1)을 추가합니다. t1은 진입 시점으로부터 정해진 캔들 수 이후의 시간입니다.
    
    Args:
        t_events (pd.Series): 진입 시점(t0)의 인덱스
        df_ohlcv (pd.DataFrame): 전체 OHLCV 데이터프레임 (시간 인덱스 필요)
        num_candles_max_hold (int): 최대 보유 캔들 수
        
    Returns:
        pd.Series: 각 진입 시점에 대한 수직 장벽(t1)의 시간
    """
    t1 = df_ohlcv.index.searchsorted(t_events + pd.Timedelta(hours=num_candles_max_hold))
    t1 = t1[t1 < len(df_ohlcv.index)] # 데이터 범위를 벗어나는 인덱스 제거
    t1 = pd.Series(df_ohlcv.index[t1], index=t_events[:len(t1)]) # Series로 변환
    return t1

def create_events(df_ohlcv):
    """
    삼중 장벽 레이블링에 필요한 events 데이터프레임(t0, trgt, t1 포함)을 생성합니다.
    config.py의 TRADING_PARAMS와 DATA_PARAMS를 사용합니다.
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV 데이터 (피처가 추가되기 전 또는 후)
        
    Returns:
        pd.DataFrame: 'trgt', 't1' 컬럼을 포함하는 events 데이터프레임.
                      인덱스는 t0 (진입 시점)과 동일하게 설정.
    """
    # 1. 추세 기반 진입 신호 (t0) 생성
    trend_signals_indices = get_trend_signals(df_ohlcv)
    if trend_signals_indices.empty:
        return pd.DataFrame(columns=['trgt', 't1'])

    # 2. 변동성 (trgt) 계산
    volatility = calculate_daily_volatility(df_ohlcv)
    events_df = pd.DataFrame(volatility.loc[trend_signals_indices])
    events_df.columns = ['trgt']

    # 3. 수직 장벽 (t1) 추가
    max_hold_period = TRADING_PARAMS['num_candles_max_hold']
    vertical_barriers = add_vertical_barrier(events_df.index, df_ohlcv, max_hold_period)
    events_df['t1'] = vertical_barriers
    
    events_df.dropna(inplace=True) # trgt 또는 t1 계산 중 NaN 발생 가능성 제거
    return events_df


# The following block executes when the script is run directly.
if __name__ == '__main__':
    print("--- Running labeling script with actual feature data ---")
    # These TRADING_PARAMS and DATA_PARAMS are loaded from config.config at the top of the file.
    print(f"TRADING_PARAMS: {TRADING_PARAMS}")
    print(f"DATA_PARAMS: {DATA_PARAMS}")

    feature_matrix_path = os.path.join(PATH_PARAMS['data_path'], 'processed', 'btcusdt_feature_matrix.parquet')
    print(f"--- Loading feature matrix from: {feature_matrix_path} ---")
    try:
        df_features = pd.read_parquet(feature_matrix_path)
    except FileNotFoundError:
        print(f"ERROR: Feature matrix not found at '{feature_matrix_path}'")
        print("Please run features/build_features.py first.")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load feature matrix: {e}")
        exit(1)
    
    # Ensure timestamp is the index (build_features.py should already handle this)
    if not isinstance(df_features.index, pd.DatetimeIndex):
        if 'timestamp' in df_features.columns:
            df_features.set_index('timestamp', inplace=True)
            print("INFO: Set 'timestamp' column as DatetimeIndex for df_features.")
        elif not isinstance(df_features.index, pd.DatetimeIndex):
             print("ERROR: Loaded df_features.index is not a DatetimeIndex and 'timestamp' column not found for setting index.")
             exit(1)

    # df_sample is used by the subsequent original code (from original line 181 onwards).
    # It should contain 'open', 'high', 'low', 'close', 'volume', 'atr' for the main timeframe.
    # build_features.py ensures these columns are present for the primary timeframe.
    df_sample = df_features.copy() 
    
    # The original script's main logic (starting with print of df_sample.shape at original line 181)
    # continues below, now operating on the loaded df_sample.
    print(f"Initial df_sample.shape: {df_sample.shape}")
    print(f"Initial df_sample head:\n {df_sample.head()}")

    # events 생성 테스트
    # df_sample의 복사본을 전달하여 원본 데이터가 변경되지 않도록 함
    events_df = create_events(df_sample.copy())
    print(f"\nGenerated events_df (shape: {events_df.shape}):\n {events_df}")

    if not events_df.empty:
        # 레이블링 함수 테스트
        # molecule은 events_df의 인덱스 중 일부 또는 전체를 사용
        molecule_sample = events_df.index[:min(5, len(events_df))] # 처음 5개 이벤트만 테스트
        if not molecule_sample.empty:
            labels_df = get_triple_barrier_labels(df_sample['close'], events_df, TRADING_PARAMS['pt_sl_multipliers'], molecule_sample)
            print(f"\nGenerated labels_df (shape: {labels_df.shape}):\n {labels_df}")
        else:
            print("\nNo molecule_sample to test labeling.")
    else:
        print("\nNo events generated, skipping labeling test.")

    # 최종 데이터프레임 생성 (피처 + 레이블)
    # get_triple_barrier_labels 함수는 events_df가 비어있으면 빈 Series를 반환할 수 있으므로, 
    # events_df가 비어있지 않을 때만 레이블링을 시도하고 데이터를 병합합니다.
    if not events_df.empty:
        labels = get_triple_barrier_labels(df_sample['close'], events_df, TRADING_PARAMS['pt_sl_multipliers'], events_df.index)
        print(f"\nGenerated labels_df (shape: {labels.shape}):")
        print(labels.head())

        # 원본 데이터, 피처, 이벤트, 레이블을 모두 포함하는 최종 데이터프레임 생성
        # 't1'과 'trgt'는 events_df에 이미 있으므로, labels만 병합합니다.
        final_df = df_sample.join(events_df.drop(columns=['t1', 'trgt'], errors='ignore')).join(labels)
        print("\nFinal DataFrame with features and labels (first 5 rows with no NaN):")
        print(final_df.dropna().head())

        # 데이터 저장
        # Parquet 파일로 저장 (data/processed 폴더)
        import os
        # config.py의 PATH_PARAMS에서 'data_path'를 가져오고 'processed' 하위 폴더를 지정합니다.
        output_dir = os.path.join(PATH_PARAMS['data_path'], 'processed') 
        os.makedirs(output_dir, exist_ok=True)
        final_df_with_labels = final_df
        
        # final_df_with_labels가 실제로 데이터를 가지고 있을 때만 저장 시도
        if final_df_with_labels is not None and not final_df_with_labels.empty:
            # output_dir은 이전에 PATH_PARAMS를 통해 정확히 정의되어 있어야 함
            # (예: output_dir = os.path.join(PATH_PARAMS['data_path'], 'processed'))
            # (예: os.makedirs(output_dir, exist_ok=True)도 이미 호출됨)
            output_path = os.path.join(output_dir, 'labeled_btcusdt_data.parquet')

            # --- 여기서부터 디버깅 코드 ---
            print(f"Attempting to save to: {output_path}")
            print(f"Output directory: {output_dir}")
            
            if not os.path.exists(output_dir):
                print(f"Output directory {output_dir} does NOT exist.")
                try:
                    os.makedirs(output_dir, exist_ok=True) # 필요시 생성
                    print(f"Created directory: {output_dir}")
                except Exception as e:
                    print(f"Error creating directory {output_dir}: {e}")
            elif not os.path.isdir(output_dir):
                print(f"Error: {output_dir} exists but is NOT a directory.")
            else:
                print(f"Output directory {output_dir} exists and is a directory.")
                
            temp_file_path = os.path.join(output_dir, "temp_permission_check.tmp")
            try:
                with open(temp_file_path, "w") as f:
                    f.write("test")
                os.remove(temp_file_path)
                print(f"Successfully created and deleted a temporary file in {output_dir}. Write permission seems OK.")
            except Exception as e:
                print(f"Failed to create/delete a temporary file in {output_dir}. Write permission issue likely: {e}")
            # --- 여기까지 디버깅 코드 ---

            print(f"--- Saving labeled data to: {output_path} ---")
            try:
                final_df_with_labels.to_parquet(output_path, index=True)
                print(f"--- Successfully saved labeled data to {output_path} ---")
            except Exception as e:
                print(f"[Error] Failed to save data to Parquet: {e}")
        else:
            print("--- No data to save (final_df_with_labels is empty or None after processing events) ---")
    else: # events_df.empty 경우
        print("--- No events were generated, so no labels were created or saved. ---")
