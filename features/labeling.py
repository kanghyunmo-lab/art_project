import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import numpy as np
import pandas_ta as ta # TA-Lib 대신 pandas-ta 임포트
from config.config import TRADING_PARAMS, DATA_PARAMS, PATH_PARAMS # config.py에서 설정값 가져오기

def generate_sample_data():
    # This function is now a stub and not used for the main path.
    # It's kept to avoid breaking any potential utility uses, though unlikely.
    print("INFO: generate_sample_data is a stub and returns an empty DataFrame.")
    return pd.DataFrame()


# get_triple_barrier_labels 함수 (기존 코드 유지)
def get_triple_barrier_labels(close, events, pt_sl, molecule):
    """
    삼중 장벽 레이블링을 수행하는 함수
    
    Args:
        close (pd.Series): 종가 시리즈
        events (pd.DataFrame): 'trgt'와 't1' 컬럼을 포함한 이벤트 데이터프레임
        pt_sl (list): [이익 실현 배수, 손절 배수]
        molecule (list): 처리할 이벤트의 인덱스 리스트
        
    Returns:
        pd.DataFrame: 'ret', 'bin', 't_event_end' 컬럼을 가진 결과 데이터프레임
    """
    store = []
    total_events = len(molecule)
    processed_events = 0
    
    print(f"\n=== 삼중 장벽 레이블링 시작 (총 {total_events}개 이벤트) ===")
    print(f"사용 파라미터: pt_sl={pt_sl}")
    
    for i in molecule:
        processed_events += 1
        if processed_events % 100 == 0 or processed_events == 1:
            print(f"  처리 중: {processed_events}/{total_events} ({(processed_events/total_events*100):.1f}%)")
            
        # 디버깅 정보 초기화
        debug_info = {
            'event_idx': i,
            'entry_price': close[i],
            'trgt_val': None,
            'pt_price': None,
            'sl_price': None,
            't1': None,
            'path_length': 0,
            'path_min': None,
            'path_max': None,
            'upper_touch': None,
            'lower_touch': None,
            'result': None,
            'reason': None
        }
        
        try:
            trgt_val = events.loc[i, 'trgt']
            debug_info['trgt_val'] = trgt_val
            
            # 유효성 검사
            if pd.isna(trgt_val) or trgt_val <= 0:
                debug_info['result'] = 'SKIP'
                debug_info['reason'] = f'유효하지 않은 trgt 값: {trgt_val}'
                print(f"\n[이벤트 {i}] {debug_info['reason']}")
                continue
                
            # 장벽 가격 계산
            profit_target_distance = trgt_val * pt_sl[0]
            profit_target_price = close[i] + profit_target_distance
            stop_loss_distance = trgt_val * pt_sl[1]
            stop_loss_price = close[i] - stop_loss_distance
            
            debug_info.update({
                'pt_price': profit_target_price,
                'sl_price': stop_loss_price,
                't1': events.loc[i, 't1']
            })
            
            # 가격 경로 (진입 시점부터 수직 장벽까지)
            path = close[i:events.loc[i, 't1']]
            
            if path.empty:
                debug_info['result'] = 'SKIP'
                debug_info['reason'] = f'경로가 비어있음. t0: {i}, t1: {events.loc[i, "t1"]}'
                print(f"\n[이벤트 {i}] {debug_info['reason']}")
                continue
                
            debug_info.update({
                'path_length': len(path),
                'path_min': path.min(),
                'path_max': path.max()
            })
            
            # 각 장벽 터치 시간 찾기
            upper_barrier_touch_time = path[path >= profit_target_price].index.min()
            lower_barrier_touch_time = path[path <= stop_loss_price].index.min()
            
            debug_info.update({
                'upper_touch': upper_barrier_touch_time,
                'lower_touch': lower_barrier_touch_time
            })
            
            # 가장 먼저 도달한 장벽 시간 결정
            earliest_touch_time = pd.Series([
                lower_barrier_touch_time,
                upper_barrier_touch_time,
                events.loc[i, 't1']
            ]).dropna().min()
            
            # 레이블 결정
            if pd.isna(earliest_touch_time):
                bin_label = 0
                ret = 0
                debug_info.update({
                    'result': 'NEUTRAL',
                    'reason': '어떤 장벽에도 도달하지 않음',
                    't_event_end': events.loc[i, 't1']
                })
            elif earliest_touch_time == upper_barrier_touch_time:
                bin_label = 1
                ret = path.loc[earliest_touch_time] / close[i] - 1
                debug_info.update({
                    'result': 'LONG',
                    'reason': f'이익 실현 달성: {profit_target_price:.2f} 도달',
                    't_event_end': earliest_touch_time
                })
            elif earliest_touch_time == lower_barrier_touch_time:
                bin_label = -1
                ret = path.loc[earliest_touch_time] / close[i] - 1
                debug_info.update({
                    'result': 'SHORT',
                    'reason': f'손절 달성: {stop_loss_price:.2f} 도달',
                    't_event_end': earliest_touch_time
                })
            else:
                bin_label = 0
                ret = path.loc[earliest_touch_time] / close[i] - 1
                debug_info.update({
                    'result': 'NEUTRAL',
                    'reason': f'최대 보유 기간 도달: {events.loc[i, "t1"]}',
                    't_event_end': earliest_touch_time
                })
                
            # 디버깅 정보 출력 (처음 5개와 10개마다 1개씩 출력)
            if processed_events <= 5 or processed_events % 10 == 0:
                print(f"\n[이벤트 {i}] 결과: {debug_info['result']}")
                print(f"- 진입가: {debug_info['entry_price']:.2f}, 목표가: {debug_info['pt_price']:.2f} (+{(debug_info['pt_price']/debug_info['entry_price']-1)*100:.1f}%), "
                      f"손절가: {debug_info['sl_price']:.2f} ({(debug_info['sl_price']/debug_info['entry_price']-1)*100:.1f}%)")
                print(f"- 경로 길이: {debug_info['path_length']}개, 최소: {debug_info['path_min']:.2f}, 최대: {debug_info['path_max']:.2f}")
                print(f"- 상단 터치: {debug_info['upper_touch']}, 하단 터치: {debug_info['lower_touch']}")
                print(f"- 사유: {debug_info['reason']}")
                
            # 결과 저장
            out = pd.DataFrame({
                'ret': ret,
                'bin': bin_label,
                't_event_end': earliest_touch_time
            }, index=[i])
            store.append(out)
            
        except Exception as e:
            print(f"\n[오류] 이벤트 {i} 처리 중 예외 발생: {str(e)}")
            print(f"디버그 정보: {debug_info}")
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    if store:
        result_df = pd.concat(store)
        print("\n=== 레이블링 완료 ===")
        print(f"- 처리된 이벤트: {len(result_df)}/{total_events} ({(len(result_df)/total_events*100):.1f}%)")
        if not result_df.empty:
            label_dist = result_df['bin'].value_counts()
            print("- 레이블 분포:")
            for label, count in label_dist.items():
                print(f"  {label}: {count}개 ({(count/len(result_df)*100):.1f}%)")
        return result_df
    else:
        print("\n[경고] 처리된 이벤트가 없습니다.")
        return pd.DataFrame(columns=['ret', 'bin', 't_event_end'])

def get_trend_signals(df_ohlcv):
    """
    이동평균선 교차와 ADX, RSI를 사용하여 추세 기반 진입 신호(t0)를 생성합니다.
    config.py의 TRADING_PARAMS를 사용합니다.
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV 데이터프레임 (시간 인덱스 포함, 'high', 'low', 'close' 컬럼 필요)
        
    Returns:
        pd.Series: 진입 신호가 발생한 시점의 인덱스 (t0)와 방향(1: 롱, -1: 숏)을 포함한 튜플의 리스트
    """
    from collections import defaultdict
    import numpy as np
    
    # 파라미터 로드
    fast_ma_period = TRADING_PARAMS['fast_ma_period']
    slow_ma_period = TRADING_PARAMS['slow_ma_period']
    adx_period = TRADING_PARAMS['adx_period']
    adx_threshold = TRADING_PARAMS['adx_threshold']
    
    print(f"\n=== 트렌드 신호 생성 시작 (ADX 임계값: {adx_threshold}) ===")
    
    # 1. 이동평균선 계산
    df_ohlcv['fast_ma'] = df_ohlcv['close'].rolling(window=fast_ma_period, min_periods=1).mean()
    df_ohlcv['slow_ma'] = df_ohlcv['close'].rolling(window=slow_ma_period, min_periods=1).mean()
    
    # 2. ADX 및 DI 계산 (추세 강도 및 방향 확인)
    adx_df = df_ohlcv.ta.adx(length=adx_period)
    df_ohlcv['adx'] = adx_df[f'ADX_{adx_period}']
    df_ohlcv['plus_di'] = adx_df[f'DMP_{adx_period}']  # +DI (상승 추세 강도)
    df_ohlcv['minus_di'] = adx_df[f'DMN_{adx_period}']  # -DI (하락 추세 강도)
    
    # 3. RSI 추가 (과매수/과매도 확인용)
    rsi_period = 14
    df_ohlcv['rsi'] = df_ohlcv.ta.rsi(length=rsi_period)
    
    # 4. 추세 강도 및 방향 조건
    strong_uptrend = (df_ohlcv['adx'] > adx_threshold) & (df_ohlcv['plus_di'] > df_ohlcv['minus_di'])
    strong_downtrend = (df_ohlcv['adx'] > adx_threshold) & (df_ohlcv['minus_di'] > df_ohlcv['plus_di'])
    
    # 5. 이동평균선 교차 신호 (골든/데드 크로스)
    golden_cross = (df_ohlcv['fast_ma'].shift(1) <= df_ohlcv['slow_ma'].shift(1)) & \
                   (df_ohlcv['fast_ma'] > df_ohlcv['slow_ma'])
    dead_cross = (df_ohlcv['fast_ma'].shift(1) >= df_ohlcv['slow_ma'].shift(1)) & \
                 (df_ohlcv['fast_ma'] < df_ohlcv['slow_ma'])
    
    # 6. RSI 조건 (과매수/과매도 회피)
    oversold = df_ohlcv['rsi'] < 30  # 과매도 영역
    overbought = df_ohlcv['rsi'] > 70  # 과매수 영역
    
    # 7. 신호 생성
    # 롱 신호: 골든크로스 + 상승추세 + 과매도 아님
    long_signals = df_ohlcv[golden_cross & strong_uptrend & ~overbought].index
    # 숏 신호: 데드크로스 + 하락추세 + 과매도 아님
    short_signals = df_ohlcv[dead_cross & strong_downtrend & ~oversold].index
    
    # 8. 디버깅 정보 출력
    print(f"- 생성된 롱 신호: {len(long_signals)}개")
    print(f"- 생성된 숏 신호: {len(short_signals)}")
    
    if len(long_signals) > 0:
        print("  롱 신호 예시:", long_signals[:3].tolist(), "..." if len(long_signals) > 3 else "")
    if len(short_signals) > 0:
        print("  숏 신호 예시:", short_signals[:3].tolist(), "..." if len(short_signals) > 3 else "")
    
    # 9. 모든 신호를 하나의 시리즈로 결합 (방향 정보 포함)
    signals = []
    for idx in long_signals:
        signals.append((idx, 1))  # 1은 롱 포지션
    for idx in short_signals:
        signals.append((idx, -1))  # -1은 숏 포지션
    
    # 시간 순으로 정렬
    signals.sort(key=lambda x: x[0])
    
    print(f"- 총 생성된 신호 수: {len(signals)} (롱: {len(long_signals)}, 숏: {len(short_signals)})")
    
    return signals

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
    # 디버깅 로그 추가
    print(f"  - 최대 보유 기간: {num_candles_max_hold}시간")
    print(f"  - 입력된 t_events 수: {len(t_events)}")
    
    # t1 계산: 각 t0로부터 num_candles_max_hold 시간 후의 시간
    t1_indices = df_ohlcv.index.searchsorted(t_events + pd.Timedelta(hours=num_candles_max_hold))
    
    # 데이터 범위를 벗어나는 인덱스 제거
    valid_mask = t1_indices < len(df_ohlcv.index)
    t1_indices = t1_indices[valid_mask]
    t_events = t_events[valid_mask]  # 유효한 t_events만 유지
    
    # Series로 변환 (인덱스: t_events, 값: t1)
    t1_series = pd.Series(df_ohlcv.index[t1_indices], index=t_events)
    
    # 디버깅 정보 출력
    print(f"  - 유효한 t1 개수: {len(t1_series)}")
    if len(t1_series) > 0:
        print(f"  - t1 범위: {t1_series.iloc[0]} ~ {t1_series.iloc[-1]}")
    
    return t1_series

def create_events(df_ohlcv):
    """
    삼중 장벽 레이블링에 필요한 events 데이터프레임(t0, trgt, t1, side 포함)을 생성합니다.
    config.py의 TRADING_PARAMS와 DATA_PARAMS를 사용합니다.
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV 데이터 (피처가 추가되기 전 또는 후)
        
    Returns:
        pd.DataFrame: 'trgt', 't1', 'side' 컬럼을 포함하는 events 데이터프레임.
                      인덱스는 t0 (진입 시점)과 동일하게 설정.
                      'side'는 1(롱) 또는 -1(숏)의 값을 가집니다.
    """
    print("\n=== 이벤트 생성 시작 ===")
    
    # 1. 변동성(trgt) 계산
    print("1. 변동성(ATR) 계산 중...")
    df_ohlcv['trgt'] = calculate_daily_volatility(df_ohlcv)
    
    # 2. 추세 기반 진입 신호(t0) 및 포지션 방향(side) 생성
    print("2. 추세 신호 생성 중...")
    signals = get_trend_signals(df_ohlcv)  # (timestamp, side) 튜플 리스트 반환
    
    if not signals:
        print("경고: 생성된 신호가 없습니다.")
        return pd.DataFrame(columns=['t1', 'trgt', 'side'])
    
    # 신호를 데이터프레임으로 변환
    t_events = [s[0] for s in signals]  # 타임스탬프 추출
    sides = [s[1] for s in signals]     # 포지션 방향 추출
    
    # 3. 수직 장벽(t1) 추가
    print("3. 수직 장벽 계산 중...")
    num_candles_max_hold = TRADING_PARAMS['num_candles_max_hold']
    t1 = add_vertical_barrier(pd.DatetimeIndex(t_events), df_ohlcv, num_candles_max_hold)
    
    # 4. events 데이터프레임 생성
    print("4. 이벤트 데이터프레임 생성 중...")
    events = pd.DataFrame(index=t_events)
    events['t1'] = t1
    events['trgt'] = df_ohlcv.loc[t_events, 'trgt'].values
    events['side'] = sides  # 포지션 방향 추가 (1: 롱, -1: 숏)
    
    # 5. 유효한 이벤트만 필터링 (t1이 있는 경우만)
    valid_events = events.dropna(subset=['t1'])
    
    # 6. 디버깅 정보 출력
    print(f"\n=== 이벤트 생성 완료 ===")
    print(f"- 총 생성된 이벤트: {len(valid_events)}개")
    if len(valid_events) > 0:
        long_count = (valid_events['side'] == 1).sum()
        short_count = (valid_events['side'] == -1).sum()
        print(f"  - 롱 포지션: {long_count}개 ({long_count/len(valid_events)*100:.1f}%)")
        print(f"  - 숏 포지션: {short_count}개 ({short_count/len(valid_events)*100:.1f}%)")
        print("\n이벤트 샘플:")
        print(valid_events.head())
    
    return valid_events

# The following block executes when the script is run directly.
if __name__ == '__main__':
    print("--- Running labeling script with actual feature data ---")
    # These TRADING_PARAMS and DATA_PARAMS are loaded from config.config at the top of the file.
    print(f"TRADING_PARAMS: {TRADING_PARAMS}")
    print(f"DATA_PARAMS: {DATA_PARAMS}")

    # 훈련/테스트 데이터 구분 처리
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # 훈련 데이터 로드
        feature_matrix_path = os.path.join(PATH_PARAMS['data_path'], 'processed', 'btcusdt_feature_matrix_train.parquet')
        print(f"--- Loading TRAIN feature matrix from: {feature_matrix_path} ---")
        data_type = 'train'
    else:
        # 테스트 데이터 로드 (기본값)
        feature_matrix_path = os.path.join(PATH_PARAMS['data_path'], 'processed', 'btcusdt_feature_matrix_test.parquet')
        print(f"--- Loading TEST feature matrix from: {feature_matrix_path} ---")
        data_type = 'test'

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
    
    # 데이터 타입에 따른 날짜 범위 설정
    if data_type == 'train':
        # 훈련 데이터 처리
        TRAIN_START = DATA_PARAMS.get('train_start_date', '2017-01-01T00:00:00Z')
        TRAIN_END = DATA_PARAMS.get('train_end_date', '2023-12-31T23:59:59Z')
        df_sample = df_sample.loc[TRAIN_START:TRAIN_END]
        print(f"[INFO] TRAIN set range: {TRAIN_START} ~ {TRAIN_END}, shape after slice: {df_sample.shape}")
        output_suffix = 'train'
    else:
        # 테스트 데이터 처리 (기본값)
        TEST_START = DATA_PARAMS.get('test_start_date', '2024-01-01T00:00:00Z')
        TEST_END = DATA_PARAMS.get('test_end_date', '2025-12-31T23:59:59Z')
        df_sample = df_sample.loc[TEST_START:TEST_END]
        print(f"[INFO] TEST set range: {TEST_START} ~ {TEST_END}, shape after slice: {df_sample.shape}")
        output_suffix = 'test'

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
        
        # 'bin' 컬럼에서 NaN 값이 있는 행 제거
        if 'bin' in final_df_with_labels.columns:
            print(f"Before dropping NaNs in 'bin' column, shape: {final_df_with_labels.shape}")
            final_df_with_labels = final_df_with_labels.dropna(subset=['bin'])
            print(f"After dropping NaNs in 'bin' column, shape: {final_df_with_labels.shape}")
        else:
            print("Warning: 'bin' column not found in final_df_with_labels")
        
        # --- train feature matrix 컬럼 순서/구조 강제 맞춤 ---
        train_columns_path = os.path.join(output_dir, 'btcusdt_feature_matrix_columns.txt')
        if os.path.exists(train_columns_path):
            with open(train_columns_path, 'r', encoding='utf-8') as f:
                train_columns = [line.strip() for line in f.readlines() if line.strip()]
            # 누락된 컬럼은 NaN으로 추가
            for col in train_columns:
                if col not in final_df_with_labels.columns:
                    final_df_with_labels[col] = np.nan
            # label 컬럼들 분리
            label_cols = [c for c in final_df_with_labels.columns if c not in train_columns]
            # 컬럼 순서: train_columns + label_cols
            final_df_with_labels = final_df_with_labels[train_columns + label_cols]

        # final_df_with_labels가 실제로 데이터를 가지고 있을 때만 저장 시도
        if final_df_with_labels is not None and not final_df_with_labels.empty:
            # output_dir은 이전에 PATH_PARAMS를 통해 정확히 정의되어 있어야 함
            # (예: output_dir = os.path.join(PATH_PARAMS['data_path'], 'processed'))
            # (예: os.makedirs(output_dir, exist_ok=True)도 이미 호출됨)
            # 훈련/테스트에 따른 파일명 결정
            if output_suffix == 'train':
                output_path = os.path.join(output_dir, 'labeled_btcusdt_data_train.parquet')
            else:
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
        
            # --- 기존 파일명 형식으로도 저장 ---
            if output_suffix == 'train':
                legacy_path = os.path.join(output_dir, 'btcusdt_labeled_features_train.parquet')
            else:
                legacy_path = os.path.join(output_dir, 'btcusdt_labeled_features_test.parquet')

            try:
                final_df_with_labels.to_parquet(legacy_path, index=True)
                print(f"--- Successfully saved labeled data to {legacy_path} ---")
            except Exception as e:
                print(f"[Error] Failed to save data to Parquet: {e}")
        else:
            print("--- No data to save (final_df_with_labels is empty or None after processing events) ---")
    else: # events_df.empty 경우
        print("--- No events were generated, so no labels were created or saved. ---")
