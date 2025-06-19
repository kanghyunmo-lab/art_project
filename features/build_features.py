import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    주어진 데이터프레임에 기술적 지표를 계산하여 추가합니다.

    :param df: 'close' 컬럼을 포함하는 pandas DataFrame
    :return: 기술적 지표가 추가된 pandas DataFrame
    """
    df = df.copy()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal

    # Bollinger Bands
    df['bollinger_mavg_20d'] = df['close'].rolling(window=20).mean()
    df['bollinger_std_20d'] = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mavg_20d'] + (df['bollinger_std_20d'] * 2)
    df['bollinger_lower'] = df['bollinger_mavg_20d'] - (df['bollinger_std_20d'] * 2)

    return df

def get_daily_volatility(close, lookback=100):
    """
    일별 변동성을 계산합니다. (De Prado, 2018)
    이는 익절/손절 라인을 동적으로 설정하는 데 사용됩니다.

    :param close: 종가 데이터 pandas Series
    :param lookback: 변동성 계산 기간 (일)
    :return: 일별 변동성 pandas Series
    """
    # 일별 수익률 계산
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]).astype(int)
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Daily returns
    
    # 지수 가중 이동 표준편차 계산
    df0 = df0.ewm(span=lookback).std()
    return df0

def get_triple_barrier_labels(close, events, pt_sl, molecule):
    """
    삼중 장벽 방법을 사용하여 레이블을 생성합니다.
    (De Prado, 2018, Snippet 3.2)

    :param close: (pd.Series) 종가 데이터
    :param events: (pd.DataFrame) 다음 컬럼들을 포함:
        - 't1': 수직 장벽의 타임스탬프. np.nan이면 수직 장벽 없음.
        - 'trgt': 익절/손절 장벽의 단위를 나타내는 변동성 값.
    :param pt_sl: (list) 두 개의 non-negative float 값 리스트:
        - pt_sl[0]: 익절 배수 (profit taking)
        - pt_sl[1]: 손절 배수 (stop loss)
    :param molecule: (list) 단일 스레드에서 처리될 이벤트 인덱스 리스트 (병렬 처리를 위함)
    :return: (pd.DataFrame) 레이블이 지정된 이벤트
    """
    store = []
    for i in molecule:
        trgt_val = events.loc[i, 'trgt']
        trgt_upper = trgt_val * pt_sl[0]
        trgt_lower = -trgt_val * pt_sl[1]

        # 이벤트 발생 시점부터 수직 장벽까지의 가격 경로 추출
        path = close[i:events.loc[i, 't1']]
        
        # 상단 장벽 도달 시간
        upper_barrier_touch_time = path[path > close[i] + trgt_upper].index.min()
        
        # 하단 장벽 도달 시간
        lower_barrier_touch_time = path[path < close[i] - trgt_lower].index.min()

        # 각 장벽 도달 시간과 수직 장벽 시간을 비교하여 가장 먼저 발생한 이벤트 결정
        earliest_touch_time = pd.Series([lower_barrier_touch_time, upper_barrier_touch_time, events.loc[i, 't1']]).dropna().min()

        if earliest_touch_time == upper_barrier_touch_time:
            bin_label = 1
        elif earliest_touch_time == lower_barrier_touch_time:
            bin_label = -1
        else:
            bin_label = 0
        
        out = pd.DataFrame({
            'ret': path.loc[earliest_touch_time] / close[i] - 1,
            'bin': bin_label
        }, index=[i])
        store.append(out)
        
    return pd.concat(store)


if __name__ == '__main__':
    # --- 예제 사용법 ---
    # 1. 샘플 데이터 생성 (실제로는 데이터 파이프라인에서 로드)
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=500, freq='H'))
    price_data = np.random.randn(500).cumsum() + 50
    close_prices = pd.Series(data=price_data, index=dates)
    df = pd.DataFrame({'close': close_prices})

    # 2. 기술적 지표 추가
    df_features = add_technical_indicators(df)
    print("--- Features Added ---")
    print(df_features.tail())

    # 3. 삼중 장벽 레이블링 준비
    vol = get_daily_volatility(df_features['close'])
    
    # 거래 진입 시점(이벤트) 샘플링 (예: 특정 조건 만족 시)
    events = df_features[::10].dropna() # 10시간마다 진입
    events = events.assign(t1=df_features.index[df_features.index.searchsorted(events.index) + 24]) # 24시간 후 만료
    events = events.assign(trgt=vol.loc[events.index])
    events = events.dropna()

    # 4. 레이블 생성
    labels = get_triple_barrier_labels(df_features['close'], events, [1, 1], events.index)
    print("\n--- Triple Barrier Labels ---")
    print(labels.head())
