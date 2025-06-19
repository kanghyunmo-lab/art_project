import backtrader as bt
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier # 예시 모델

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.build_features import add_technical_indicators
from data_pipeline.collector import BinanceDataCollector # 데이터 로더 임포트
from config.credentials import INFLUXDB_BUCKET # InfluxDB 버킷 정보 임포트

class MLStrategy(bt.Strategy):
    """
    머신러닝 모델의 예측 신호에 따라 거래하는 backtrader 전략
    """
    params = (
        ('model_path', 'models/model_v1.pkl'), # 훈련된 모델 파일 경로
    )

    def __init__(self):
        """
        전략을 초기화하고, 훈련된 모델을 로드합니다.
        """
        self.dataclose = self.datas[0].close
        self.order = None

        # 훈련된 모델 로드
        try:
            with open(self.p.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.p.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.p.model_path}. Please train and save the model first.")
            self.model = None
            self.cerebro.runstop() # 모델 없으면 중단

    def next(self):
        """
        각 bar(시간 단계)마다 호출되는 메서드
        """
        if self.order or self.model is None:
            return # 보류 중인 주문이 있거나 모델이 없으면 아무것도 하지 않음

        # 현재까지의 데이터를 DataFrame으로 변환
        # backtrader의 데이터를 pandas로 변환하기 위해 약간의 트릭이 필요
        dates = [bt.num2date(self.datas[0].datetime[i]) for i in range(-len(self.datas[0]), 0)]
        closes = self.datas[0].close.get(size=len(self.datas[0]))
        df = pd.DataFrame({'close': closes}, index=pd.to_datetime(dates))

        # 피처 엔지니어링
        df_features = add_technical_indicators(df)
        
        # 마지막 행 (현재 시점)의 피처를 모델 입력으로 사용
        current_features = df_features.iloc[-1:].drop(columns=['close'])
        
        if current_features.isnull().values.any():
            return # 피처 계산에 충분한 데이터가 없으면 건너뛰기

        # 모델 예측
        prediction = self.model.predict(current_features)[0]

        # 거래 로직
        if not self.position: # 포지션이 없으면
            if prediction == 1: # 매수 신호
                print(f"{self.datas[0].datetime.date(0)}: BUY signal received. Creating order.")
                self.order = self.buy()
            elif prediction == -1: # 매도 신호
                print(f"{self.datas[0].datetime.date(0)}: SELL signal received. Creating order.")
                self.order = self.sell()
        else: # 포지션이 있으면
            if self.position.size > 0 and prediction == -1: # 롱 포지션 + 매도 신호
                print(f"{self.datas[0].datetime.date(0)}: CLOSE LONG signal received. Closing position.")
                self.order = self.close()
            elif self.position.size < 0 and prediction == 1: # 숏 포지션 + 매수 신호
                print(f"{self.datas[0].datetime.date(0)}: CLOSE SHORT signal received. Closing position.")
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                print(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')

        self.order = None


if __name__ == '__main__':
    # --- 백테스트 준비 ---
    # 1. Cerebro 엔진 생성
    cerebro = bt.Cerebro()

    # 2. 가짜 모델 생성 및 저장 (실제로는 훈련된 모델을 사용해야 함)
    print("--- Creating a dummy model for demonstration ---")
    if not os.path.exists('models'): os.makedirs('models')
    dummy_features = pd.DataFrame(np.random.rand(100, 5), columns=['rsi_14d', 'macd', 'macd_signal', 'macd_hist', 'bollinger_mavg_20d'])
    dummy_labels = np.random.choice([-1, 0, 1], size=100)
    dummy_model = RandomForestClassifier().fit(dummy_features, dummy_labels)
    with open('models/model_v1.pkl', 'wb') as f:
        pickle.dump(dummy_model, f)
    print("Dummy model 'models/model_v1.pkl' created.")

    # 3. InfluxDB에서 실제 데이터 로드
    print("\n--- Loading data from InfluxDB for backtesting ---")
    try:
        data_collector = BinanceDataCollector()
        # 최근 1년치 데이터를 불러옵니다. 필요에 따라 기간을 조정할 수 있습니다.
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        
        print(f"Fetching data from {start_time} to {end_time}...")
        
        df_data = data_collector.query_data_from_influxdb(
            bucket=INFLUXDB_BUCKET,
            measurement='crypto_prices_hourly',
            symbol='BTCUSDT',
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )

        if df_data.empty:
            print("Error: No data loaded from InfluxDB. Please check the following:")
            print("1. InfluxDB 서비스가 실행 중인지 확인하세요.")
            print("2. config/credentials.py 파일에 올바른 InfluxDB 설정이 있는지 확인하세요.")
            print("3. 지정한 버킷과 측정값이 존재하는지 확인하세요.")
            sys.exit(1)
            
        print(f"Successfully loaded {len(df_data)} records from InfluxDB.")
        print(f"Date range: {df_data.index.min()} to {df_data.index.max()}")
        
        # 데이터 전처리 (필요한 컬럼만 선택하고, 인덱스 설정)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df_data.columns]
            print(f"Error: Missing required columns in the data: {missing}")
            sys.exit(1)
            
        df_data = df_data[required_columns].dropna()
        
        # backtrader용 데이터 피드 생성
        data_feed = bt.feeds.PandasData(
            dataname=df_data,
            datetime=None,  # 인덱스를 datetime으로 사용
            open=0,        # open price 컬럼 인덱스
            high=1,        # high price 컬럼 인덱스
            low=2,         # low price 컬럼 인덱스
            close=3,       # close price 컬럼 인덱스
            volume=4,      # volume 컬럼 인덱스
            openinterest=-1  # 사용 안 함
        )
        
        cerebro.adddata(data_feed)
        
    except Exception as e:
        print(f"Error loading data from InfluxDB: {str(e)}")
        print("\n추가 정보:")
        print("1. InfluxDB 서비스가 실행 중인지 확인하세요.")
        print("2. config/credentials.py 파일에 올바른 InfluxDB 설정이 있는지 확인하세요.")
        print(f"3. 버킷 '{INFLUXDB_BUCKET}'과 측정값 'crypto_prices_hourly'가 존재하는지 확인하세요.")
        sys.exit(1)

    # 4. 전략 추가
    print("\n--- Adding Strategy ---")
    cerebro.addstrategy(MLStrategy, model_path='models/model_v1.pkl')

    # 5. 초기 자본금 및 거래 비용 설정
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 수수료 0.1%
    print(f"Initial Portfolio Value: {initial_cash:.2f}")

    # 6. 분석기 추가
    print("\n--- Adding Analyzers ---")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # 7. 백테스트 실행
    print("\n--- Running Backtest ---")
    try:
        results = cerebro.run()
        strat = results[0]

        # 8. 결과 출력
        print("\n=== Backtest Results ===")
        print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        print(f"Net Profit: {cerebro.broker.getvalue() - initial_cash:.2f} ({(cerebro.broker.getvalue()/initial_cash - 1)*100:.2f}%)")
        
        # 분석 결과 출력
        print("\n--- Performance Analysis ---")
        
        # 샤프 지수
        sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
        print(f"Sharpe Ratio: {sharpe_ratio.get('sharperatio', 0):.2f}")
        
        # 최대 낙폭
        drawdown = strat.analyzers.drawdown.get_analysis()
        print(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
        print(f"Max Drawdown Period: {drawdown['max']['len']} bars")
        
        # 수익률 분석
        returns = strat.analyzers.returns.get_analysis()
        print(f"\nReturn: {returns['rtot']*100:.2f}%")
        print(f"Annual Return: {returns['rnorm100']:.2f}%")
        
        # 거래 분석
        trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
        if hasattr(trade_analysis, 'total') and trade_analysis.total.total > 0:
            print("\n--- Trade Analysis ---")
            print(f"Total Trades: {trade_analysis.total.total}")
            print(f"Winning Trades: {getattr(trade_analysis.won, 'total', 0)}")
            print(f"Losing Trades: {getattr(trade_analysis.lost, 'total', 0)}")
            print(f"Win Rate: {getattr(trade_analysis.won, 'total', 0) / trade_analysis.total.total * 100:.2f}%")
            print(f"Average Win: {getattr(trade_analysis.won, 'pnl', {}).get('average', 0):.2f}")
            print(f"Average Loss: {abs(getattr(trade_analysis.lost, 'pnl', {}).get('average', 0)):.2f}")
            print(f"Profit Factor: {getattr(trade_analysis, 'pnl', {}).get('gross', {}).get('total', 0) / abs(getattr(trade_analysis, 'pnl', {}).get('gross', {}).get('total', 1)):.2f}")
        else:
            print("\nNo trades were made during the backtest period.")
            
        # 차트 그리기 (선택 사항)
        # cerebro.plot(style='candlestick')
            
    except Exception as e:
        print(f"\nError during backtest: {str(e)}")
        import traceback
        traceback.print_exc()

        # 9. 그래프 출력
        cerebro.plot(style='candlestick', volume=True, barup='green', bardown='red')
