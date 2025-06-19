import backtrader as bt
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier # 예시 모델

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.build_features import add_technical_indicators

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

    # 3. 가짜 데이터 생성 (실제로는 데이터 파이프라인에서 로드)
    print("\n--- Preparing dummy data for backtesting ---")
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=500, freq='D'))
    price_data = np.random.randn(500).cumsum() + 100
    df_data = pd.DataFrame({
        'open': price_data - np.random.rand(500) * 2,
        'high': price_data + np.random.rand(500) * 2,
        'low': price_data - np.random.rand(500) * 2,
        'close': price_data,
        'volume': np.random.randint(1000, 5000, size=500)
    }, index=dates)
    data_feed = bt.feeds.PandasData(dataname=df_data)
    cerebro.adddata(data_feed)

    # 4. 전략 추가
    cerebro.addstrategy(MLStrategy, model_path='models/model_v1.pkl')

    # 5. 초기 자본금 및 거래 비용 설정
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001) # 수수료 0.1%

    # 6. 분석기 추가
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # 7. 백테스트 실행
    print("\n--- Running Backtest ---")
    results = cerebro.run()
    strat = results[0]

    # 8. 결과 출력
    print("\n--- Backtest Finished ---")
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    analysis = strat.analyzers
    print(f"Sharpe Ratio: {analysis.sharpe_ratio.get_analysis()['sharperatio']:.2f}")
    print(f"Max Drawdown: {analysis.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    
    trade_analysis = analysis.trade_analyzer.get_analysis()
    if 'total' in trade_analysis and trade_analysis.total.total > 0:
        print(f"Total Trades: {trade_analysis.total.total}")
        print(f"Winning Trades: {trade_analysis.won.total}")
        print(f"Losing Trades: {trade_analysis.lost.total}")
        print(f"Win Rate: {trade_analysis.won.total / trade_analysis.total.total * 100:.2f}%")

    # 9. 그래프 출력
    cerebro.plot()
