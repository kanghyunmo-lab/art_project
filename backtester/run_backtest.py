# backtester/run_backtest.py
import os
import sys
import logging
import pandas as pd
import joblib
import backtrader as bt
from datetime import datetime

# --- Project Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project-specific Imports ---
from config.config import BACKTESTER_PARAMS

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'backtest.log')),
        logging.StreamHandler()
    ]
)

class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('data_df', None),
    )

    def __init__(self):
        if self.p.model is None:
            raise ValueError("Model not provided to the strategy")
        if self.p.data_df is None:
            raise ValueError("Dataframe not provided to the strategy")

        self.model = self.p.model
        self.feature_columns = self.model.feature_names_in_
        self.p.data_df.index = pd.to_datetime(self.p.data_df.index)
        self.order = None
        self.position_size = 0
        self.leverage = 1
        self.entry_price = None
        self.liquidation_buffer = 0.005  # 0.5% 여유 (강제청산 방지)
        self.stop_loss = None
        self.trailing_stop = None
        self.highest_price = None
        self.lowest_price = None
        self.max_loss_pct = 0.05  # 최대 5% 손실 제한
        self.atr_mult = 2  # ATR 손절 배수

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.position_size = 1
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
                self.lowest_price = order.executed.price
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}, Leverage: {self.leverage}x')
            else:
                self.position_size = 0
                self.entry_price = None
                self.stop_loss = None
                self.trailing_stop = None
                self.highest_price = None
                self.lowest_price = None
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        logging.info(f'{dt.isoformat()} - {txt}')

    def _long_trend_filter(self, row):
        # 4h, 1d MACD, MA, RSI 등 활용 (상승장: MACD>0, MA>MA_4h, RSI>50)
        macd_4h = row.get('macd_4h', 0)
        macd_1d = row.get('macd_1d', 0)
        ma_1h = row.get('bollinger_mavg_20d', 0)
        ma_4h = row.get('bollinger_mavg_20d_4h', 0)
        ma_1d = row.get('bollinger_mavg_20d_1d', 0)
        rsi_4h = row.get('rsi_14d_4h', 50)
        rsi_1d = row.get('rsi_14d_1d', 50)
        # 상승장: MACD>0, MA_1h>MA_4h, MA_4h>MA_1d, RSI>50
        return (macd_4h > 0) and (macd_1d > 0) and (ma_1h > ma_4h > ma_1d) and (rsi_4h > 50) and (rsi_1d > 50)

    def _short_trend_filter(self, row):
        macd_4h = row.get('macd_4h', 0)
        macd_1d = row.get('macd_1d', 0)
        ma_1h = row.get('bollinger_mavg_20d', 0)
        ma_4h = row.get('bollinger_mavg_20d_4h', 0)
        ma_1d = row.get('bollinger_mavg_20d_1d', 0)
        rsi_4h = row.get('rsi_14d_4h', 50)
        rsi_1d = row.get('rsi_14d_1d', 50)
        # 하락장: MACD<0, MA_1h<MA_4h, MA_4h<MA_1d, RSI<50
        return (macd_4h < 0) and (macd_1d < 0) and (ma_1h < ma_4h < ma_1d) and (rsi_4h < 50) and (rsi_1d < 50)

    def _takeprofit_signal(self, row, price):
        # 이동평균선 이탈, RSI 과매수/과매도 등
        ma_1h = row.get('bollinger_mavg_20d', 0)
        rsi_1h = row.get('rsi_14d', 50)
        # 익절 조건: 가격이 MA 아래로 이탈 or RSI>80(과매수) or RSI<20(과매도)
        if price < ma_1h or rsi_1h > 80 or rsi_1h < 20:
            return True
        return False

    def next(self):
        if self.order:
            return
        current_dt = pd.Timestamp(self.data.datetime.datetime(0))
        if current_dt in self.p.data_df.index:
            try:
                row = self.p.data_df.loc[current_dt]
                features = row[self.feature_columns].values.reshape(1, -1)
                prediction = self.model.predict(features)[0]
                # 신뢰도(softmax 확률) 추출
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features)[0]
                    conf = max(proba)  # 예측 클래스의 확률
                else:
                    conf = 0.5  # fallback
                # 신뢰도 기반 레버리지 결정
                if conf >= 0.9:
                    self.leverage = 10
                elif conf >= 0.7:
                    self.leverage = 5
                else:
                    self.leverage = 2
                cash = self.broker.getcash()
                value = self.broker.getvalue()
                price = self.data.close[0]
                # ATR 기반 손절폭 계산
                atr = row.get('atr_14d', 0)
                # 레버리지 반영 주문 크기 계산 (1% * 레버리지)
                if price > 0:
                    size = (value * 0.01 * self.leverage) / price
                else:
                    size = 0
                # 진입 시 손절/트레일링스탑 초기화
                if prediction == 2 and self.position_size == 0 and size > 0:
                    if self._long_trend_filter(row):
                        self.stop_loss = price - self.atr_mult * atr if atr > 0 else price * (1 - self.max_loss_pct)
                        self.trailing_stop = self.stop_loss
                        self.highest_price = price
                        self.log(f'BUY SIGNAL (MFT CONFIRM) - Price: {price:.2f}, Size: {size:.4f}, SL: {self.stop_loss:.2f}')
                        self.order = self.buy(size=size)
                    else:
                        self.log(f'BUY SIGNAL IGNORED (Long trend filter not passed)')
                    return
                # 포지션 보유 중 동적 손절/익절 체크
                if self.position_size > 0 and self.entry_price:
                    # Trailing Stop: 최고가 갱신 시 손절 라인도 상승
                    if self.highest_price is None or price > self.highest_price:
                        self.highest_price = price
                        if atr > 0:
                            self.trailing_stop = max(self.trailing_stop, price - self.atr_mult * atr)
                    # 절대 최대 손실 제한
                    max_stop = self.entry_price * (1 - self.max_loss_pct)
                    # 손절 라인 중 가장 높은 값 적용
                    effective_stop = max(self.stop_loss or 0, self.trailing_stop or 0, max_stop)
                    # 손절 조건
                    if price <= effective_stop:
                        self.log(f'STOP LOSS TRIGGERED! Price: {price:.2f} <= Stop: {effective_stop:.2f}')
                        self.order = self.close()
                        return
                    # 강제 청산(마진콜) 시뮬레이션
                    liq_price = self.entry_price * (1 - (1/self.leverage) - self.liquidation_buffer)
                    if price <= liq_price:
                        self.log(f'LIQUIDATION TRIGGERED! Price: {price:.2f} <= Liq.Price: {liq_price:.2f} (Lev:{self.leverage}x)')
                        self.order = self.close()
                        return
                    # 추세 반전 익절 조건
                    if self._takeprofit_signal(row, price):
                        self.log(f'TAKE PROFIT (Trend Reversal) - Price: {price:.2f}')
                        self.order = self.close()
                        return
                # 매도 신호(청산) - 기존 필터 유지
                if prediction == 0 and self.position_size > 0:
                    if self._short_trend_filter(row):
                        pos_size = self.position.size
                        self.log(f'SELL SIGNAL (MFT CONFIRM) - Price: {price:.2f}, Size: {pos_size:.4f}')
                        self.order = self.close()
                    else:
                        self.log(f'SELL SIGNAL IGNORED (Short trend filter not passed)')
                # Hold(1) - do nothing
            except Exception as e:
                self.log(f'Error in next(): {str(e)}')
                if 'features' in locals():
                    self.log(f'Feature values: {features}')

def run_backtest():
    """Main function to run the backtest."""
    logging.info("--- Starting Backtest ---")
    
    # 1. Load Data
    data_path = os.path.join(PROJECT_ROOT, BACKTESTER_PARAMS['data_path'])
    try:
        df = pd.read_parquet(data_path)
        df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)
        logging.info(f"Successfully loaded data from {data_path}. Shape: {df.shape}")

        # =================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 디버깅 코드 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # =================================================================
        logging.info("\n--- DEBUG: Columns in loaded DataFrame ---")
        logging.info(df.columns.tolist())
        logging.info("------------------------------------------\n")
        # =================================================================
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 디버깅 코드 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # =================================================================
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}. Please run feature engineering first.")
        return

    # 2. Load Model
    model_path = os.path.join(PROJECT_ROOT, BACKTESTER_PARAMS['model_path'])
    try:
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please train the model first.")
        return

    # 3. Setup Backtrader
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(MLStrategy, model=model, data_df=df)

    # Set initial capital and commission
    cerebro.broker.setcash(BACKTESTER_PARAMS['initial_cash'])
    cerebro.broker.setcommission(commission=BACKTESTER_PARAMS['commission'])

    # 4. Run Backtest
    initial_value = cerebro.broker.getvalue()
    logging.info(f"Starting portfolio value: {initial_value:,.2f}")
    
    results = cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    logging.info(f"Final portfolio value: {final_value:,.2f}")
    returns = (final_value - initial_value) / initial_value * 100
    logging.info(f"Total Return: {returns:.2f}%")

    # 5. Plot and Save Results with error handling
    plot_path = os.path.join(PROJECT_ROOT, 'backtester', 'backtest_result.png')
    logging.info(f"Attempting to save backtest plot to {plot_path}")
    try:
        # matplotlib 설정 수정
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 이미지 생성
        import matplotlib.pyplot as plt
        
        # Backtrader의 플로팅 시도
        try:
            figs = cerebro.plot(style='candlestick', volume=False, iplot=False)
            if figs and len(figs) > 0 and len(figs[0]) > 0:
                fig = figs[0][0]
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logging.info(f"Successfully saved plot to {plot_path}")
            else:
                raise ValueError("Plotting produced no figures")
        except (AttributeError, ValueError) as e:
            logging.warning(f"Backtrader plotting failed: {str(e)}")
            logging.info("Attempting to create simple portfolio value plot instead...")
            
            # 간단한 포트폴리오 가치 플롯 생성
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, results[0].observers.broker.lines.value)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Successfully saved simplified plot to {plot_path}")
    
    except Exception as e:
        logging.error(f"Error while plotting: {str(e)}")
        logging.error("Unable to save plot, but backtest results are still valid")

if __name__ == '__main__':
    run_backtest()
