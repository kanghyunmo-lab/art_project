import datetime

class RiskManager:
    """
    시스템의 중앙 통제 장치로서, 모든 주문을 최종 심사하여 리스크를 관리합니다.
    """
    def __init__(self, config, portfolio):
        """
        RiskManager를 초기화합니다.

        :param config: (dict) 리스크 관리 설정값 (예: {'max_risk_per_trade': 0.01, 'max_mdd': 0.20})
        :param portfolio: (Portfolio) 현재 포트폴리오 상태 객체
        """
        self.config = config
        self.portfolio = portfolio
        self.last_error_time = None
        self.error_count = 0

    def check_order_risk(self, order_event):
        """
        [2계층] 개별 주문의 리스크를 검사합니다.
        
        :param order_event: (OrderEvent) 검사할 주문 이벤트
        :return: (bool) 주문이 리스크 한도를 통과하면 True, 아니면 False
        """
        max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01) # 기본값 1%
        trade_risk = order_event.quantity * order_event.stop_loss_price
        portfolio_value = self.portfolio.get_total_value()

        if trade_risk > portfolio_value * max_risk_per_trade:
            print(f"[RISK] Order REJECTED: Trade risk ({trade_risk}) exceeds max risk per trade.")
            return False
        
        print("[RISK] Order risk check PASSED.")
        return True

    def check_portfolio_mdd(self):
        """
        [2계층] 포트폴리오의 최대 낙폭(MDD)을 검사합니다.

        :return: (bool) MDD가 한도 이내이면 True, 아니면 False
        """
        max_mdd = self.config.get('max_mdd', 0.20) # 기본값 20%
        current_mdd = self.portfolio.get_current_mdd()

        if current_mdd > max_mdd:
            print(f"[RISK] MDD Alert: Current MDD ({current_mdd:.2%}) exceeds limit ({max_mdd:.2%}). Halting trades.")
            return False
        
        print("[RISK] Portfolio MDD check PASSED.")
        return True

    def check_system_health(self, connectivity_ok=True, api_error=False):
        """
        [3계층] 시스템의 전반적인 상태를 검사합니다.
        
        :param connectivity_ok: (bool) 거래소와의 연결 상태
        :param api_error: (bool) 최근 API 호출에서 오류가 발생했는지 여부
        :return: (bool) 시스템이 건강하면 True, 아니면 False
        """
        if not connectivity_ok:
            print("[RISK] System HALTED: Connectivity issue detected.")
            return False

        # API 오류율 한도 검사 (예: 1분 내 5회 이상 오류 시 중단)
        if api_error:
            now = datetime.datetime.now()
            if self.last_error_time and (now - self.last_error_time) < datetime.timedelta(minutes=1):
                self.error_count += 1
            else:
                self.error_count = 1
            self.last_error_time = now

            if self.error_count > self.config.get('max_api_errors', 5):
                print("[RISK] System HALTED: High API error rate detected.")
                return False

        print("[RISK] System health check PASSED.")
        return True

    def check_volatility_circuit_breaker(self, market_volatility):
        """
        [3계층] 시장 변동성 서킷 브레이커를 검사합니다.
        
        :param market_volatility: (float) 현재 시장 변동성 지표 (예: VIX, ATR 등)
        :return: (bool) 변동성이 허용 범위 내이면 True, 아니면 False
        """
        volatility_limit = self.config.get('volatility_limit', 3.0) # 예: 평소 변동성의 3배
        if market_volatility > volatility_limit:
            print(f"[RISK] Circuit Breaker TRIGGERED: Extreme market volatility ({market_volatility}) detected. Halting trades.")
            return False
        
        print("[RISK] Volatility circuit breaker check PASSED.")
        return True


if __name__ == '__main__':
    # --- 예제 사용법 ---
    # 1. 가짜 포트폴리오 및 주문 객체 생성
    class MockPortfolio:
        def get_total_value(self): return 100000  # 총 자산 10만 달러
        def get_current_mdd(self): return 0.15 # 현재 MDD 15%

    class MockOrderEvent:
        def __init__(self, quantity, stop_loss_price):
            self.quantity = quantity
            self.stop_loss_price = stop_loss_price

    # 2. 리스크 매니저 설정 및 초기화
    risk_config = {
        'max_risk_per_trade': 0.01, # 거래당 최대 리스크 1%
        'max_mdd': 0.20,            # 최대 낙폭 20%
        'max_api_errors': 5,        # 1분 내 최대 API 오류 5회
        'volatility_limit': 3.0     # 변동성 한계
    }
    portfolio = MockPortfolio()
    risk_manager = RiskManager(risk_config, portfolio)

    # 3. 리스크 검사 시뮬레이션
    print("--- Testing Safe Order ---")
    safe_order = MockOrderEvent(quantity=10, stop_loss_price=50) # 리스크: 500 (자산의 0.5%)
    risk_manager.check_order_risk(safe_order)

    print("\n--- Testing Risky Order ---")
    risky_order = MockOrderEvent(quantity=30, stop_loss_price=50) # 리스크: 1500 (자산의 1.5%)
    risk_manager.check_order_risk(risky_order)

    print("\n--- Testing Other Checks ---")
    risk_manager.check_portfolio_mdd()
    risk_manager.check_system_health(api_error=False)
    risk_manager.check_volatility_circuit_breaker(market_volatility=2.5)

    print("\n--- Testing Circuit Breaker ---")
    risk_manager.check_volatility_circuit_breaker(market_volatility=3.5)
