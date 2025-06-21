# Project ART: AI 기반 암호화폐 자동매매 시스템

## 1. 프로젝트 비전 (Vision)

**Project ART(Algorithmic Risk-managed Trading)**는 인간의 감정적 편향과 물리적 한계를 극복하고, 데이터에 기반한 일관된 원칙으로 24시간 암호화폐 시장에 대응하는 완전 자동화된 '퀀트 펀드' 시스템 구축을 목표로 합니다.

이 시스템은 머신러닝 예측 모델과 강화학습 에이전트를 통해 시장 상황에 동적으로 대응하며, 레버리지를 활용하여 수익을 극대화합니다. 모든 거래는 다층적 리스크 관리 시스템의 엄격한 통제 하에 안전하게 실행됩니다.

## 2. 핵심 아키텍처 (Core Architecture)

본 시스템은 실시간 데이터 처리에 최적화된 **이벤트 기반 아키텍처(Event-Driven Architecture)**를 채택하여 모듈의 독립성과 확장성을 확보합니다.

- **데이터 피드 핸들러:** 시장 데이터 수집 및 `MarketEvent` 생성
- **신호 및 의사결정 엔진:** ML/RL 모델 기반 `SignalEvent` 및 `OrderEvent` 생성
- **리스크 관리 모듈:** 포트폴리오 규칙 기반 주문 최종 심사
- **실행 핸들러:** 실제 거래소에 주문 제출 및 `FillEvent` 생성
- **포트폴리오 관리자:** `FillEvent` 기반 실시간 포지션 및 성과 추적

## 3. 개발 로드맵 (Roadmap)

- **Phase 1 (기반 구축):** 예측 모델(v1), 리스크 관리 프레임워크, 백테스팅 엔진 완성.
- **Phase 2 (자동화 전환):** 모델 고도화(v2) 및 규칙 기반 자동매매 시스템 통합.
- **Phase 3 (동적 최적화):** DRL 에이전트를 통한 동적 레버리지 및 포지션 관리 구현.

## 4. 프로젝트 구조 (Project Structure)

```
├── data_pipeline/        # 실시간·과거 시장 데이터 수집
│   └── collector.py      # InfluxDB/Binance 등에서 OHLCV 수집
├── features/             # 피처 엔지니어링
│   └── build_features.py # 기술 지표, 펀딩레이트 등 생성
├── models/               # ML 모델 및 학습 코드
│   ├── base.py           # BaseLogger / BaseModel / BaseTrainer 인터페이스
│   ├── data_loader.py    # 데이터 로딩·전처리(DataFrame→Features)
│   ├── training.py       # XGBoostModel 등 모델 학습 로직
│   ├── evaluation.py     # ModelEvaluator – 성능 측정
│   └── train_model_new.py# 메인 학습 스크립트(CLI)
├── backtester/           # 백테스트 엔진 (walk-forward, 전략 검증)
├── risk_management/      # 다층 리스크 관리 규칙 및 포지션 sizing
├── execution/            # 거래소 API 인터페이스 및 주문 실행
├── tests/                # 테스트 스위트
│   ├── unit_test_*.py    # 각 모듈 단위 테스트
│   └── integration_test_pipeline.py # 전체 파이프라인 통합 테스트
├── config/               # 경로·API 키 등 설정
├── scripts/              # 데이터 수집·유틸리티 스크립트
└── requirements.txt      # Python 의존성 목록
```

### 전체 파이프라인 흐름
```
Raw Data
   ↓
data_pipeline/collector.py
   ↓
features/build_features.py (라벨/피처 생성)
   ↓
models/data_loader.py (전처리·스플릿)
   ↓
models/training.py (모델 학습)
   ↓
models/evaluation.py (성능 평가)
   ↓
backtester/ (전략 검증)
   ↓
execution/ (실거래 주문)
```

각 디렉터리 및 주요 파일의 용도를 한눈에 볼 수 있도록 정리했습니다. 이 구조를 기준으로 개발 · 테스트 · 배포가 이루어집니다.

## 5. 설치 및 설정 (Setup)

1.  **가상환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **의존성 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API 키 및 설정:**
    - `config/credentials_example.py` 파일을 `config/credentials.py`로 복사합니다.
    - `config/credentials.py` 파일에 실제 Binance API 키, Telegram 봇 토큰 등 필요한 정보를 입력합니다.

## 6. 사용법 (Usage)

> (추후 각 모듈 실행 방법에 대한 안내가 추가될 예정입니다.)

- **과거 데이터 수집:**
  ```bash
  python scripts/data_collector.py
  ```
- **모델 훈련:**
  ```bash
  python models/train_model.py
  ```
- **백테스트 실행:**
  ```bash
  python backtester/run_backtest.py
  ```
