# Project ART (자동화된 거래) - 개발 진행 상황

## 프로젝트 개요
Project ART는 암호화폐 자동 거래 시스템을 개발하는 프로젝트입니다. 주요 기능으로는:
- 실시간 암호화폐 가격 데이터 수집
- 기술적 분석 및 삼중장벽 전략 기반 신호 생성
- 리스크 관리 및 포트폴리오 최적화
- 실시간 거래 실행 및 모니터링
- 성과 분석 및 백테스팅

## PRD (제품 요구사항)
### 1. 시스템 요구사항
- Python 3.8 이상
- 주요 의존성 패키지:
  - python-binance: 암호화폐 데이터 수집
  - influxdb-client: 시계열 데이터 저장
  - python-telegram-bot: 실시간 알림
  - backtrader: 백테스팅
  - numpy, pandas: 데이터 처리

### 2. 핵심 기능 요구사항
1. **데이터 수집 및 처리**
   - Binance API를 통한 실시간/과거 데이터 수집
   - InfluxDB를 통한 데이터 저장 및 관리
   - 데이터 정합성 검증 및 에러 처리

2. **거래 전략**
   - 삼중장벽 전략 구현
   - 기술적 지표 계산 (MA, RSI, MACD 등)
   - 트렌드 감지 및 신호 생성

3. **리스크 관리**
   - 포지션 크기 최적화
   - 손절/익절 레벨 설정
   - 포트폴리오 리밸런싱

4. **알림 시스템**
   - 텔레그램을 통한 실시간 알림
   - 거래 신호 알림
   - 에러 발생 알림
   - 성과 보고서 알림

5. **백테스팅**
   - 과거 데이터 기반 백테스팅
   - 성과 지표 계산
   - 전략 최적화

### 3. 보안 요구사항
- API 키 및 비밀번호는 환경 변수나 별도의 credentials 파일로 관리
- 모든 API 호출에 에러 처리 및 재시도 로직 구현
- 로깅 및 모니터링 시스템 구현

## 프로젝트 진행 체크리스트
### 1. 환경 설정
- [x] Python 환경 설정
- [x] 필요한 패키지 설치
- [x] 환경 변수 설정
- [x] credentials 파일 설정

### 2. 데이터 파이프라인
- [x] Binance 데이터 수집기 구현
- [x] InfluxDB 데이터 저장 구현
- [x] 데이터 검증 및 에러 처리
- [x] InfluxDB 설치 및 실행 확인

### 3. 알림 시스템
- [x] 텔레그램 알림 구현
- [x] 비동기 API 적용
- [x] 에러 처리 및 재시도 로직

### 4. 통합 테스트 (현재 진행 중)
- [ ] 데이터 수집 → 저장 → 알림 흐름 테스트
- [ ] 에러 시나리오 테스트
- [ ] 성능 테스트
- [ ] 스케줄링 테스트

## 작업 목록
- [x] 프로젝트 폴더 구조 및 __init__.py 패키지화
- [x] requirements.txt, .gitignore, README.md, config/credentials_example.py 생성
- [x] config/credentials.py 파일에 실제 API 키/토큰 입력 (USER 직접 수행)
- [x] features/build_features.py: 기술적 지표 및 삼중장벽 레이블링 함수 구현
- [x] risk_management/manager.py: RiskManager 클래스 구현
- [x] backtester/run_backtest.py: backtrader 기반 백테스팅 템플릿 구현
- [x] data_pipeline/collector.py: BinanceDataCollector (과거/실시간 수집) 구현
- [x] notifications/telegram_bot.py: TelegramNotifier 구현
- [x] data_pipeline/collector.py의 fetch_historical_data로 바이낸스 과거 데이터 수집 테스트
- [x] 수집 데이터 InfluxDB 저장 및 알림 연동 테스트
- [x] InfluxDB 브라우저(UI) 접속 문제 해결
- [ ] 전체 데이터 파이프라인/알림/백테스팅 통합 테스트

## 현재 목표
- **백테스팅 시스템 구축 시작**: 안정화된 데이터 파이프라인을 기반으로, 백테스팅에 필요한 데이터를 InfluxDB에서 불러오는 기능을 구현합니다.
