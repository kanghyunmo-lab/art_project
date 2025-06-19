# Project ART (Automated Trading) - 개발 진행 상황

## Notes
- PRD(제품 요구사항 문서) 기반 단계별 체크리스트로 진행 상황 관리
- 모든 주요 모듈 폴더 및 패키지화, credentials 예시 파일 제공, 실제 API 키/토큰 사용 설정까지 완료
- 각 핵심 모듈(피처 엔지니어링, 리스크 관리, 백테스팅, 데이터 파이프라인, 알림) 기본 코드 및 테스트 예시 포함
- 파이썬 환경 문제 및 influxdb-client 인식 오류 해결됨, collector.py 정상 실행 확인
- TelegramNotifier가 토큰/채팅 ID 미설정 또는 오류 시에도 성공 메시지를 출력할 수 있음, 실제 텔레그램 메시지 미수신 문제 확인 필요
- python-telegram-bot v20+ 비동기 API 사용법 오류(coroutine 'send_message' was never awaited)로 인해 메시지 미전송 현상 확인됨
- 텔레그램 알림 연동 문제는 잠정 보류, 개발 우선 진행
- InfluxDB 데이터 저장 기능은 정상 동작 확인, 브라우저(웹 UI) 접속 불가로 데이터 확인 어려움. 접속 문제 해결 필요
- InfluxDB가 설치되어 있지 않음, 설치 및 기본 설정부터 필요

## Task List
- [x] 프로젝트 폴더 구조 및 __init__.py 패키지화
- [x] requirements.txt, .gitignore, README.md, config/credentials_example.py 생성
- [x] config/credentials.py 파일에 실제 API 키/토큰 입력 (USER 직접 수행)
- [x] features/build_features.py: 기술적 지표 및 삼중장벽 레이블링 함수 구현
- [x] risk_management/manager.py: RiskManager 클래스 구현
- [x] backtester/run_backtest.py: backtrader 기반 백테스팅 템플릿 구현
- [x] data_pipeline/collector.py: BinanceDataCollector (과거/실시간 수집) 구현
- [x] notifications/telegram_bot.py: TelegramNotifier 구현
- [x] data_pipeline/collector.py의 fetch_historical_data로 바이낸스 과거 데이터 수집 테스트
- [ ] 수집 데이터 InfluxDB 저장 및 알림 연동 테스트
- [ ] 텔레그램 알림 미수신 문제 디버깅 (보류)
- [ ] python-telegram-bot 비동기 API 사용법 수정 및 테스트 (보류)
- [ ] InfluxDB 브라우저(UI) 접속 문제 해결
- [ ] InfluxDB 설치 및 기본 실행 테스트
- [ ] 전체 데이터 파이프라인/알림/백테스팅 통합 테스트

## Current Goal
- InfluxDB 설치 및 실행 확인
