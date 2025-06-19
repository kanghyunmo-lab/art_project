# Project ART: AI 기반 암호화폐 자동매매 시스템 개발 계획

## Notes
- 24시간 무중단, AI 기반 암호화폐 자동매매 시스템 구축 목표
- 머신러닝 예측 모델(1차: XGBoost/LightGBM, 2차: BiLSTM-Transformer), 강화학습(DRL) 에이전트, 다층적 리스크 관리, 이벤트 기반 아키텍처 적용
- 투자 전략: 레버리지 활용, 양방향(롱/숏) 거래
- PRD 및 개발 체크리스트의 상세 요구사항, 성공 지표, 단계별 과업 반영
- 데이터 파이프라인, 피처 엔지니어링, 모델 훈련/저장, 리스크 관리, 백테스팅, 알림, 자동화, DRL 등 단계별 완료 기준 명확화
- 스윙 트레이딩에 적합한 다중 타임프레임(1h, 4h, 1d) 분석 전략 도입, 과도한 프레임 추가는 피처 노이즈 및 복잡성 증가 위험
- 환경 변수(.env) 기반 InfluxDB 연결 구조로 개선, gitignore/requirements.txt/dotenv 적용
- 최근 인증 오류(401 Unauthorized)는 .env 내 InfluxDB 토큰/조직 정보 미일치로 확인, 사용자 직접 수정 필요
- 디버깅용 로깅 코드는 제거하여 코드 원상 복구
- 파이썬 환경/라이브러리 불안정성 발견 시, 진단용 임포트 테스트 및 pip 캐시/재설치로 해결 가능함을 확인
- 파이썬 3.13.x와 influxdb-client 등 주요 라이브러리 간 호환성 문제 발견, 안정적인 개발을 위해 Python 3.11 기반 가상환경(venv) 구성이 필수적임을 확인
- Python 3.11 설치 완료, 이후 단계부터 새 버전 기반 환경 적용 예정
- 필수 라이브러리 재설치 및 환경 정상 동작 완료, 이제 인증 정보(.env)만 수정하면 InfluxDB 연결이 정상화됨을 확인
- 주석 내 숨겨진 문자/인코딩 문제로 인해 SyntaxError가 발생할 수 있음을 확인, 해당 이슈 해결 경험 추가

## Task List
### Phase 0: 프로젝트 구조 및 문서화
- [x] README.md 최신화(PRD 기반)
- [x] plan.md(개발 계획) 최신화 및 체크리스트 작성
- [x] scripts 폴더 구조 정리 및 스크립트 이동
- [x] Git 커밋

### Phase 1: 데이터/피처 파이프라인 MVP
- [x] 다중 타임프레임(1h/4h/1d) 과거 데이터 수집
- [x] InfluxDB 설치 및 원격 접속 설정
- [x] 실시간 데이터 수집기(바이낸스 웹소켓) 구현 및 DB 연동
- [ ] 대체 데이터(펀딩비, 온체인 등) 수집기 구현
- [x] .env 파일 생성 및 gitignore에 추가
- [x] requirements.txt에 python-dotenv 추가
- [x] InfluxDB 연결부 dotenv 적용
- [x] 환경 변수 기반 연결 정상 동작 확인(침묵의 실패 재현 방지)
- [x] 실시간/과거 데이터 수집 및 InfluxDB 적재
- [x] collector.py 스크립트 수정(데이터 수집 코드 활성화)
- [x] collector.py의 query_data_from_influxdb 메소드의 SyntaxError/들여쓰기 문제 해결
- [ ] 피처 엔지니어링 스크립트(features/build_features.py) 작성(다중 타임프레임 통합/정렬)
- [ ] 삼중 장벽 레이블링 함수(get_triple_barrier_labels) 구현
- [ ] 예측 모델(v1: XGBoost/LightGBM) 훈련 및 저장, 워크 포워드 검증
- [ ] 리스크 관리 모듈 및 단위 테스트 작성
- [ ] 백테스팅 엔진 템플릿 구축 및 v1 모델 성과 평가
- [ ] 알림 시스템(Telegram) 연동 및 테스트
- [ ] 파이썬 환경 불안정성 진단 및 라이브러리 재설치 (pip uninstall, cache purge, 재설치)
- [x] Python 3.11 설치
- [x] Python 3.11 기반 가상환경(venv) 생성 및 프로젝트 환경 재구성
- [x] 필수 라이브러리 재설치 및 환경 정상 동작 검증

### Phase 2: 자동화/실거래 테스트
- [ ] 규칙 기반 실행 엔진 및 페이퍼 트레이딩
- [ ] 거래소 API 핸들러 연동(테스트넷)
- [ ] 예측 모델(v2: BiLSTM-Transformer) 구현 및 성능 비교

### Phase 3: 강화학습 및 동적 최적화
- [ ] DRL 시뮬레이션 환경 구축 및 PPO 에이전트 통합

## Current Goal
피처 엔지니어링 관련 논의
