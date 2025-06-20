# Project ART: AI 기반 암호화폐 자동매매 시스템 개발 계획

## 개요
이 문서는 AI 기반 암호화폐 자동매매 시스템 "ART" 프로젝트의 개발 계획 및 진행 상황을 추적합니다.

**범례:**
- `[x]` : 완료된 작업
- `[~]` : 진행 중인 작업
- `[ ]` : 시작 전 작업

## 주요 참고 사항 (Notes)
- 프로젝트는 AI 기반 암호화폐 자동매매 시스템 구축을 목표로 하며, 이벤트 기반 아키텍처, ML/RL 모델, 다층적 리스크 관리가 핵심입니다.
- 현재 **Phase 1 (데이터/피처 파이프라인 MVP)** 가 진행 중입니다.
- **주요 완료 사항**: 다중 타임프레임 데이터 수집, InfluxDB 연동, 실시간 수집기, Python 3.11 환경 구성, `collector.py` 오류 수정, `build_features.py` 내 look-ahead bias 점검, `labeling.py` 실행 오류 및 저장 문제 해결.
- **`pandas-ta` 관련 이슈**: `TA-Lib` 설치 문제로 `pandas-ta` 사용 중. `numpy.NaN` 관련 `ImportError` 발생 시 `pandas-ta` 내부 코드를 직접 수정하여 해결한 경험이 있습니다 (`numpy` 1.25.2 버전과 `pandas-ta` 0.3.14b0).
- 이 `plan.md`의 "Current Goal" 섹션은 현재 집중해야 할 가장 중요한 작업을 명시합니다.

---

## 🎯 현재 목표 (Current Goal)
**Phase 1: [ ] 대체 데이터(펀딩비, 온체인 등) 수집기 구현**

---

## 세부 작업 목록 (Task List)

### Phase 0: 프로젝트 구조 및 문서화
- [x] README.md 최신화 (PRD 기반)
- [x] plan.md (본 개발 계획 문서) 최신화 및 체크리스트 작성
- [x] `scripts` 폴더 구조 정리 및 스크립트 이동
- [x] 초기 Git 커밋 및 브랜치 전략 논의

### Phase 1: 데이터/피처 파이프라인 MVP
- **데이터 수집 및 저장:**
  - [x] 다중 타임프레임(1h/4h/1d) 과거 데이터 수집 (바이낸스 API)
  - [x] InfluxDB 설치 및 원격 접속 설정
  - [x] 실시간 데이터 수집기 (`collector.py` - 바이낸스 웹소켓) 구현 및 DB 연동
  - [x] `.env` 파일 생성 및 `gitignore`에 추가 (API 키, DB 정보 등)
  - [x] `requirements.txt`에 `python-dotenv` 추가
  - [x] InfluxDB 연결부 `dotenv` 적용 및 환경 변수 기반 연결 정상 동작 확인
  - [x] `collector.py` 스크립트 로직 수정 및 오류 해결 (데이터 수집 코드 활성화, 쿼리 문제 등)
- **피처 엔지니어링 및 레이블링:**
  - [~] 피처 엔지니어링 스크립트 (`features/build_features.py`) 개선 (다중 타임프레임 통합/정렬, 기술적 지표 추가)
  - [x] `build_features.py` 내 look-ahead bias 검토 및 수정 (`pd.merge_asof` 적용)
  - [~] 삼중 장벽 레이블링 함수 (`get_triple_barrier_labels`) 검토 (현재 `build_features.py`에 통합, 필요시 분리 또는 `features/labeling.py`로 기능 이전 검토)
  - [x] `features/labeling.py` 스크립트 실행 및 디버깅 완료
    - [x] `pandas-ta` `ImportError` (numpy.NaN vs numpy.nan) 문제 해결 (라이브러리 코드 직접 수정)
    - [x] 파일 저장 오류 (`WinError 5` 등 권한/경로/파일점유 문제) 디버깅 및 해결
    - [x] 실제 피처 데이터(`btcusdt_feature_matrix.parquet`) 기반 이벤트 생성, 레이블링, 결과 저장 (`labeled_btcusdt_data.parquet`) 정상 동작 확인
- **신규 데이터 파이프라인 구축:**
  - **[ ] 대체 데이터(펀딩비, 온체인 등) 수집기 구현**  <-- **여기에 집중!**
- **모델링 준비:**
  - [ ] 예측 모델 (v1: XGBoost/LightGBM) 훈련 및 저장, 워크 포워드 검증
- **환경 및 기타:**
  - [x] Python 3.11 설치 및 가상환경(`.venv`) 재구성
  - [x] 필수 라이브러리 재설치 및 환경 정상 동작 검증

### Phase 2: 자동화/실거래 테스트
- [ ] 리스크 관리 모듈 및 단위 테스트 작성
- [ ] 백테스팅 엔진 템플릿 구축 및 v1 모델 성과 평가
- [ ] 알림 시스템 (Telegram 등) 연동 및 테스트
- [ ] 규칙 기반 실행 엔진 및 페이퍼 트레이딩

### Phase 3: 강화학습 및 동적 최적화
- [ ] RL 에이전트 학습 환경 구축
- [ ] RL 모델 훈련 및 통합

---
*이 문서는 지속적으로 업데이트됩니다.*
