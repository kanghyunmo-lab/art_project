# Collectors 모듈

이 디렉터리에는 외부 API 또는 다른 소스로부터 다양한 금융 데이터를 수집하여 데이터베이스(예: InfluxDB)에 저장하는 스크립트들이 포함되어 있습니다.

## `collect_funding_rate.py`

**설명:**

특정 암호화폐 선물 시장의 펀딩 비율(Funding Rate) 데이터를 바이낸스 API를 통해 수집하여 InfluxDB에 저장하는 스크립트입니다.

**필요 설정:**

스크립트를 실행하기 전에 다음 설정이 필요합니다:

1.  **`.env` 파일:**
    *   `BINANCE_API_KEY`: 바이낸스 API 키
    *   `BINANCE_API_SECRET`: 바이낸스 API 시크릿 키
    *   `INFLUXDB_URL`: InfluxDB 접속 URL (예: `http://localhost:8086`)
    *   `INFLUXDB_TOKEN`: InfluxDB 접속 토큰
    *   `INFLUXDB_ORG`: InfluxDB 조직(Organization)
    *   `INFLUXDB_BUCKET_FUNDING_RATE`: 펀딩비 데이터를 저장할 버킷 이름 (예: `funding_rates`)

2.  **`config.py` 파일 (해당하는 경우):**
    *   스크립트 동작에 필요한 추가적인 설정이 있다면 `config.py` 파일 내에 정의될 수 있습니다. (현재 `collect_funding_rate.py`는 주로 `.env` 파일과 커맨드라인 인자를 사용합니다.)

**사용법:**

스크립트는 커맨드라인 인터페이스(CLI)를 통해 실행되며, 수집 모드(`recent` 또는 `historical`)와 대상 심볼(`--symbol`)을 지정해야 합니다.

*   **최근 펀딩비 데이터 수집 (기본값: 최근 1000개):**

    ```bash
    python collectors/collect_funding_rate.py --mode recent --symbol BTCUSDT
    ```

*   **과거 특정 기간 펀딩비 데이터 수집:**

    `--start` 와 `--end` 옵션을 사용하여 기간을 지정합니다. 날짜 형식은 `YYYY-MM-DD` 입니다.

    ```bash
    python collectors/collect_funding_rate.py --mode historical --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01
    ```

    특정 시작 시간부터 가장 최근까지 수집하려면 `--end`를 생략할 수 있습니다.

    ```bash
    python collectors/collect_funding_rate.py --mode historical --symbol BTCUSDT --start 2023-01-01
    ```

**데이터 저장:**

*   수집된 펀딩비 데이터는 InfluxDB의 지정된 버킷에 저장됩니다.
*   Measurement 이름: `funding_rate` (기본값, 스크립트 내 변수로 지정)
*   Tags:
    *   `symbol`: 암호화폐 심볼 (예: `BTCUSDT`)
*   Fields:
    *   `fundingRate`: 펀딩 비율 값 (float)
    *   `fundingTime`: 펀딩 시간의 Unix timestamp (integer, milliseconds)
*   Timestamp: InfluxDB에 기록되는 시간 (일반적으로 펀딩 시간과 동일하게 설정됨)
