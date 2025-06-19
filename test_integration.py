# -*- coding: utf-8 -*-
print("--- EXECUTING LATEST test_integration.py ---")
"""
Project ART 통합 테스트 스크립트

이 스크립트는 다음과 같은 흐름을 테스트합니다:
1. Binance 데이터 수집
2. InfluxDB 저장
3. 텔레그램 알림
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.collector import BinanceDataCollector
from notifications.telegram_bot import TelegramNotifier
from config.credentials import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    INFLUXDB_TOKEN,
    INFLUXDB_ORG,
    INFLUXDB_BUCKET,
    INFLUXDB_URL
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """통합 테스트 메인 함수"""
    notifier = None  # notifier를 미리 None으로 초기화
    try:
        # 1. 데이터 수집기 초기화 (생성자는 인수를 받지 않음)
        collector = BinanceDataCollector()
        logger.info("데이터 수집기 초기화 완료")

        # 2. 알림 시스템 초기화
        notifier = TelegramNotifier(
            token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID
        )
        logger.info("알림 시스템 초기화 완료")

        # 3. 데이터 수집 테스트
        logger.info("데이터 수집 테스트 시작")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)  # 24시간 데이터 수집
        
        start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
        end_str = end_time.strftime("%d %b, %Y %H:%M:%S")

        df = collector.fetch_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_str=start_str,
            end_str=end_str
        )
        logger.info(f"데이터 수집 완료: {len(df)}개의 레코드")

        # 4. 데이터 저장 테스트
        logger.info("데이터 저장 테스트 시작")
        success = collector.save_to_influxdb(df, "crypto_prices_hourly")
        if success:
            logger.info("데이터 저장 성공")
            await notifier.send_message(
                "데이터 수집 및 저장 테스트 성공!\n"
                f"- 기간: {start_time} ~ {end_time}\n"
                f"- 레코드 수: {len(df)}개"
            )
        else:
            logger.error("데이터 저장 실패")
            await notifier.send_message(
                "데이터 저장 실패!\n"
                "에러가 발생했습니다. 로그를 확인해주세요."
            )

    except Exception as e:
        logger.error(f"통합 테스트 중 예외 발생: {e}")
        if notifier:  # notifier가 초기화되었는지 확인
            await notifier.send_message(
                f"통합 테스트 중 에러 발생!\n"
                f"에러 내용: {str(e)}"
            )
        else:
            logger.error("Notifier가 초기화되지 않아 에러 메시지를 텔레그램으로 보낼 수 없습니다.")

if __name__ == '__main__':
    asyncio.run(main())
