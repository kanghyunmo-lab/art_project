# test_telegram_simple.py
import telegram
import asyncio # asyncio 라이브러리 임포트

BOT_TOKEN = "8159716567:AAExhzT5_Coj1Wxf1apdh-WAH63m8FuTuoc"  # 실제 토큰으로 변경
CHAT_ID = "5965442851"     # 실제 챗 ID로 변경 (문자열로!)

async def send_test_message(): # 비동기 함수로 정의
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        # bot.send_message는 코루틴이므로 await 사용
        await bot.send_message(chat_id=CHAT_ID, text="Async Simple Python script test message")
        print("Async Simple test message sent successfully via Python.")
    except Exception as e:
        print(f"Error in async simple Python test: {e}")

if __name__ == '__main__':
    # 비동기 함수 실행
    asyncio.run(send_test_message())