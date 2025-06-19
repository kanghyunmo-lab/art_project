# test_telegram_async.py
import telegram
import asyncio # asyncio 라이브러리 임포트

# --- 중요! --- 
# 아래 BOT_TOKEN과 CHAT_ID를 실제 값으로 변경해주세요.
# CHAT_ID는 반드시 문자열 형태로 따옴표로 감싸주세요. (예: "123456789" 또는 "@channelname")
BOT_TOKEN = "YOUR_BOT_TOKEN"  # 실제 토큰으로 변경
CHAT_ID = "YOUR_CHAT_ID"      # 실제 챗 ID로 변경 (문자열로!)

async def send_test_message(): # 비동기 함수로 정의
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        # bot.send_message는 코루틴이므로 await 사용
        await bot.send_message(chat_id=CHAT_ID, text="Async Simple Python script test message - 확인용")
        print("Async Simple test message sent successfully via Python.")
    except telegram.error.TelegramError as e:
        print(f"Telegram API Error in async simple Python test: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.content}")
    except Exception as e:
        print(f"General Error in async simple Python test: {e}")

if __name__ == '__main__':
    # 비동기 함수 실행
    print(f"Attempting to send a message to CHAT_ID: {CHAT_ID} using BOT_TOKEN: {BOT_TOKEN[:10]}...")
    asyncio.run(send_test_message())
