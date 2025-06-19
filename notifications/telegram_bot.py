import os
import asyncio
from telegram import Bot
from telegram.error import TelegramError

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# API 키는 config 파일에서 로드 (존재하지 않을 경우를 대비한 예외 처리 포함)
try:
    from config.credentials import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    print("Warning: credentials.py not found. Using dummy credentials. Please create config/credentials.py")
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"  # Can be a numeric ID or a channel name like '@channelname'

class TelegramNotifier:
    """
    비동기 방식으로 텔레그램 봇을 통해 메시지와 이미지를 전송합니다.
    python-telegram-bot v20+의 비동기 API를 사용합니다.
    """
    def __init__(self, token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
        """
        텔레그램 봇을 초기화합니다.

        :param token: (str) 텔레그램 봇 토큰
        :param chat_id: (str or int) 메시지를 보낼 채팅 ID 또는 채널 이름
        """
        if token == "YOUR_TELEGRAM_BOT_TOKEN" or not token:
            print("Telegram token is not configured. Notifier will be disabled.")
            self.bot = None
            self.chat_id = None  # 명시적으로 None 설정
        elif chat_id == "YOUR_TELEGRAM_CHAT_ID" or not chat_id:
            self.bot = Bot(token=token)  # 토큰은 유효할 수 있으나 chat_id가 문제
            self.chat_id = None
            print("Telegram chat_id is not configured or is a placeholder. Notifier will be disabled for sending.")
        else:
            self.bot = Bot(token=token)
            self.chat_id = chat_id
            print("Telegram notifier initialized successfully.")

    def _is_initialized(self):
        """초기화 상태를 확인하는 내부 동기 함수"""
        if not self.bot or not self.chat_id:
            print("Telegram notifier is not properly initialized. Check your token and chat_id.")
            return False
        return True

    def send_message(self, text):
        """
        지정된 채팅 ID로 텍스트 메시지를 보냅니다. (동기 래퍼)
        
        :param text: (str) 보낼 메시지 내용
        :return: (bool) 성공 여부
        """
        if not self._is_initialized():
            print(f"Telegram disabled. Message not sent: {text}")
            return False

        async def _send():
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text)
                print("Telegram message sent successfully.")
                return True
            except TelegramError as e:
                print(f"Error sending Telegram message: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error in send_message: {e}")
                return False

        try:
            return asyncio.run(_send())
        except RuntimeError as e:
            print(f"Could not run asyncio task (maybe an event loop is already running?): {e}")
            return False

    def send_photo(self, photo_path, caption=""):
        """
        지정된 채팅 ID로 사진을 보냅니다. (동기 래퍼)

        :param photo_path: (str) 보낼 사진 파일의 경로
        :param caption: (str, optional) 사진과 함께 보낼 캡션
        :return: (bool) 성공 여부
        """
        if not self._is_initialized():
            print(f"Telegram disabled. Photo not sent: {photo_path}")
            return False

        if not os.path.exists(photo_path):
            print(f"Error: Photo file not found at {photo_path}")
            return False

        async def _send():
            try:
                with open(photo_path, 'rb') as photo_file:
                    await self.bot.send_photo(chat_id=self.chat_id, photo=photo_file, caption=caption)
                print(f"Telegram photo sent successfully: {photo_path}")
                return True
            except TelegramError as e:
                print(f"Error sending Telegram photo: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error in send_photo: {e}")
                return False

        try:
            return asyncio.run(_send())
        except RuntimeError as e:
            print(f"Could not run asyncio task (maybe an event loop is already running?): {e}")
            return False
            
    # 동기식 호출을 위한 래퍼 메서드 (기존 코드와의 호환성을 위해 유지)
    def send_message_sync(self, text):
        """
        동기식으로 메시지를 보내는 래퍼 메서드입니다.
        내부적으로 asyncio를 사용하여 비동기 함수를 실행합니다.
        
        :param text: (str) 보낼 메시지 내용
        :return: (bool) 성공 여부
        """
        try:
            # 현재 실행 중인 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 이벤트 루프가 있으면 해당 루프에서 실행
                return loop.run_until_complete(self.send_message(text))
            except RuntimeError:  # No running event loop
                # 실행 중인 이벤트 루프가 없으면 새로 생성
                return asyncio.run(self.send_message(text))
        except Exception as e:
            print(f"Error in send_message_sync: {e}")
            return False
            
    def send_photo_sync(self, photo_path, caption=""):
        """
        동기식으로 사진을 보내는 래퍼 메서드입니다.
        내부적으로 asyncio를 사용하여 비동기 함수를 실행합니다.
        
        :param photo_path: (str) 보낼 사진 파일의 경로
        :param caption: (str, optional) 사진과 함께 보낼 캡션
        :return: (bool) 성공 여부
        """
        try:
            # 현재 실행 중인 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 이벤트 루프가 있으면 해당 루프에서 실행
                return loop.run_until_complete(self.send_photo(photo_path, caption))
            except RuntimeError:  # No running event loop
                # 실행 중인 이벤트 루프가 없으면 새로 생성
                return asyncio.run(self.send_photo(photo_path, caption))
        except Exception as e:
            print(f"Error in send_photo_sync: {e}")
            return False

async def main():
    # --- 예제 사용법 ---
    # 1. 알림 객체 생성
    # 이 스크립트를 실행하기 전에 config/credentials.py에 실제 토큰과 채팅 ID를 입력해야 합니다.
    notifier = TelegramNotifier()

    # 2. 메시지 전송 (비동기 방식)
    print("--- 비동기 메시지 테스트 ---")
    await notifier.send_message(
        "안녕하세요! Project ART에서 테스트 메시지를 보내드립니다.\n"
        "이 메시지는 알림 시스템이 정상적으로 작동하는지 확인하기 위한 테스트 메시지입니다."
    )

    # 3. 이미지 전송 (테스트용 이미지 파일 생성)
    print("\n--- 비동기 이미지 테스트 ---")
    test_photo_path = 'test_plot.png'
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 테스트용 간단한 플롯 생성
        plt.figure(figsize=(8, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title('테스트 플롯')
        plt.savefig(test_photo_path)
        plt.close()
        
        # 비동기로 사진 전송
        await notifier.send_photo(test_photo_path, caption="Project ART 테스트 플롯")
        
        # 테스트 파일 정리
        if os.path.exists(test_photo_path):
            os.remove(test_photo_path)
            print(f"테스트 파일 {test_photo_path}가 삭제되었습니다.")
            
    except ImportError:
        print("matplotlib이 설치되어 있지 않아 이미지 테스트를 건너뜁니다.")
        
    # 4. 동기식 래퍼 메서드 테스트
    print("\n--- 동기식 래퍼 메서드 테스트 ---")
    notifier.send_message_sync(
        "이 메시지는 동기식 래퍼 메서드를 테스트하기 위한 메시지입니다.\n"
        "이 메시지를 받으셨다면 동기식 래퍼 메서드가 정상적으로 작동합니다."
    )

if __name__ == '__main__':
    # 비동기 메인 함수 실행
    asyncio.run(main())
