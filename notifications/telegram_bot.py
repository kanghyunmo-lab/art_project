import telegram
import os

# 프로젝트 루트 경로를 기준으로 모듈을 임포트하기 위한 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# API 키는 config 파일에서 로드 (존재하지 않을 경우를 대비한 예외 처리 포함)
try:
    from config.credentials import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    print("Warning: credentials.py not found. Using dummy credentials. Please create config/credentials.py")
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID" # Can be a numeric ID or a channel name like '@channelname'

class TelegramNotifier:
    """
    텔레그램 봇을 통해 메시지와 이미지를 전송합니다.
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
            self.chat_id = None # 명시적으로 None 설정
            print("Telegram token is not configured or is a placeholder. Notifier will be disabled.")
        elif chat_id == "YOUR_TELEGRAM_CHAT_ID" or not chat_id:
            self.bot = telegram.Bot(token=token) # 토큰은 유효할 수 있으나 chat_id가 문제
            self.chat_id = None
            print("Telegram chat_id is not configured or is a placeholder. Notifier will be disabled for sending.")
        else:
            self.bot = telegram.Bot(token=token)
            self.chat_id = chat_id

    def send_message(self, text):
        """
        지정된 채팅 ID로 텍스트 메시지를 보냅니다.

        :param text: (str) 보낼 메시지 내용
        :return: (bool) 성공 여부
        """
        if not self.bot or not self.chat_id:
            print(f"Telegram disabled. Message not sent: {text}")
            return False
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text)
            print(f"Telegram message sent successfully.")
            return True
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False

    def send_photo(self, photo_path, caption=""):
        """
        지정된 채팅 ID로 사진을 보냅니다.

        :param photo_path: (str) 보낼 사진 파일의 경로
        :param caption: (str, optional) 사진과 함께 보낼 캡션
        :return: (bool) 성공 여부
        """
        if not self.bot or not self.chat_id:
            print(f"Telegram disabled. Photo not sent: {photo_path}")
            return False
        
        if not os.path.exists(photo_path):
            print(f"Error: Photo file not found at {photo_path}")
            return False

        try:
            with open(photo_path, 'rb') as photo_file:
                self.bot.send_photo(chat_id=self.chat_id, photo=photo_file, caption=caption)
            print(f"Telegram photo sent successfully: {photo_path}")
            return True
        except Exception as e:
            print(f"Error sending Telegram photo: {e}")
            return False

if __name__ == '__main__':
    # --- 예제 사용법 ---
    # 1. 알림 객체 생성
    # 이 스크립트를 실행하기 전에 config/credentials.py에 실제 토큰과 채팅 ID를 입력해야 합니다.
    notifier = TelegramNotifier()

    # 2. 메시지 전송
    print("--- Sending a test message ---")
    notifier.send_message(
        "Hello from Project ART!\n"
        "This is a test message to confirm the notification system is working."
    )

    # 3. 이미지 전송 (테스트용 이미지 파일 생성)
    print("\n--- Sending a test photo ---")
    test_photo_path = 'test_plot.png'
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 3, 2, 4])
        plt.title("Test Plot")
        plt.savefig(test_photo_path)
        plt.close()
        print(f"Created a dummy plot at '{test_photo_path}'")
        
        notifier.send_photo(test_photo_path, caption="This is a test plot from the backtester.")
        
        # 테스트 후 생성된 파일 삭제
        os.remove(test_photo_path)
        print(f"Removed the dummy plot file.")

    except ImportError:
        print("Matplotlib not installed. Skipping photo sending test.")
    except Exception as e:
        print(f"An error occurred during the photo test: {e}")
