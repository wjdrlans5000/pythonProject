import time
import requests
import os
from stock_trade_by_backTesting import backtest_stock
from stock_analyzer import StockAnalyzer

# 설정 파일 읽기
def load_config():
    config = {}
    with open("telegram_config.txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key.strip()] = value.strip()
    return config

config = load_config()
BOT_TOKEN = config.get("BOT_TOKEN")
ALLOWED_CHAT_ID = config.get("CHAT_ID")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# 메시지 보내기
def send_message(chat_id, text):
    url = f"{BASE_URL}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    requests.post(url, data=data)

# 명령 처리
def handle_command(chat_id, text):
    parts = text.split()
    analyzer = StockAnalyzer()
    if text.startswith("/help"):
        send_message(chat_id, "✅ 주식 분석 봇이 준비되었습니다.\n명령어: /t TSLA 또는 /t 005930.KS(삼성전자)")
    elif text.startswith("/b"):
        if len(parts) < 2:
            send_message(chat_id, "❗ 사용법: /b TSLA")
        else:
            symbol = parts[1].upper()
            backtest_stock(symbol)
            # send_message(chat_id, backtest_report)
    elif text.startswith("/s"):
        if len(parts) < 2:
            send_message(chat_id, "❗ 사용법: /s TSLA")
        else:
            symbol = parts[1].upper()
            analyzer_report = analyzer.get_stock_info(symbol)
            send_message(chat_id, analyzer_report)
    else:
        send_message(chat_id, "❗ 알 수 없는 명령입니다.\n/b 또는 /s 명령을 사용하세요.")


# 메시지 수신 루프
def run_bot():
    last_update_id = None
    send_message(ALLOWED_CHAT_ID, "✅ 주식 분석 텔레그램 봇이 실행되었습니다.")

    while True:
        try:
            url = f"{BASE_URL}/getUpdates"
            if last_update_id:
                url += f"?offset={last_update_id + 1}"

            response = requests.get(url).json()

            if "result" in response:
                for update in response["result"]:
                    last_update_id = update["update_id"]

                    if "message" in update:
                        chat_id = str(update["message"]["chat"]["id"])
                        text = update["message"].get("text", "")

                        if chat_id == ALLOWED_CHAT_ID:
                            handle_command(chat_id, text)
                        else:
                            send_message(chat_id, "⛔ 권한이 없습니다.")

            time.sleep(1)  # 서버 부하 방지

        except Exception as e:
            print(f"에러 발생: {e}")
            time.sleep(3)

if __name__ == "__main__":
    run_bot()
