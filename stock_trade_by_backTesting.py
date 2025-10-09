# backtesting_1002_fixed.py
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import requests
import datetime
import os


def load_telegram_config(config_path="telegram_config.txt"):
    """
    같은 경로의 telegram_config.txt 파일에서 BOT_TOKEN과 CHAT_ID를 읽어옴
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ 설정 파일을 찾을 수 없습니다: {config_path}")

    bot_token = None
    chat_id = None

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # 주석이나 공백 무시
                continue
            if line.startswith("BOT_TOKEN="):
                bot_token = line.split("=", 1)[1].strip()
            elif line.startswith("CHAT_ID="):
                chat_id = line.split("=", 1)[1].strip()

    if not bot_token or not chat_id:
        raise ValueError("⚠️ 설정 파일에 BOT_TOKEN 또는 CHAT_ID가 없습니다.")

    return bot_token, chat_id

# 텔레그램 전송 함수
def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    response = requests.post(url, data=payload)
    return response.json()


# ----------------------------
# MACD & Signal 계산 함수
# ----------------------------
def compute_macd(df, short=12, long=26, signal=9):
    df['EMA_short'] = df['Close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_prev'] = df['MACD'].shift(1)
    df['Signal_prev'] = df['Signal'].shift(1)
    return df

# ----------------------------
# 볼린저밴드 계산 함수
# ----------------------------
def compute_bollinger(df, window=20, num_std=2):
    df['BB_MID'] = df['Close'].rolling(window).mean()
    df['BB_STD'] = df['Close'].rolling(window).std()
    df['BB_UPPER'] = df['BB_MID'] + num_std * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - num_std * df['BB_STD']
    return df

# RSI 계산
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# ADX 계산
# ADX + DI 계산
def compute_adx(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(span=window, adjust=False).mean()

    up_move = high.diff()
    down_move = low.diff() * -1

    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)

    df['PDI'] = (plus_dm.ewm(span=window, adjust=False).mean() / df['ATR']) * 100
    df['MDI'] = (minus_dm.ewm(span=window, adjust=False).mean() / df['ATR']) * 100
    df['ADX'] = (abs(df['PDI'] - df['MDI']) / (df['PDI'] + df['MDI'])).ewm(span=window, adjust=False).mean() * 100

    return df

# 이동평균선 기울기 추가
def compute_ma_slope(df, window=20):
    df['MA'] = df['Close'].rolling(window).mean()
    df['MA_SLOPE'] = df['MA'].diff()
    return df

# ----------------------------
# Strategy (데이터프레임에서 계산된 지표 사용)
# ----------------------------
class MultiIndicatorStrategy(Strategy):
    window_days = 20
    adx_threshold = 20
    size_pct = 1.0

    def init(self):
        # int(self.equity / self.data.Close[-1]) # cash 내 구매가능 수량
        # 백테스트용 전체 df에서 마지막날 기록
        self._last_day = self.data.index[-1]

        # 가격/지표(데이터프레임에 이미 계산되어 있어야 함)
        self.close    = self.I(lambda x: x, self.data.Close)
        self.macd     = self.I(lambda x: x, self.data.MACD)
        self.signal   = self.I(lambda x: x, self.data.Signal)

        self.bb_mid   = self.I(lambda x: x, self.data.BB_MID)
        self.bb_upper = self.I(lambda x: x, self.data.BB_UPPER)
        self.bb_lower = self.I(lambda x: x, self.data.BB_LOWER)

        self.rsi      = self.I(lambda x: x, self.data.RSI)
        self.ma_slope = self.I(lambda x: x, self.data.MA_SLOPE)

        # ADX 계열 (PDI/MDI/ADX) — 반드시 df에 PDI,MDI,ADX 컬럼이 있어야 함
        self.pdi      = self.I(lambda x: x, self.data.PDI)
        self.mdi      = self.I(lambda x: x, self.data.MDI)
        self.adx      = self.I(lambda x: x, self.data.ADX)
        self.ema_short      = self.I(lambda x: x, self.data.EMA_short)
        self.ema_long      = self.I(lambda x: x, self.data.EMA_long)

        self.pending_buy = None
        self.pending_sell = None
        self.trade_logs = []

    # 헬퍼: 추세판단 (인덱스는 음수 인덱스로 사용 가능)
    def is_uptrend(self, idx):
        try:
            return (self.adx[idx] >= self.adx_threshold) and (self.pdi[idx] > self.mdi[idx]) and \
                ((self.ema_short[idx] > self.ema_long[idx]) or (self.ma_slope[idx] > 0))
        except Exception:
            return False

    def is_downtrend(self, idx):
        try:
            margin = 3
            return (self.adx[idx] >= self.adx_threshold) and (self.mdi[idx] >= self.pdi[idx] + margin) and \
                ((self.ema_short[idx] < self.ema_long[idx]) or (self.ma_slope[idx] < 0)) and \
                (self.close[idx] < self.ema_short[idx])
        except Exception:
            return False

    def next(self):
        # 안전하게 이전 및 현재 값 얻기
        try:
            i0 = -1  # today
            i1 = -2  # yesterday
            macd_prev = float(self.macd[i1])
            sig_prev  = float(self.signal[i1])
            macd_now  = float(self.macd[i0])
            sig_now   = float(self.signal[i0])
        except Exception:
            return

        # 매수 신호: MACD 골든 크로스 감지 -> pending_buy 설정
        if (not self.position or self.position.is_short) and (macd_prev <= sig_prev) and (macd_now > sig_now):
            self.pending_buy = {'remaining': self.window_days, 'reason_signal': 'MACD골든크로스'}
            # print({'self.pending_buy': self.pending_buy
            #           ,'당일종가' : self.close[i0]
            #           ,'time': self.data.index[-1]})
            # 즉시 동일일 진입 가능성 체크 (MACD 0 돌파 or BB 돌파)
            try:
                if (macd_now >= 0)  and self.is_uptrend(i0):
                    size = int(self.equity / self.data.Close[-1])
                    if size > 0:
                        self.buy(size=size)
                        self.trade_logs.append(('BUY', 'MACD0', self.data.index[-1], self.close[-1]))
                        self.pending_buy = None
                    return
            except Exception:
                pass
            try:
                if (self.close[i0] > self.bb_mid[i0]) and (self.close[i1] <= self.bb_mid[i1]) and (not self.is_downtrend(i0)):
                    size = int(self.equity / self.data.Close[-1])
                    if size > 0:
                        self.buy(size=size)
                        self.trade_logs.append(('BUY', 'BB_UP', self.data.index[-1], self.close[-1]))
                        self.pending_buy = None
                    return
            except Exception:
                pass

        # 매도 신호: MACD 데드크로스 감지 -> pending_sell 설정
        if self.position.is_long and (macd_prev >= sig_prev) and (macd_now < sig_now):
            self.pending_sell = {'remaining': self.window_days, 'reason_signal': 'MACD데드크로스'}
            # print({'self.pending_sell': self.pending_sell
            #           ,'time': self.data.index[-1]})
            # 즉시 조건 체크
            try:
                if (macd_now < 0) and (self.macd[i1] >= 0) and self.is_uptrend(i0):
                    # self.sell() # 얘는 매도와 동시에 숏포지션 잡음
                    self.position.close() # 얘는 청산만
                    self.trade_logs.append(('SELL', 'MACD0_DOWN', self.data.index[-1], self.close[-1]))
                    self.pending_sell = None
                    return
            except Exception:
                pass
            try:
                if (self.close[i1] >= self.bb_mid[i1]) and (self.close[i0] < self.bb_mid[i0]):
                    self.position.close()
                    self.trade_logs.append(('SELL', 'BB_DOWN', self.data.index[-1], self.close[-1]))
                    self.pending_sell = None
                    return
                if self.rsi[i0] > 72:
                    self.position.close()
                    self.trade_logs.append(('SELL', 'RSI_OVER', self.data.index[-1], self.close[-1]))
                    self.pending_sell = None
                    return
            except Exception:
                pass

        # pending_buy 체크
        # 당일엔 보조 조건이 안 맞아서 진입 실패했다면, self.pending_buy가 살아있음.
        # 이후 최대 window_days (20일) 동안 지켜보다가
        # 조건(예: MACD 0선 돌파, BB 상향 돌파, RSI 저점 반등 등)이 나오면 매수 체결.
        if self.pending_buy is not None:
            self.pending_buy['remaining'] -= 1
            # print({'self.remaining': self.pending_buy['remaining']
            #           ,'is_uptrend(i0)':self.is_uptrend(i0)
            #           ,'is_downtrend(i0)':self.is_downtrend(i0)
            #           ,'당일.macd[i0]': self.macd[i0]
            #           ,'전일.macd[i1]': self.macd[i1]
            #           ,'당일종가' : self.close[i0]
            #           ,'당일 BB_MID' : self.bb_mid[i0]
            #           ,'time': self.data.index[i0]
            #        })
            try:
                if (self.macd[i0] >= 0) and (self.macd[i1] < 0) and self.is_uptrend(i0):
                    size = int(self.equity / self.data.Close[-1])
                    if size > 0:
                        self.buy(size=size)
                        self.trade_logs.append(('BUY', 'MACD0_pending', self.data.index[-1], self.close[-1]))
                        self.pending_buy = None
                    return
                if (self.close[i0] > self.bb_mid[i0]) and (self.close[i1] <= self.bb_mid[i1]) and (not self.is_downtrend(i0)):
                    size = int(self.equity / self.data.Close[-1])
                    if size > 0:
                        self.buy(size=size)
                        self.trade_logs.append(('BUY', 'BB_pending', self.data.index[-1], self.close[-1]))
                        self.pending_buy = None
                    return
                if (self.rsi[i0] < 30) and (not self.is_uptrend(i0)) and (not self.is_downtrend(i0)):
                    size = int(self.equity / self.data.Close[-1])
                    if size > 0:
                        self.buy(size=size)
                        self.trade_logs.append(('BUY', 'RSI_box_pending', self.data.index[-1], self.close[-1]))
                        self.pending_buy = None
                    return
            except Exception:
                pass
            if self.pending_buy['remaining'] <= 0:
                self.pending_buy = None

        # pending_sell 체크
        if self.pending_sell is not None:
            self.pending_sell['remaining'] -= 1
            # print({'self.remaining': self.pending_sell['remaining']
            #           ,'is_uptrend(i0)':self.is_uptrend(i0)
            #           ,'당일.macd[i0]': self.macd[i0]
            #           ,'전일.macd[i1]': self.macd[i1]
            #           ,'당일종가' : self.close[i0]
            #           ,'당일 BB_MID' : self.bb_mid[i0]
            #           ,'time': self.data.index[i0]
            #        })
            try:
                if self.is_uptrend(i0):
                    if (self.macd[i0] < 0) and (self.macd[i1] >= 0):
                        self.position.close()
                        self.trade_logs.append(('SELL', 'MACD0_pending', self.data.index[-1], self.close[-1]))
                        self.pending_sell = None
                        return
                else:
                    if (self.macd[i0] < 0) and (self.macd[i1] >= 0):
                        self.position.close()
                        self.trade_logs.append(('SELL', 'MACD0_pending', self.data.index[-1], self.close[-1]))
                        self.pending_sell = None
                        return
                    if (self.close[i1] >= self.bb_mid[i1]) and (self.close[i0] < self.bb_mid[i0]):
                        self.position.close()
                        self.trade_logs.append(('SELL', 'BB_pending', self.data.index[-1], self.close[-1]))
                        self.pending_sell = None
                        return
                    if self.rsi[i0] > 72:
                        self.position.close()
                        self.trade_logs.append(('SELL', 'RSI_pending', self.data.index[-1], self.close[-1]))
                        self.pending_sell = None
                        return
            except Exception:
                pass
            if self.pending_sell['remaining'] <= 0:
                self.pending_sell = None

        # 마지막날 강제 청산 (오늘 남아 있는 포지션만)
        if self.data.index[i0] == self._last_day:
            if self.position and self.position.is_long:

                self.position.close()
                self.trade_logs.append(('FINAL_CLOSE', 'FINAL_CLOSE', self.data.index[i0], self.close[i0]))
                return

# ----------------------------
# 실행: yfinance → 지표 계산 → backtest
# ----------------------------
if __name__ == "__main__":
    BOT_TOKEN, CHAT_ID = load_telegram_config()
    symbols = [
        "TSLA", "TEM", "NVDA", "ORCL", "QQQ", "SPY", "FIG", "MSFT", "META", "AMZN", "GOOGL", "TSM", "AAPL", "IONQ", "AVGO"
    ]
    # symbol = "TSLA"
    # symbol = "005930.KS"
    start = "2023-01-01"   # 실제 분석 시작보다 최소 2달 앞
    end   = datetime.datetime.today().strftime("%Y-%m-%d")

    results = []  # 전체 결과 저장용

    for symbol in symbols:
        print(f"\n📈 === {symbol} 백테스트 시작 ===")

        # 데이터 다운로드
        df = yf.download(symbol, start=start, end=end, group_by="column", auto_adjust=False)
        df.columns = df.columns.droplevel(1)

        # backtesting이 기대하는 컬럼명 보장
        df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
        df = df.dropna().copy()

        # === 여기서 지표를 전체 DF 기준으로 먼저 계산 ===
        # MACD + Bollinger 계산
        df = compute_macd(df, short=12, long=26, signal=9)
        df = compute_bollinger(df, window=20, num_std=2)
        df = compute_rsi(df, window=14)
        df = compute_adx(df, window=14)
        df = compute_ma_slope(df, window=20)

        # 실제 테스트 구간 (지표 안정화 이후)
        df = df.loc["2023-10-02":].dropna().copy()
        print(df.head())

        # 백테스트 실행
        bt = Backtest(df, MultiIndicatorStrategy, cash=1000, commission=0,
                      exclusive_orders=True, finalize_trades=True, trade_on_close=True)
        stats = bt.run()

        # BUY/SELL/FINAL_CLOSE 구분용
        logs = getattr(stats._strategy, "trade_logs", [])
        trade_df = pd.DataFrame(logs, columns=["Signal", "Reason", "Date", "Price"]) if logs else pd.DataFrame()
        if not trade_df.empty:
            real_trades = trade_df[trade_df["Signal"].isin(["BUY", "SELL"])].shape[0]
            has_final_close = "FINAL_CLOSE" in trade_df["Signal"].values
            last_log = trade_df.iloc[-1].to_dict()
            last_log_str = f"{last_log['Signal']} ({last_log['Reason']}) @ {last_log['Date'].strftime('%Y-%m-%d')} ${round(float(last_log['Price']),2)}"
        else:
            real_trades = 0
            has_final_close = False
            last_log_str = "None"


        # 결과 요약 저장
        result = {
            "Symbol": symbol,
            "Start": df.index[0].strftime("%Y-%m-%d"),
            "End": df.index[-1].strftime("%Y-%m-%d"),
            "Final Equity": round(stats["Equity Final [$]"], 2),
            "Return [%]": round(stats["Return [%]"], 2),
            "Buy & Hold Return [%]": round(stats["Buy & Hold Return [%]"], 2),
            "Win Rate [%]": round(stats["Win Rate [%]"], 2),
            "Trades": int(stats["# Trades"]),
            "Best Trade [%]": round(stats["Best Trade [%]"], 2),
            "Worst Trade [%]": round(stats["Worst Trade [%]"], 2),
            "Has FINAL_CLOSE": has_final_close,
            "Last Signal Log": last_log_str
        }

        results.append(result)

        # Equity, Return 등은 원금 cash 대비 총 자산 및 수익률을 의미
        print(stats)
        print("\n".join(str(log) for log in getattr(stats._strategy, "trade_logs", [])))
        # bt.plot()

        # 신호발생 종목 텔레그램 알람추가
        if not has_final_close and last_log['Date'].strftime('%Y-%m-%d') >= (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d") :
            report_text = (
                "=====================\n"
                f"📊 종목명: {symbol}\n"
                f"💰 수익률: {round(stats["Return [%]"], 2):,}\n"
                f"📌 마지막신호: {last_log_str}\n"
            )
            send_telegram_message(BOT_TOKEN, CHAT_ID, report_text)

    # ====== 결과 요약표 ======
    summary_df = pd.DataFrame(results)
    print("\n📊 전체 종목 결과 요약:")
    print(summary_df.to_string(index=False))




