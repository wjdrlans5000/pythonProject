import pandas as pd
import numpy as np
import glob
import os
import requests
import datetime

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
    df['EMA_short'] = df['종가'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['종가'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_prev'] = df['MACD'].shift(1)
    df['Signal_prev'] = df['Signal'].shift(1)
    return df

# ----------------------------
# 볼린저밴드 계산 함수
# ----------------------------
def compute_bollinger(df, window=20, num_std=2):
    df['BB_MID'] = df['종가'].rolling(window).mean()
    df['BB_STD'] = df['종가'].rolling(window).std()
    df['BB_UPPER'] = df['BB_MID'] + num_std * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - num_std * df['BB_STD']
    return df

# RSI 계산
def compute_rsi(df, window=14):
    delta = df['종가'].diff()
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
    high = df['고가']
    low = df['저가']
    close = df['종가']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(span=window, adjust=False).mean()

    up_move = high.diff()
    down_move = low.diff() * -1

    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)

    df['+DI'] = (plus_dm.ewm(span=window, adjust=False).mean() / df['ATR']) * 100
    df['-DI'] = (minus_dm.ewm(span=window, adjust=False).mean() / df['ATR']) * 100
    df['ADX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(span=window, adjust=False).mean() * 100

    return df

# 이동평균선 기울기 추가
def compute_ma_slope(df, window=20):
    df['MA'] = df['종가'].rolling(window).mean()
    df['MA_SLOPE'] = df['MA'].diff()
    return df

# ----------------------------
# 백테스트 함수 (윈도우 기반)
# ----------------------------
def backtest_with_window(df, window_days, adx_threshold=20, init_capital=1_000_000):
    trades = []
    cash = init_capital
    position = 0
    entry_price, entry_date, entry_reason = 0.0, None, None

    # 마지막 신호 여부 기록용
    last_signal = {"매수신호": None, "매도신호": None}

    i = 1
    while i < len(df):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        current_price = float(row['종가'])
        current_adx = float(row['ADX'])
        current_pdi = float(row['+DI'])
        current_mdi = float(row['-DI'])
        ema_short = float(row['EMA_short']) if not pd.isna(row['EMA_short']) else np.nan
        ema_long  = float(row['EMA_long']) if not pd.isna(row['EMA_long']) else np.nan
        ma_slope  = float(row['MA_SLOPE']) if not pd.isna(row['MA_SLOPE']) else 0.0


        # 미래 window_days 구간 계산 (날짜 기준)
        current_date = row['일자']
        end_date = current_date + pd.Timedelta(days=window_days)
        # 오늘부터 20일 데이터 future 에 저장
        future = df[(df['일자'] >= current_date) & (df['일자'] <= end_date)]

        # 추세 여부 판별
        margin = 3  # -DI와 +DI 차이 기준

        # 상승추세
        is_uptrend = (current_adx >= adx_threshold) and (current_pdi > current_mdi) and ( (ema_short > ema_long) or (ma_slope > 0.0) )
        # 하락추세
        is_downtrend = (
                (current_adx >= adx_threshold) and
                (current_mdi >= current_pdi + margin) and
                ( (ema_short < ema_long) or (ma_slope < 0.0) ) and
                (current_price < ema_short)
        )
        is_range = not (is_uptrend or is_downtrend)

        # 매수 조건
        if position == 0:
            # 어제 macd signal 아래, 오늘 macd 시그널 돌파
            if (prev_row['MACD'] <= prev_row['Signal']) and (row['MACD'] > row['Signal']):
                # 어제 macd 0 이하인데 오늘 macd 0 돌파
                zero_cross = future[(future['MACD'] >= 0) & (future['MACD_prev'] < 0)]
                # 오늘 종가가 bb중단선 돌파 & 어제 종가가 bb중단선 아래
                bb_cross = future[(future['종가'] > future['BB_MID']) &
                                  (future['종가'].shift(1) <= future['BB_MID'].shift(1))]

                candidate = []
                if not zero_cross.empty and is_uptrend:
                    candidate.append((zero_cross.index[0], "상승추세 MACD 0선 돌파"))
                if not bb_cross.empty and (not is_downtrend):
                    candidate.append((bb_cross.index[0], "BB 중단선 상향 돌파"))
                if row['RSI'] < 30 and is_range:
                    candidate.append((i, "박스권 RSI 과매도로 인한 매수"))

                if candidate:
                    idx_cross, reason = sorted(candidate, key=lambda x: x[0])[0]
                    entry_row = df.loc[idx_cross]
                    potential_entry_price = float(entry_row['종가']) * 1.001
                    qty = int(np.floor(cash / potential_entry_price))
                    if qty > 0:
                        entry_price = potential_entry_price
                        position = qty
                        entry_date = entry_row['일자']
                        entry_reason = reason
                        cash -= position * entry_price
                        # 매수일로 점프
                        i = idx_cross +1
                        continue

        # 매도 조건
        else:
            # 어제 macd가 signal 보다 위 , 오늘 macd 시그널 보다 하락
            if (prev_row['MACD'] >= prev_row['Signal']) and (row['MACD'] < row['Signal']):
                candidate = []
                if is_uptrend:  # 추세장 매도
                    # 오늘 macd 0 보다 작고 어제 macd 0 보다 클때 0 하락돌파
                    zero_cross = future[(future['MACD'] < 0) & (future['MACD_prev'] >= 0)]
                    if not zero_cross.empty:
                        candidate.append((zero_cross.index[0], "추세장 MACD 0선 하락 돌파"))
                elif is_downtrend:
                    candidate.append((i, "하락 추세장 매도"))
                else:  # 박스권 매도
                    # 오늘 macd 0 보다 작고 어제 macd 0 보다 클때 0 하락돌파
                    zero_cross = future[(future['MACD'] < 0) & (future['MACD_prev'] >= 0)]
                    # 어제 종가가 어제 bb중단보다 컸는데, 오늘 종가가 bb_mid 아래로 내러감
                    bb_cross = future[(future['종가'].shift(1) >= future['BB_MID'].shift(1))
                                        & (future['종가'] < future['BB_MID'])]
                    if not zero_cross.empty:
                        candidate.append((zero_cross.index[0], "박스권 MACD 0선 하락 돌파"))
                    if not bb_cross.empty:
                        candidate.append((bb_cross.index[0], "박스권 BB 중단선 하락 돌파"))
                    if row['RSI'] > 72:
                        candidate.append((i, "박스권 RSI 과매수"))

                if candidate:
                    idx_cross, reason = sorted(candidate, key=lambda x: x[0])[0]
                    if entry_date is not None and df.loc[idx_cross, '일자'] == entry_date:
                        i += 1
                        continue
                    exit_row = df.loc[idx_cross]
                    exit_price = float(exit_row['종가']) * 0.999
                    cash += position * exit_price
                    pnl = (exit_price - entry_price) * position
                    ret = (exit_price / entry_price - 1) * 100
                    holding_days = (exit_row['일자'] - entry_date).days if entry_date is not None else None
                    trades.append({
                        '매수일자': entry_date,
                        '매수가격': entry_price,
                        '매도일자': exit_row['일자'],
                        '매도가격': exit_price,
                        '수량': position,
                        '손익금액': pnl,
                        '수익률(%)': ret,
                        '보유일수': holding_days,
                        '매수사유': entry_reason,
                        '매도사유': reason
                    })
                    position, entry_price, entry_date, entry_reason = 0, 0.0, None, None
                    # 매도일로 점프
                    i = idx_cross
                    continue

        i += 1

    # 마지막 강제 청산
    if position > 0:
        last_row = df.iloc[-1]
        exit_price = float(last_row['종가']) * 0.999
        cash += position * exit_price
        pnl = (exit_price - entry_price) * position
        ret = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
        holding_days = (last_row['일자'] - entry_date).days if entry_date is not None else None
        trades.append({
            '매수일자': entry_date,
            '매수가격': entry_price,
            '매도일자': last_row['일자'],
            '매도가격': exit_price,
            '수량': position,
            '손익금액': pnl,
            '수익률(%)': ret,
            '보유일수': holding_days,
            '매수사유': entry_reason,
            '매도사유': '강제청산(기간종료)'
        })

    # 마지막 날짜 기준 매수/매도 신호 확인
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    if (prev_row['MACD'] <= prev_row['Signal']) and (last_row['MACD'] > last_row['Signal']):
        last_signal["매수신호"] = "MACD 상승 돌파"
    elif (prev_row['MACD'] >= prev_row['Signal']) and (last_row['MACD'] < last_row['Signal']):
        last_signal["매도신호"] = "MACD 하락 돌파"

    if (prev_row['종가'] <= prev_row['BB_MID']) and (last_row['종가'] > last_row['BB_MID']):
        last_signal["매수신호"] = "BB 중단선 상향 돌파"
    elif (prev_row['종가'] >= prev_row['BB_MID']) and (last_row['종가'] < last_row['BB_MID']):
        last_signal["매도신호"] = "BB 중단선 하락 돌파"

    final_equity = cash
    total_return_pct = (final_equity - init_capital) / init_capital * 100
    win_rate = (sum(1 for t in trades if t['손익금액'] > 0) / len(trades) * 100) if trades else 0

    return {
        'window_days': window_days,
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate_pct': win_rate,
        'trades': trades,
        'last_signal': last_signal
    }
# ----------------------------
# 실행: data/*.xlsx 를 순회
# ----------------------------
if __name__ == "__main__":
    today = datetime.datetime.today().strftime("%Y%m%d")
    files = glob.glob(f"data/{today}/*.xlsx")
    if not files:
        print("data 폴더에 .xlsx 파일이 없습니다. 파일 위치를 확인하세요.")

    # 모든 요약 결과 저장용 리스트
    all_summaries = []

    for file in files:
        stock_name = os.path.splitext(os.path.basename(file))[0]
        print(f"\n▶ 처리 시작: {stock_name}")

        df = pd.read_excel(file, sheet_name="Sheet1")
        df['일자'] = pd.to_datetime(df['일자'])
        df = df.sort_values('일자').reset_index(drop=True)
        # MACD + Bollinger 계산
        df = compute_macd(df, short=12, long=26, signal=9)
        df = compute_bollinger(df, window=20, num_std=2)
        df = compute_rsi(df, window=14)
        df = compute_adx(df, window=14)
        df = compute_ma_slope(df, window=20)
        # 단기
        # df = compute_macd(df, short=5, long=34, signal=5)
        # df = compute_bollinger(df, window=10, num_std=2)
        # df = compute_rsi(df, window=7)
        # df = compute_adx(df, window=10)
        # df = compute_ma_slope(df, window=10)

        results = []
        # 날짜 폴더 경로 지정
        output_dir = os.path.join(os.getcwd(), today)

        # 폴더 없으면 생성
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"backtest_results_{stock_name}.xlsx")

        writer = pd.ExcelWriter(output_path, engine="openpyxl")

        for n in [20]:
            r = backtest_with_window(df, adx_threshold=20, window_days=n)
            results.append(r)

            trades_df = pd.DataFrame(r['trades'])
            if trades_df.empty:
                trades_df = pd.DataFrame(columns=['매수일자','매수가격','매도일자','매도가격',
                                                  '수량','손익금액','수익률(%)','누적수익률(%)',
                                                  '보유일수','매수사유','매도사유'])
            trades_df.to_excel(writer, sheet_name=f"매매내역_N{n}", index=False)

            # 마지막 신호 기록
            #last_signal_df = pd.DataFrame([r['last_signal']])
            #last_signal_df.to_excel(writer, sheet_name=f"마지막신호_N{n}", index=False)

        # 요약 저장
        summary_df = pd.DataFrame([{
            "종목명": stock_name,
            "최종자산": r['final_equity'],
            "총수익률(%)": r['total_return_pct'],
            "거래횟수": r['num_trades'],
            "승률(%)": r['win_rate_pct'],
            "마지막매수신호": r['last_signal'].get("매수신호", None),
            "마지막매도신호": r['last_signal'].get("매도신호", None)
        } for r in results])
        summary_df.to_excel(writer, sheet_name="요약", index=False)

        writer.close()
        print(f"완료: {output_path} 저장됨")
        print("\n=== 요약 ===")
        print(summary_df.to_string(index=False))
        # FinalReport용으로 누적
        all_summaries.append(summary_df)

    # 모든 종목 요약을 하나의 DataFrame으로 합치기
    if all_summaries:
        final_report_df = pd.concat(all_summaries, ignore_index=True)
        final_report_path = "FinalReport.xlsx"
        final_report_df.to_excel(final_report_path, sheet_name="요약", index=False)
        print(f"\n📊 모든 요약을 합쳐서 {final_report_path} 저장 완료!")

        # 전체 수익률 평균 (또는 합계 선택 가능)
        avg_return = final_report_df["총수익률(%)"].mean()

        # 텔레그램용 문자열 만들기
        report_text = "📊 백테스트 결과 요약\n\n"
        for idx, row in final_report_df.iterrows():
         # 마지막매수, 마지막매도 둘 다 None 이면 skip
            if pd.isna(row['마지막매수신호']) and pd.isna(row['마지막매도신호']):
                continue
            report_text += (
                f"✅ {row['종목명']}\n"
                f"  - 최종자산: {row['최종자산']:,}\n"
                f"  - 총수익률: {row['총수익률(%)']:.2f}%\n"
                f"  - 거래횟수: {row['거래횟수']}\n"
                f"  - 승률: {row['승률(%)']:.2f}%\n"
                f"  - 마지막매수신호: {row['마지막매수신호']}\n"
                f"  - 마지막매도신호: {row['마지막매도신호']}\n\n"
            )

        # 전체 대상 종목 수
        total_stocks = len(final_report_df)
        # 최종자산 합계
        total_assets = final_report_df["최종자산"].sum()

        # 신호발생 종목 수 (마지막매수 or 마지막매도 둘 중 하나라도 있는 경우)
        signal_count = final_report_df[
            final_report_df["마지막매수신호"].notna() | final_report_df["마지막매도신호"].notna()
            ].shape[0]

        # 전체 요약 추가
        report_text += (
            "=====================\n"
            f"📊 전체 대상 종목 수: {total_stocks}\n"
            f"💰 전체 최종자산 합계: {total_assets:,}\n"
            f"📌 신호발생 종목 수: {signal_count}\n"
            f"📈 전체 평균 수익률: {avg_return:.2f}%\n"
        )
        print(report_text)

        BOT_TOKEN = ""
        CHAT_ID = ""
        send_telegram_message(BOT_TOKEN, CHAT_ID, report_text)