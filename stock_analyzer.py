# stock_analyzer.py
import requests
import yfinance as yf
from yahooquery import search as yq_search
import pandas as pd
import datetime
from typing import Optional
from urllib.parse import quote
import json

class StockAnalyzer:
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
        )

    # ---------- formatting helpers ----------
    def format_price(self, value, ticker: str) -> str:
        if value is None:
            return "N/A"
        try:
            if ticker.endswith(".KS") or ticker.endswith(".KQ"):
                return f"₩{value:,.0f}"
            else:
                return f"${value:,.2f}"
        except Exception:
            return str(value)

    def format_market_cap(self, value, ticker: str) -> str:
        if value is None or value == 0:
            return "N/A"
        try:
            # Korean stock market (원화)
            if ticker.endswith(".KS") or ticker.endswith(".KQ"):
                # value expected in KRW
                # 1조 = 1e12, 1억 = 1e8
                if value >= 1_000_000_000_000:
                    return f"₩{value / 1_000_000_000_000:.1f}조"
                if value >= 100_000_000:
                    return f"₩{value / 100_000_000:.1f}억"
                return f"₩{value:,.0f}"
            else:
                # USD market
                if value >= 1_000_000_000_000:
                    return f"${value / 1_000_000_000_000:.1f}T"
                if value >= 1_000_000_000:
                    return f"${value / 1_000_000_000:.1f}B"
                if value >= 1_000_000:
                    return f"${value / 1_000_000:.1f}M"
                return f"${value:,.0f}"
        except Exception:
            return str(value)

    def format_percent(self, value) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return str(value)

    # ---------- ticker/name resolution ----------
    def get_ticker_by_name(self, query: str) -> Optional[str]:
        """
        Accepts company name in Korean/English or ticker.
        Returns a Yahoo-style ticker (e.g., '005930.KS' or 'AAPL') or None.
        Priority:
         1) If input already looks like Yahoo ticker (.KS/.KQ or uppercase symbol) -> return it.
         2) Naver AC JSON search for Korean names -> returns code + .KS/.KQ
         3) yahooquery.search backup for international names
         4) final fallback: try yfinance Ticker info check
        """
        if not query:
            return None
        q = query.strip()

        # Already a Yahoo-style ticker?
        if q.endswith((".KS", ".KQ")) or q.isupper():
            return q.upper()

        # 1) Naver AC search (works for Korean company names)
        try:
            url = f"https://ac.stock.naver.com/ac?q={quote(q)}&target=stock"
            headers = {"User-Agent": self.user_agent}
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200 and res.text:
                data = res.json()
                # data: { "query": "...", "items": [ {code, name, typeCode, ...}, ... ] }
                items = data.get("items", [])
                search_query = data.get("query", "").strip()
                if items:
                    # prefer exact name match if available
                    exact = next((it for it in items if it.get("name") == search_query), None)
                    target = exact or items[0]
                    code = target.get("code")
                    type_code = target.get("typeCode", "").upper()  # KOSPI/KOSDAQ
                    nation = target.get("nationName", "")
                    if code:
                        market = ".KS" if type_code == "KOSPI" else ".KQ"
                        # If it's clearly US item, return None to let yahooquery handle
                        if nation and "미국" in nation:
                            return None
                        return f"{code}{market}"
        except Exception:
            # ignore and continue to next resolution
            pass

        # 2) yahooquery search (backup, good for global/US names)
        try:
            res = yq_search(q)
            if isinstance(res, dict) and res.get("quotes"):
                # return first symbol
                symbol = res["quotes"][0].get("symbol")
                if symbol:
                    return symbol.upper()
        except Exception:
            pass

        # 3) final fallback: try yfinance ticker validity (fast_info or history)
        try:
            t = yf.Ticker(q.upper())
            # fast_info might exist; if not, try info or history
            fi = getattr(t, "fast_info", None)
            info = getattr(t, "info", None)
            if fi or (info and info.get("regularMarketPrice") is not None):
                return q.upper()
            # last ditch: if it has history
            hist = t.history(period="1mo")
            if hist is not None and len(hist) > 0:
                return q.upper()
        except Exception:
            pass

        return None

    # ---------- dividend helpers ----------
    def calc_ttm_dividend_and_yield(self, ticker: str, price_raw: float):
        """
        Returns (ttm_dividend_amount, ttm_yield_percent)
        - ttm_dividend_amount: numeric (in same currency units as price)
        - ttm_yield_percent: float percentage (e.g., 1.23 for 1.23%)
        If no dividend data -> (0.0, 0.0)
        """
        try:
            stock = yf.Ticker(ticker)
            divs = getattr(stock, "dividends", None)

            # 빠른 탈출: 데이터 없음
            if divs is None or len(divs) == 0:
                return 0.0, 0.0

            # DataFrame이면 Series로 변환 (일반적으로 yfinance는 Series를 줌)
            if isinstance(divs, pd.DataFrame):
                divs = divs.iloc[:, 0]

            # 안전하게 DatetimeIndex로 변환
            idx = pd.to_datetime(divs.index, errors="coerce")
            # drop invalid timestamps
            valid_mask = ~idx.isna()
            divs = divs.loc[valid_mask]
            idx = idx[valid_mask]

            if len(idx) == 0:
                return 0.0, 0.0

            # 만약 index가 tz-aware이면 동일한 tz로 기준 시점 생성
            tz = idx.tz
            if tz is not None:
                # tz-aware 현재 시각
                now = pd.Timestamp.now(tz=tz)
                one_year_ago = now - pd.DateOffset(months=12)
            else:
                # naive
                now = pd.Timestamp.now()
                one_year_ago = now - pd.DateOffset(months=12)

            # index를 비교 가능한 형태로 유지 (tz-aware이면 그대로, naive이면 그대로)
            # (이미 idx.tz matches one_year_ago.tz above)
            divs.index = idx

            # 1) pandas convenience: last("12M") 사용 시도 (많은 경우에 동작)
            ttm_sum = 0.0
            try:
                last_12m = divs.loc("12ME")
                if not last_12m.empty:
                    ttm_sum = float(last_12m.sum())
                else:
                    # fallback to explicit comparison
                    recent = divs[divs.index >= one_year_ago]
                    ttm_sum = float(recent.sum()) if not recent.empty else 0.0
            except Exception:
                # fallback explicit comparison (handles tz-aware vs naive because one_year_ago was created with matching tz above)
                recent = divs[divs.index >= one_year_ago]
                ttm_sum = float(recent.sum()) if not recent.empty else 0.0

            if price_raw and price_raw > 0 and ttm_sum > 0:
                ttm_yield = (ttm_sum / price_raw) * 100.0
            else:
                ttm_yield = 0.0

            return round(ttm_sum, 8), round(ttm_yield, 4)

        except Exception as e:
            # 개발 중에는 에러 로그 남겨 디버깅에 도움되게
            print(f"[calc_ttm_dividend_and_yield] {ticker} error: {e}")
            return 0.0, 0.0



    # ---------- main info method ----------
    def get_stock_info(self, query: str) -> str:
        """
        query: ticker (AAPL, 005930.KS) or company name (삼성전자)
        returns: formatted multi-line string (ready for telegram)
        """
        if not query or not isinstance(query, str):
            return "❌ 잘못된 입력입니다."

        # 1) resolve ticker if needed
        ticker = query.strip()
        resolved = self.get_ticker_by_name(ticker)
        if resolved:
            ticker = resolved

        # 2) load yfinance info
        try:
            stock = yf.Ticker(ticker)
            info = getattr(stock, "info", {}) or {}
        except Exception as e:
            return f"❌ 데이터 로드 실패: {e}"

        # required numeric price
        price_raw = info.get("regularMarketPrice") or info.get("currentPrice")
        if price_raw is None:
            return f"⚠️ '{ticker}' 종목의 가격 정보를 불러올 수 없습니다."

        # safe numeric extraction for change percent
        change_pct_raw = info.get("regularMarketChangePercent")
        try:
            change_pct_val = float(change_pct_raw) if change_pct_raw is not None else None
        except Exception:
            change_pct_val = None

        # basic fields
        name = info.get("longName") or info.get("shortName") or ticker
        market_cap_raw = info.get("marketCap") or 0
        pe_ratio = info.get("trailingPE")
        eps = info.get("trailingEps")
        pbr = info.get("priceToBook")
        vol = info.get("volume") or None
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        sector = info.get("sector") or "N/A"
        industry = info.get("industry") or "N/A"
        country = info.get("country") or "N/A"

        # 3) dividend calculations
        ttm_div_amount, ttm_yield_pct = self.calc_ttm_dividend_and_yield(ticker, price_raw)

        # 4) format for display
        price_fmt = self.format_price(price_raw, ticker)
        market_cap_fmt = self.format_market_cap(market_cap_raw, ticker)
        change_fmt = self.format_percent(change_pct_val) if change_pct_val is not None else "N/A"
        vol_fmt = f"{int(vol):,}" if vol else "N/A"

        pe_display = f"{pe_ratio:.2f}" if (pe_ratio is not None and pd.notna(pe_ratio)) else "N/A"
        eps_display = f"{eps:.2f}" if (eps is not None and pd.notna(eps)) else "N/A"
        pbr_display = f"{pbr:.2f}" if (pbr is not None and pd.notna(pbr)) else "N/A"

        # dividend displays
        if ttm_div_amount <= 0 or ttm_yield_pct <= 0:
            dividend_line = "💸 배당률: 배당 없음"
        else:
            # show amounts in currency unit consistent with ticker
            # ttm_div_amount returned in same units as price (KRW or USD)
            ttm_div_fmt = self.format_price(ttm_div_amount, ticker)
            dividend_line = f"💸 배당률: 최근 12개월 {ttm_yield_pct:.2f}% ({ttm_div_fmt})"

        # sign
        sign = "-"
        try:
            if change_pct_val is not None:
                if change_pct_val > 0:
                    sign = "▲"
                elif change_pct_val < 0:
                    sign = "▼"
                else:
                    sign = "-"
        except Exception:
            sign = "-"

        # 5) momentum (6 months)
        momentum_text = "데이터 부족"
        try:
            hist6 = stock.history(period="6mo")["Close"]
            if hist6 is not None and len(hist6) > 5:
                recent_change = (hist6.iloc[-1] / hist6.iloc[0] - 1) * 100
                if recent_change > 15:
                    momentum_text = f"최근 6개월 +{recent_change:.1f}% (강한 상승)"
                elif recent_change < -10:
                    momentum_text = f"최근 6개월 {recent_change:.1f}% (약세)"
                else:
                    momentum_text = f"최근 6개월 {recent_change:.1f}% (완만)"
        except Exception:
            momentum_text = "데이터 부족"

        # 6) investment opinion simple rule (B: 적극형)
        opinion_parts = []
        try:
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 10:
                    opinion_parts.append("저평가 가능")
                elif pe_ratio < 25:
                    opinion_parts.append("밸류 적정")
                else:
                    opinion_parts.append("고평가 주의")
            else:
                opinion_parts.append("PER 데이터 부족")
        except Exception:
            opinion_parts.append("PER 판정 불가")

        try:
            # if ttm exists but 10y N/A, still we can judge ttm
            if ttm_yield_pct > 0:
                if ttm_yield_pct > 4:
                    opinion_parts.append("배당 매력 높음 (최근)")
                elif ttm_yield_pct > 1:
                    opinion_parts.append("보통 배당 (최근)")
                else:
                    opinion_parts.append("배당 낮음 (최근)")
            else:
                opinion_parts.append("배당 데이터 부족")
        except Exception:
            opinion_parts.append("배당 판정 불가")

        if momentum_text and "강한 상승" in momentum_text:
            opinion_parts.append("단기 모멘텀 양호")
        elif momentum_text and "약세" in momentum_text:
            opinion_parts.append("단기 모멘텀 약함")

        score_notes = " / ".join(opinion_parts)

        # 7) compose message
        lines = [
            f"📊 종목 분석: {name} ({ticker})",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💵 현재가: {price_fmt} ({sign}{change_fmt})",
            f"🏦 시가총액: {market_cap_fmt}",
            f"📈 PER: {pe_display} | EPS: {eps_display} | PBR: {pbr_display}",
            f"{dividend_line}",
            f"🔢 거래량: {vol_fmt}",
            f"📅 52주 최고/최저: {high_52 if high_52 is not None else 'N/A'} / {low_52 if low_52 is not None else 'N/A'}",
            f"🏭 섹터: {sector} | 업종: {industry} | 국가: {country}",
            f"📊 모멘텀: {momentum_text}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💬 투자 의견 요약: {score_notes}",
            "📡 데이터 출처: Yahoo Finance / yfinance",
        ]

        return "\n".join(lines)


# quick test (only when running this module directly)
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    print(analyzer.get_stock_info("TSLA"))
    print()
    print(analyzer.get_stock_info("삼성전자"))
