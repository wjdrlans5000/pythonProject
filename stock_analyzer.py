# stock_analyzer.py
import requests
import yfinance as yf
from yahooquery import search as yq_search
import pandas as pd
from typing import Optional
from urllib.parse import quote
import json

class StockAnalyzer:
    def __init__(self):
        pass

    def format_market_cap(self, cap: Optional[float]) -> str:
        """시가총액을 읽기 쉬운 문자열로 변환 (C 스타일: 억/조 단위)."""
        if not cap or cap <= 0:
            return "N/A"
        trillion = 1_000_000_000_000
        billion = 1_000_000_000
        million = 1_000_000
        if cap >= trillion:
            # 조 단위 (trillion)
            return f"${cap / trillion:.1f}조 달러"
        elif cap >= billion:
            # 억 단위: 1 billion = 10억 -> 하지만 요청하신 표현은 억 달러 단위 (1e8)
            # 여기서는 '억 달러'로 보여주기 위해 1e8으로 나눔 (사용자 원문 스타일 유지)
            return f"${cap / 1e8:,.1f}억 달러"
        elif cap >= million:
            return f"${cap / million:.1f}백만 달러"
        else:
            return f"${cap:,.0f}"

    # -------------------------
    # 티커 변환: 네이버 우선 -> yahooquery 백업
    # -------------------------
    def get_ticker_by_name(self, query: str) -> Optional[str]:
        """
        종목명 또는 심볼 입력 시 자동으로 Yahoo Finance 형태의 티커 변환
        - 네이버 금융 우선(한국 종목), 실패 시 yahooquery로 시도
        - 반환 예: '005930.KS', 'TSLA', 'AAPL'
        """
        if not query:
            return None

        q = query.strip()

        # 이미 Yahoo 형식(대문자 심볼 또는 .KS/.KQ 포함)인 경우 바로 리턴
        if q.endswith((".KS", ".KQ")):
            return q.upper()

        # 1) 네이버 금융 검색 시도 (한국 종목 우선)
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
            }
            search_url = f"https://ac.stock.naver.com/ac?q={quote(q)}&target=stock"
            res = requests.get(search_url, headers=headers, timeout=6)
            if res.status_code != 200:
                print(f"[네이버 검색 실패] 상태코드: {res.status_code}")
            else:
                data = json.loads(res.text)
                query = data.get("query", "").strip()
                items = data.get("items", [])

                if not items:
                    return None  # 종목 없음

                # 1. query와 name이 정확히 일치하는 항목 찾기
                exact_match = next((item for item in items if item.get("name") == query), None)

                # 2. 없다면 첫 번째 item 사용
                target = exact_match if exact_match else items[0]

                code = target.get("code")
                type_code = target.get("typeCode")  # KOSPI / KOSDAQ 구분 가능
                nationName = target.get("nationName")  # 한국/미국 구분
                if nationName == "미국" :
                    return None

                # Yahoo Finance 형식 추가 (.KS / .KQ)
                market = ".KS" if type_code == "KOSPI" else ".KQ"
                return f"{code}{market}"
        except Exception as e:
            # 네이버 검색 실패하면 다음 단계로
            print(f"[네이버 검색 오류] {e}")
            pass

        # 2) yahooquery 검색(영문/국제 종목 매칭) - backup
        try:
            res = yq_search(q)
            # 결과 형식: dict with 'quotes' list
            if isinstance(res, dict) and "quotes" in res and len(res["quotes"]) > 0:
                sym = res["quotes"][0].get("symbol")
                if sym:
                    return sym.upper()
        except Exception:
            pass

        # 3) 마지막으로 입력값을 심볼로 시도해보기 (yfinance 유효성 체크)
        try:
            t = yf.Ticker(q.upper())
            info = getattr(t, "fast_info", None) or getattr(t, "info", None)
            if info:
                return q.upper()
        except Exception:
            pass

        return None

    # -------------------------
    # 메인: 기본정보 + 10년 평균 배당률 + 간단 분석 리턴
    # -------------------------
    def get_stock_info(self, query: str) -> str:
        """
        query: 티커(예: 'AAPL' 또는 '005930.KS') 또는 회사명(예: '삼성전자')
        반환: 텔레그램 전송용 포맷된 텍스트
        """
        if not query or not isinstance(query, str):
            return "❌ 잘못된 입력입니다. /s 뒤에 종목을 입력하세요."

        # 1) 티커 확인/변환
        ticker = query.strip()
        # 만약 한글/혼합 입력일 가능성 있으면 변환 시도
        # if not (ticker.endswith((".KS", ".KQ")) or ticker.isupper()):
        resolved = self.get_ticker_by_name(ticker)
        if  resolved:
            ticker = resolved

        # 2) 데이터 로드 (yfinance)
        try:
            stock = yf.Ticker(ticker)
            info = getattr(stock, "info", {}) or {}
        except Exception as e:
            return f"❌ 데이터 로드 실패: {e}"

        # 필수 확인
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            return f"⚠️ '{ticker}' 종목의 가격 정보를 불러올 수 없습니다."

        # 3) 주요 지표 추출 (안전하게 가져오기)
        name = info.get("longName") or info.get("shortName") or ticker
        change_pct = info.get("regularMarketChangePercent")  # 소수점(%) 형태
        market_cap = info.get("marketCap") or 0
        pe_ratio = info.get("trailingPE")
        eps = info.get("trailingEps")
        pbr = info.get("priceToBook")
        div_yield = info.get("dividendYield") or 0.0  # 소수(예: 0.02)
        vol = info.get("volume") or None
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        sector = info.get("sector") or "N/A"
        industry = info.get("industry") or "N/A"
        country = info.get("country") or "N/A"

        # 4) 시가총액/배당(10년평균) 계산
        market_cap_str = self.format_market_cap(market_cap)

        # 10년 평균 배당률 계산 (존재하면 %로 반환)
        avg_div_yield = "N/A"
        try:
            div_hist = getattr(stock, "dividends", None)
            if isinstance(div_hist, (pd.Series, pd.DataFrame)) and not div_hist.empty:
                # 연간 배당 합계 & 연평균 수익률 계산
                div_hist.index = pd.to_datetime(div_hist.index)
                yearly = div_hist.groupby(div_hist.index.year).sum()
                # 동일 기간의 평균 주가(연평균) 구하기
                price_hist = stock.history(period="10y")["Close"]
                if not price_hist.empty:
                    price_yearly = price_hist.resample("YE").mean()
                    merged = pd.concat([yearly, price_yearly], axis=1).dropna()
                    merged.columns = ["Dividend", "Price"]
                    if not merged.empty and (merged["Price"] > 0).any():
                        merged["Yield"] = merged["Dividend"] / merged["Price"]
                        avg = merged["Yield"].mean()
                        if pd.notna(avg):
                            avg_div_yield = round(float(avg) * 100, 2)
        except Exception:
            avg_div_yield = "N/A"

        # 5) 기술적(모멘텀) 간단 계산: 6개월 퍼포먼스
        momentum_text = "N/A"
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

        # 6) 투자의견 (간단한 rules - 적극형 B 스타일)
        opinion_parts = []
        # PER 기반
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

        # 배당 기반
        try:
            if avg_div_yield != "N/A":
                if avg_div_yield > 4:
                    opinion_parts.append("배당 매력 높음")
                elif avg_div_yield > 1:
                    opinion_parts.append("보통 배당")
                else:
                    opinion_parts.append("배당 낮음")
            else:
                opinion_parts.append("배당 데이터 부족")
        except Exception:
            opinion_parts.append("배당 판정 불가")

        # 모멘텀 기반
        if momentum_text and "강한 상승" in momentum_text:
            opinion_parts.append("단기 모멘텀 양호")
        elif momentum_text and "약세" in momentum_text:
            opinion_parts.append("단기 모멘텀 약함")

        # 종합 간단 등급 도출 (점수 대신 등급)
        score_notes = " / ".join(opinion_parts)

        # 7) 메시지 포맷팅 (텔레그램 전송용)
        sign = "▲" if (change_pct and change_pct > 0) else "▼" if (change_pct and change_pct < 0) else "-"
        change_pct_display = f"{abs(change_pct):.2f}" if change_pct is not None else "N/A"

        # 볼륨 표기
        vol_str = f"{vol:,}" if vol else "N/A"

        pe_display = f"{pe_ratio:.2f}" if (pe_ratio is not None and pd.notna(pe_ratio)) else "N/A"
        eps_display = f"{eps:.2f}" if (eps is not None and pd.notna(eps)) else "N/A"
        pbr_display = f"{pbr:.2f}" if (pbr is not None and pd.notna(pbr)) else "N/A"
        div_display = f"{div_yield * 100:.2f}%" if div_yield else "0.00%"

        avg_div_display = f"{avg_div_yield}%" if avg_div_yield != "N/A" else "N/A"

        msg_lines = [
            f"📊 종목 분석: {name} ({ticker})",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💵 현재가: ${price:,.2f} ({sign}{change_pct_display}%)",
            f"🏦 시가총액: {market_cap_str}",
            f"📈 PER: {pe_display} | EPS: {eps_display} | PBR: {pbr_display}",
            f"💸 배당률(최근): {div_display} | 10년 평균: {avg_div_display}",
            f"🔢 거래량: {vol_str}",
            f"📅 52주 최고/최저: {high_52} / {low_52}",
            f"🏭 섹터: {sector} | 업종: {industry} | 국가: {country}",
            f"📊 모멘텀: {momentum_text}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💬 투자 의견 요약: {score_notes}",
            "📡 데이터 출처: Yahoo Finance"
        ]

        return "\n".join(msg_lines)


# Usage example:
analyzer = StockAnalyzer()
print(analyzer.get_stock_info("TSLA"))
print(analyzer.get_stock_info("NAVER"))
