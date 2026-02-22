"""
SmartStock AI Analyzer — Data Fetcher
Fetches OHLCV, fundamentals, and computes technicals via yfinance.
Returns DataAgentOutput (Pydantic). No Gemini calls.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.agents import (
    DataAgentOutput,
    Fundamentals,
    PriceSnapshot,
    Technicals,
    VolumeInfo,
)
from schemas.enums import Sector
from schemas.config import settings
from utils.logger import log_agent


_SECTOR_MAP: dict[str, Sector] = {
    "technology": Sector.TECHNOLOGY,
    "healthcare": Sector.HEALTHCARE,
    "financial services": Sector.FINANCE,
    "financial": Sector.FINANCE,
    "energy": Sector.ENERGY,
    "consumer cyclical": Sector.CONSUMER,
    "consumer defensive": Sector.CONSUMER,
    "industrials": Sector.INDUSTRIAL,
    "real estate": Sector.REAL_ESTATE,
    "utilities": Sector.UTILITIES,
    "basic materials": Sector.MATERIALS,
    "communication services": Sector.TELECOM,
}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    Returns empty DataFrame on failure.
    """
    log_agent("DataFetcher", f"OHLCV 데이터 수집 중: {ticker} (기간: {period})")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            log_agent("DataFetcher", f"[yellow]{ticker}: OHLCV 데이터 없음[/yellow]")
            return pd.DataFrame()
        df = df.reset_index()
        log_agent("DataFetcher", f"OHLCV 수집 완료: {len(df)} rows")
        return df
    except Exception as e:
        log_agent("DataFetcher", f"[red]OHLCV 수집 실패: {e}[/red]")
        return pd.DataFrame()


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data: PER, PBR, ROE, debt_ratio, revenue, operating_profit.
    Returns dict with metric keys. Missing values are None.
    """
    log_agent("DataFetcher", f"펀더멘탈 데이터 수집 중: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        fundamentals = {
            "market_cap": info.get("marketCap"),
            "per": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_ratio": info.get("debtToEquity"),
            "revenue": info.get("totalRevenue"),
            "operating_profit": info.get("operatingMargins"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margin": info.get("profitMargins"),
            "beta": info.get("beta"),
        }

        # Additional info for DataAgentOutput
        meta = {
            "company_name": info.get("shortName") or info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "exchange": info.get("exchange", ""),
            "currency": info.get("currency", "USD"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "open_price": info.get("regularMarketOpen", 0),
            "high_52w": info.get("fiftyTwoWeekHigh", 0),
            "low_52w": info.get("fiftyTwoWeekLow", 0),
            "prev_close": info.get("regularMarketPreviousClose", 0),
            "volume": info.get("regularMarketVolume") or info.get("volume", 0),
            "avg_volume_10d": info.get("averageDailyVolume10Day", 0),
            "avg_volume_3m": info.get("averageVolume", 0),
        }

        log_agent("DataFetcher", f"펀더멘탈 수집 완료: {ticker}")
        return {**fundamentals, **meta}

    except Exception as e:
        log_agent("DataFetcher", f"[red]펀더멘탈 수집 실패: {e}[/red]")
        return {
            "company_name": ticker,
            "sector": "",
            "exchange": "",
            "currency": "USD",
        }


def fetch_technicals(df: pd.DataFrame) -> dict:
    """
    Compute technical indicators from OHLCV DataFrame.
    Returns dict with: rsi_14, macd, macd_signal, macd_histogram,
    bb_upper, bb_middle, bb_lower, ma_20, ma_60, mdd, volatility.
    Returns empty dict if DataFrame is empty.
    """
    if df.empty or "Close" not in df.columns:
        log_agent("DataFetcher", "[yellow]기술적 지표 계산 불가: 데이터 없음[/yellow]")
        return {}

    log_agent("DataFetcher", "기술적 지표 계산 중...")
    close = df["Close"].astype(float)
    result: dict = {}

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    result["rsi_14"] = round(float(rsi.iloc[-1]), 2) if not rsi.empty and pd.notna(rsi.iloc[-1]) else None

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    result["macd"] = round(float(macd_line.iloc[-1]), 4) if pd.notna(macd_line.iloc[-1]) else None
    result["macd_signal"] = round(float(signal_line.iloc[-1]), 4) if pd.notna(signal_line.iloc[-1]) else None
    result["macd_histogram"] = round(float(histogram.iloc[-1]), 4) if pd.notna(histogram.iloc[-1]) else None

    # Bollinger Bands (20, 2)
    ma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    result["bb_upper"] = round(float((ma_20 + 2 * std_20).iloc[-1]), 2) if pd.notna(ma_20.iloc[-1]) else None
    result["bb_middle"] = round(float(ma_20.iloc[-1]), 2) if pd.notna(ma_20.iloc[-1]) else None
    result["bb_lower"] = round(float((ma_20 - 2 * std_20).iloc[-1]), 2) if pd.notna(ma_20.iloc[-1]) else None

    # Moving Averages
    result["ma_20"] = round(float(ma_20.iloc[-1]), 2) if pd.notna(ma_20.iloc[-1]) else None
    ma_60 = close.rolling(window=60).mean()
    result["ma_60"] = round(float(ma_60.iloc[-1]), 2) if len(close) >= 60 and pd.notna(ma_60.iloc[-1]) else None

    # MDD (Maximum Drawdown)
    cumulative_max = close.cummax()
    drawdown = (close - cumulative_max) / cumulative_max
    result["mdd"] = round(float(drawdown.min()) * 100, 2) if not drawdown.empty else None

    # Volatility (annualized, 20-day)
    daily_returns = close.pct_change().dropna()
    if len(daily_returns) >= 20:
        vol_20 = daily_returns.tail(20).std() * np.sqrt(252)
        result["volatility"] = round(float(vol_20) * 100, 2)
    else:
        result["volatility"] = None

    log_agent("DataFetcher", "기술적 지표 계산 완료")
    return result


def fetch_all(ticker: str, period: str = "1y") -> DataAgentOutput:
    """
    Full data pipeline: OHLCV → fundamentals → technicals → DataAgentOutput.
    Sets anomalies=["데이터 없음"] if OHLCV is empty.
    """
    anomalies: list[str] = []

    # 1) OHLCV
    df = fetch_ohlcv(ticker, period)

    # 2) Fundamentals
    fund_data = fetch_fundamentals(ticker)

    # 3) If OHLCV empty, flag anomaly and return partial
    if df.empty:
        anomalies.append("데이터 없음")
        return DataAgentOutput(
            ticker=ticker.upper(),
            company_name=fund_data.get("company_name", ticker),
            anomalies=anomalies,
            fetched_at=datetime.now(),
        )

    # 4) Save OHLCV CSV
    csv_path = settings.cache_dir / f"{ticker.upper()}_history.csv"
    df.to_csv(csv_path, index=False)

    # 5) Technicals
    tech_data = fetch_technicals(df)

    # 6) Build price snapshot
    curr = fund_data.get("current_price", 0) or 0
    prev = fund_data.get("prev_close", 0) or 0
    change_pct = round(((curr - prev) / prev) * 100, 2) if prev else 0.0

    price = PriceSnapshot(
        current=curr,
        open=fund_data.get("open_price", 0) or 0,
        high_52w=fund_data.get("high_52w", 0) or 0,
        low_52w=fund_data.get("low_52w", 0) or 0,
        change_pct=change_pct,
    )

    # 7) Volume
    cur_vol = fund_data.get("volume", 0) or 0
    avg_10 = fund_data.get("avg_volume_10d", 0) or 1
    avg_3m = fund_data.get("avg_volume_3m", 0) or 1
    volume = VolumeInfo(
        current=cur_vol,
        avg_10d=avg_10,
        avg_3m=avg_3m,
        relative_volume=round(cur_vol / avg_10, 2) if avg_10 else 1.0,
    )

    # 8) Fundamentals model
    fundamentals = Fundamentals(
        market_cap=fund_data.get("market_cap"),
        per=fund_data.get("per"),
        pbr=fund_data.get("pbr"),
        roe=fund_data.get("roe"),
        debt_ratio=fund_data.get("debt_ratio"),
        revenue=fund_data.get("revenue"),
        operating_profit=fund_data.get("operating_profit"),
        eps=fund_data.get("eps"),
        dividend_yield=fund_data.get("dividend_yield"),
        profit_margin=fund_data.get("profit_margin"),
        beta=fund_data.get("beta"),
    )

    # 9) Technicals model
    technicals = Technicals(
        rsi_14=tech_data.get("rsi_14"),
        macd=tech_data.get("macd"),
        macd_signal=tech_data.get("macd_signal"),
        macd_histogram=tech_data.get("macd_histogram"),
        bb_upper=tech_data.get("bb_upper"),
        bb_middle=tech_data.get("bb_middle"),
        bb_lower=tech_data.get("bb_lower"),
        ma_20=tech_data.get("ma_20"),
        ma_60=tech_data.get("ma_60"),
        mdd=tech_data.get("mdd"),
        volatility=tech_data.get("volatility"),
    )

    # 10) Sector mapping
    raw_sector = (fund_data.get("sector") or "").lower()
    sector = _SECTOR_MAP.get(raw_sector, Sector.OTHER)

    # 11) Data quality checks
    if fundamentals.per is None:
        anomalies.append("PER 데이터 없음")
    if fundamentals.roe is None:
        anomalies.append("ROE 데이터 없음")
    if technicals.ma_60 is None:
        anomalies.append("60일 이동평균 데이터 부족")

    return DataAgentOutput(
        ticker=ticker.upper(),
        company_name=fund_data.get("company_name", ticker),
        sector=sector,
        exchange=fund_data.get("exchange", ""),
        currency=fund_data.get("currency", "USD"),
        price=price,
        volume=volume,
        fundamentals=fundamentals,
        technicals=technicals,
        history_df_path=str(csv_path),
        anomalies=anomalies,
        fetched_at=datetime.now(),
    )
