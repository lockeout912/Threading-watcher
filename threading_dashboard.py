import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🔥 Threading + VPD Signal Dashboard",
    layout="wide"
)

st.title("🔥 Threading + VPD Signal Dashboard")
st.write(
    "This dashboard scans a 4/9 EMA threading system **plus** a "
    "volume-pressure divergence (VPD) check to flag ENTRY / EXIT signals."
)

# --------------------------------------------------
# YOUR WATCHLIST (EDIT THIS ANYTIME)
# --------------------------------------------------
watchlist = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "AMD", "AMZN", "META", "PLTR", "SOFI",
    "GME", "AMC", "RIOT", "MARA", "MSTR",
    "COIN", "U", "WBD", "BITO", "BTC-USD"
]


# --------------------------------------------------
# DATA FETCHING
# --------------------------------------------------
@st.cache_data(ttl=60)
def fetch_history(ticker: str) -> pd.DataFrame:
    """
    Try to get recent intraday data with fallbacks.
    Always returns a DataFrame (possibly empty), never raises.
    """
    try:
        # First try: 1-minute for last 2 days
        df = yf.Ticker(ticker).history(period="2d", interval="1m")
        if df is not None and not df.empty:
            return df

        # Fallback: 5-minute for last 5 days
        df = yf.Ticker(ticker).history(period="5d", interval="5m")
        if df is not None and not df.empty:
            return df

        # Fallback: 1-hour for last month
        df = yf.Ticker(ticker).history(period="1mo", interval="1h")
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()
    except Exception:
        # Any error, just return empty
        return pd.DataFrame()


# --------------------------------------------------
# INDICATOR CALCS
# --------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA4, EMA9, returns, and VPD score inputs.
    Expects columns: 'Close', 'Volume'.
    """
    df = df.copy()

    # EMAs
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

    # Returns & basic VPD-style signal driver
    df["ret"] = df["Close"].pct_change()
    # price * volume direction, gives weight to big-volume moves
    df["vp"] = df["ret"].fillna(0) * df["Volume"].fillna(0)

    return df


def get_ema_signal(df: pd.DataFrame) -> str:
    """
    4/9 EMA threading signal based on last two candles.
    """
    if df.shape[0] < 3:
        return "No clear EMA signal"

    ema4_now = float(df["EMA4"].iloc[-1])
    ema9_now = float(df["EMA9"].iloc[-1])
    ema4_prev = float(df["EMA4"].iloc[-2])
    ema9_prev = float(df["EMA9"].iloc[-2])

    # Cross up = bullish entry
    if ema4_now > ema9_now and ema4_prev <= ema9_prev:
        return "ENTRY (Bullish EMA cross)"

    # Cross down = bearish exit
    if ema4_now < ema9_now and ema4_prev >= ema9_prev:
        return "EXIT (Bearish EMA cross)"

    return "No clear EMA signal"


def get_vpd_signal(df: pd.DataFrame, lookback: int = 20) -> tuple[float, str]:
    """
    Very simple VPD (volume-pressure divergence style) signal.
    Uses last N bars of vp = return * volume.
    Returns (vpd_score, signal_text).
    """
    if df.shape[0] < lookback + 5:
        return 0.0, "No clear VPD signal"

    recent = df.tail(lookback).copy()

    # Sum of vp over lookback
    vpd_score = float(recent["vp"].sum())

    last_ret = float(recent["ret"].iloc[-1])

    # Basic interpretation
    if vpd_score > 0 and last_ret > 0:
        vpd_signal = "Bullish VPD (volume backing upside)"
    elif vpd_score < 0 and last_ret < 0:
        vpd_signal = "Bearish VPD (volume backing downside)"
    else:
        vpd_signal = "No clear VPD signal"

    return vpd_score, vpd_signal


# --------------------------------------------------
# PER-TICKER SCAN
# --------------------------------------------------
def scan_ticker(ticker: str) -> dict:
    """
    Pull data for one ticker, compute indicators, and
    return a dict used as one row in the table.
    """
    df_raw = fetch_history(ticker)

    if df_raw is None or df_raw.empty:
        return {
            "Ticker": ticker,
            "Price": None,
            "EMA4": None,
            "EMA9": None,
            "VPD Score": None,
            "EMA Signal": "No data",
            "VPD Signal": "No data",
            "Final Signal": "No data"
        }

    df = compute_indicators(df_raw)

    # Latest values
    last_close = float(df["Close"].iloc[-1])
    last_ema4 = float(df["EMA4"].iloc[-1])
    last_ema9 = float(df["EMA9"].iloc[-1])

    ema_signal = get_ema_signal(df)
    vpd_score, vpd_signal = get_vpd_signal(df)

    # --------------------------------------------------
    # COMBINE SIGNALS
    # --------------------------------------------------
    final_signal = "No clear signal"
    source = ""

    ema_is_entry = ema_signal.startswith("ENTRY")
    ema_is_exit = ema_signal.startswith("EXIT")
    vpd_bull = vpd_signal.startswith("Bullish")
    vpd_bear = vpd_signal.startswith("Bearish")

    # Strong confluence first
    if ema_is_entry and vpd_bull:
        final_signal = "🚀 STRONG ENTRY (EMA + VPD)"
        source = "EMA + VPD"
    elif ema_is_exit and vpd_bear:
        final_signal = "⚠ STRONG EXIT (EMA + VPD)"
        source = "EMA + VPD"
    else:
        # Fall back to whichever is giving a directional read
        if ema_signal != "No clear EMA signal":
            final_signal = ema_signal
            source = "EMA"
        elif vpd_signal != "No clear VPD signal":
            final_signal = vpd_signal
            source = "VPD"
        else:
            final_signal = "No clear signal"
            source = "-"

    # Append source tag so you can see at a glance
    labeled_final = f"{final_signal}  [{source}]"

    return {
        "Ticker": ticker,
        "Price": round(last_close, 2),
        "EMA4": round(last_ema4, 2),
        "EMA9": round(last_ema9, 2),
        "VPD Score": round(vpd_score, 2),
        "EMA Signal": ema_signal,
        "VPD Signal": vpd_signal,
        "Final Signal": labeled_final,
    }


# --------------------------------------------------
# MAIN DASHBOARD RENDER
# --------------------------------------------------
st.subheader(f"🔍 Scanning {len(watchlist)} tickers...")

rows = []
for t in watchlist:
    row = scan_ticker(t)
    rows.append(row)

signals_df = pd.DataFrame(rows)

st.dataframe(
    signals_df,
    use_container_width=True,
)

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
