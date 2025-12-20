import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
    "volume-pressure divergence (VPD) check to flag signals with "
    "**Direction + Strength + Levels** (Trigger / Target Zone / Invalidation)."
)

# --------------------------------------------------
# WATCHLIST (EDIT ANYTIME)
# --------------------------------------------------
watchlist = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "AMD", "AMZN", "META", "PLTR", "SOFI",
    "GME", "AMC", "RIOT", "MARA", "MSTR",
    "COIN", "U", "WBD", "BITO",
    # Crypto via Yahoo:
    "BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"
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
        t = yf.Ticker(ticker)

        # 1-minute for last 2 days
        df = t.history(period="2d", interval="1m")
        if df is not None and not df.empty:
            return df

        # 5-minute for last 5 days
        df = t.history(period="5d", interval="5m")
        if df is not None and not df.empty:
            return df

        # 1-hour for last month
        df = t.history(period="1mo", interval="1h")
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------
# INDICATOR CALCS
# --------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA4, EMA9, returns, and VPD score inputs.
    Expects columns: 'Close', 'Volume' (and ideally 'High','Low').
    """
    df = df.copy()

    # EMAs
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

    # Returns & simple VPD driver: return * volume
    df["ret"] = df["Close"].pct_change()
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

def get_vpd_signal(df: pd.DataFrame, lookback: int = 20):
    """
    Simple VPD (volume-pressure divergence style) signal.
    vp = return * volume, summed across lookback.
    Returns (vpd_score, signal_text).
    """
    if df.shape[0] < lookback + 5:
        return 0.0, "No clear VPD signal"

    recent = df.tail(lookback).copy()

    vpd_score = float(recent["vp"].sum())
    last_ret = float(recent["ret"].iloc[-1])

    if vpd_score > 0 and last_ret > 0:
        vpd_signal = "Bullish VPD (volume backing upside)"
    elif vpd_score < 0 and last_ret < 0:
        vpd_signal = "Bearish VPD (volume backing downside)"
    else:
        vpd_signal = "No clear VPD signal"

    return vpd_score, vpd_signal

# --------------------------------------------------
# ALERT LOGIC (Direction / Strength / Levels)
# --------------------------------------------------
def infer_direction(ema_signal: str, vpd_signal: str) -> str:
    """
    Direction priority:
    - Confluence if available
    - EMA if it has a directional read
    - VPD if it has a directional read
    - else Neutral
    """
    if ema_signal.startswith("ENTRY") and vpd_signal.startswith("Bullish"):
        return "Bullish"
    if ema_signal.startswith("EXIT") and vpd_signal.startswith("Bearish"):
        return "Bearish"

    if ema_signal.startswith("ENTRY"):
        return "Bullish"
    if ema_signal.startswith("EXIT"):
        return "Bearish"

    if vpd_signal.startswith("Bullish"):
        return "Bullish"
    if vpd_signal.startswith("Bearish"):
        return "Bearish"

    return "Neutral"

def map_strength(ema_signal: str, vpd_signal: str):
    """
    Converts confluence into a strength tier + numeric score.
    """
    ema_entry = ema_signal.startswith("ENTRY")
    ema_exit  = ema_signal.startswith("EXIT")
    vpd_bull  = vpd_signal.startswith("Bullish")
    vpd_bear  = vpd_signal.startswith("Bearish")

    score = 0
    # EMA cross is a major event
    if ema_entry or ema_exit:
        score += 55
    # VPD confirmation adds strength
    if vpd_bull or vpd_bear:
        score += 35
    # Directional alignment bonus
    if (ema_entry and vpd_bull) or (ema_exit and vpd_bear):
        score += 10

    if score >= 80:
        return "Extreme", score
    if score >= 60:
        return "High", score
    if score >= 40:
        return "Medium", score
    return "Low", score

def calc_atr_like(df: pd.DataFrame, window: int = 14) -> float:
    """
    Lightweight ATR-ish measure using High/Low/Close.
    Falls back safely if High/Low not available.
    """
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return float("nan")
    if df.shape[0] < window + 2:
        return float("nan")

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr)

def compute_levels(df: pd.DataFrame, direction: str) -> dict:
    """
    Produces trigger / target zone / invalidation based on EMA9 + ATR-ish range.
    This is intentionally conservative + stable for prototyping.
    """
    last = df.iloc[-1]
    current = float(last["Close"])
    ema9 = float(last["EMA9"])

    atr = calc_atr_like(df, window=14)
    if np.isnan(atr) or atr == 0:
        # fallback: 0.2% of price (or 1 cent)
        atr = max(current * 0.002, 0.01)

    if direction == "Bullish":
        trigger = max(current, ema9)
        target_low = trigger + 0.8 * atr
        target_high = trigger + 2.0 * atr
        invalidation = trigger - 1.0 * atr

    elif direction == "Bearish":
        trigger = min(current, ema9)
        target_low = trigger - 2.0 * atr
        target_high = trigger - 0.8 * atr
        invalidation = trigger + 1.0 * atr

    else:
        trigger = current
        target_low = current
        target_high = current
        invalidation = current

    lo = float(min(target_low, target_high))
    hi = float(max(target_low, target_high))

    return {
        "current": current,
        "trigger": float(trigger),
        "target_low": lo,
        "target_high": hi,
        "invalidation": float(invalidation),
        "atr": float(atr),
    }

def move_metrics(current: float, target_low: float, target_high: float) -> dict:
    """
    Returns % and $ move ranges from current to target zone.
    """
    low_usd = target_low - current
    high_usd = target_high - current
    low_pct = (low_usd / current) * 100 if current else 0
    high_pct = (high_usd / current) * 100 if current else 0

    return {
        "move_usd_low": float(low_usd),
        "move_usd_high": float(high_usd),
        "move_pct_low": float(low_pct),
        "move_pct_high": float(high_pct),
    }

def is_crypto_yahoo(ticker: str) -> bool:
    return ticker.endswith("-USD")

# --------------------------------------------------
# PER-TICKER SCAN
# --------------------------------------------------
def scan_ticker(ticker: str) -> dict:
    df_raw = fetch_history(ticker)

    if df_raw is None or df_raw.empty:
        return {
            "Ticker": ticker,
            "Direction": "Neutral",
            "Strength": "Low",
            "Score": 0,
            "Price": None,
            "Trigger": None,
            "Target Low": None,
            "Target High": None,
            "Move % (low)": None,
            "Move % (high)": None,
            "Move $ (low)": None,
            "Move $ (high)": None,
            "Invalidation": None,
            "EMA4": None,
            "EMA9": None,
            "VPD Score": None,
            "Confirmations": "No data",
            "Final Signal": "No data",
        }

    df = compute_indicators(df_raw)

    # Latest values
    last_close = float(df["Close"].iloc[-1])
    last_ema4 = float(df["EMA4"].iloc[-1])
    last_ema9 = float(df["EMA9"].iloc[-1])

    ema_signal = get_ema_signal(df)
    vpd_score, vpd_signal = get_vpd_signal(df)

    # Combine signals (your original confluence logic preserved)
    final_signal = "No clear signal"
    source = ""

    ema_is_entry = ema_signal.startswith("ENTRY")
    ema_is_exit = ema_signal.startswith("EXIT")
    vpd_bull = vpd_signal.startswith("Bullish")
    vpd_bear = vpd_signal.startswith("Bearish")

    if ema_is_entry and vpd_bull:
        final_signal = "🚀 STRONG ENTRY (EMA + VPD)"
        source = "EMA + VPD"
    elif ema_is_exit and vpd_bear:
        final_signal = "⚠ STRONG EXIT (EMA + VPD)"
        source = "EMA + VPD"
    else:
        if ema_signal != "No clear EMA signal":
            final_signal = ema_signal
            source = "EMA"
        elif vpd_signal != "No clear VPD signal":
            final_signal = vpd_signal
            source = "VPD"
        else:
            final_signal = "No clear signal"
            source = "-"

    labeled_final = f"{final_signal}  [{source}]"

    # NEW: direction / strength / levels
    direction = infer_direction(ema_signal, vpd_signal)
    strength, score = map_strength(ema_signal, vpd_signal)
    levels = compute_levels(df, direction)
    moves = move_metrics(levels["current"], levels["target_low"], levels["target_high"])

    confirmations = []
    if ema_signal != "No clear EMA signal":
        confirmations.append(ema_signal)
    if vpd_signal != "No clear VPD signal":
        confirmations.append(vpd_signal)

    # formatting precision
    prec = 4 if is_crypto_yahoo(ticker) else 2

    return {
        "Ticker": ticker,
        "Direction": direction,
        "Strength": strength,
        "Score": score,
        "Price": round(last_close, prec),
        "Trigger": round(levels["trigger"], prec),
        "Target Low": round(levels["target_low"], prec),
        "Target High": round(levels["target_high"], prec),
        "Move % (low)": round(moves["move_pct_low"], 2),
        "Move % (high)": round(moves["move_pct_high"], 2),
        "Move $ (low)": round(moves["move_usd_low"], prec),
        "Move $ (high)": round(moves["move_usd_high"], prec),
        "Invalidation": round(levels["invalidation"], prec),
        "EMA4": round(last_ema4, prec),
        "EMA9": round(last_ema9, prec),
        "VPD Score": round(vpd_score, 2),
        "Confirmations": " | ".join(confirmations) if confirmations else "-",
        "Final Signal": labeled_final,
    }

# --------------------------------------------------
# MAIN DASHBOARD RENDER
# --------------------------------------------------
st.subheader(f"🔍 Scanning {len(watchlist)} tickers...")

rows = []
for t in watchlist:
    rows.append(scan_ticker(t))

signals_df = pd.DataFrame(rows)

# --------------------------------------------------
# TOP ALERT CARDS (fast digestion)
# --------------------------------------------------
st.subheader("🚨 Top Alerts (Highest Score First)")

top_df = signals_df.copy()
top_df = top_df[top_df["Direction"].isin(["Bullish", "Bearish"])]
top_df = top_df.sort_values(["Score"], ascending=False).head(8)

if top_df.empty:
    st.info("No strong directional alerts right now. Market might be chopping or signals are neutral.")
else:
    for _, r in top_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {r['Ticker']} — **{r['Direction']} | {r['Strength']}** (Score: {int(r['Score'])})")
            st.write(f"**Current:** {r['Price']}   |   **Trigger:** {r['Trigger']}")
            st.write(
                f"**Target Zone:** {r['Target Low']}–{r['Target High']}   |   "
                f"**Move:** {r['Move % (low)']:+.2f}% to {r['Move % (high)']:+.2f}%  /  "
                f"{r['Move $ (low)']:+} to {r['Move $ (high)']:+}"
            )
            st.write(f"**Invalidation:** {r['Invalidation']}")
            st.write(f"**Why:** {r['Confirmations']}")

st.divider()

# --------------------------------------------------
# FULL TABLE
# --------------------------------------------------
st.subheader("📊 Full Scan Table")

st.dataframe(
    signals_df,
    use_container_width=True,
)

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
