import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🔥 Threading Entry/Exit Signal Dashboard",
    layout="wide",
)

# -----------------------------
# WATCHLIST (EDIT ANYTIME)
# -----------------------------
WATCHLIST = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "AMD", "AMZN", "BTC", "PLTR", "SOFI",
    "GME", "AMC", "RIOT", "MARA", "MSTR",
    "U", "COIN", "WBD", "BABA", "AA",
]

# -----------------------------
# DATA FETCHING
# -----------------------------
def fetch_price_history(ticker: str) -> pd.DataFrame | None:
    """
    Try multiple timeframes / intervals until we get data.
    Returns a DataFrame or None if all attempts fail.
    """
    try:
        # 1) 1-minute candles, last 2 days
        df = yf.Ticker(ticker).history(period="2d", interval="1m")
        if not df.empty:
            return df

        # 2) 5-minute candles, last 5 days
        df = yf.Ticker(ticker).history(period="5d", interval="5m")
        if not df.empty:
            return df

        # 3) 1-hour candles, last month
        df = yf.download(ticker, period="1mo", interval="1h")
        if not df.empty:
            return df

        # If still nothing, give up on this ticker
        return None

    except Exception:
        # If anything blows up (bad ticker, connection, etc.)
        return None

# -----------------------------
# SIGNAL LOGIC (4 / 9 EMA THREADING)
# -----------------------------
def compute_threading_signal(ticker: str) -> dict | None:
    df = fetch_price_history(ticker)
    if df is None or df.empty:
        return None

    # Make sure we have a Close column
    if "Close" not in df.columns:
        return None

    # EMAs
    df["EMA4"] = df["Close"].ewm(span=4).mean()
    df["EMA9"] = df["Close"].ewm(span=9).mean()

    # Need at least 2 candles to compare previous vs latest
    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "No clear signal"

    # Bullish threading: EMA4 crosses UP through EMA9
    if prev["EMA4"] < prev["EMA9"] and latest["EMA4"] > latest["EMA9"]:
        signal = "ENTRY (Bullish Threading)"

    # Bearish threading: EMA4 crosses DOWN through EMA9
    elif prev["EMA4"] > prev["EMA9"] and latest["EMA4"] < latest["EMA9"]:
        signal = "EXIT (Bearish Threading)"

    return {
        "Ticker": ticker,
        "Price": round(float(latest["Close"]), 2),
        "EMA4": round(float(latest["EMA4"]), 2),
        "EMA9": round(float(latest["EMA9"]), 2),
        "Signal": signal,
    }

# -----------------------------
# UI
# -----------------------------
st.title("🔥 Threading Entry/Exit Signal Dashboard")
st.write(
    "This dashboard scans a 4/9 EMA threading system across "
    "your watchlist and flags **ENTRY** / **EXIT** signals."
)

st.write(f"📈 Scanning **{len(WATCHLIST)}** tickers...")

results: list[dict] = []

with st.spinner("Pulling data from Yahoo Finance..."):
    for symbol in WATCHLIST:
        data = compute_threading_signal(symbol)
        if data is not None:
            results.append(data)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
if not results:
    st.warning("No data available. Market might be closed or data could not be fetched.")
else:
    df_results = pd.DataFrame(results)

    # Nice, compact table
    st.subheader("🔎 Signals")
    st.dataframe(
        df_results.sort_values("Ticker").reset_index(drop=True),
        use_container_width=True,
    )

st.caption(
    f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
)
