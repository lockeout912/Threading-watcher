import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Threading Entry/Exit Signal Dashboard",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Threading Entry/Exit Signal Dashboard")
st.write(
    "This dashboard scans a 4/9 EMA threading system across your watchlist "
    "and flags **ENTRY / EXIT** signals. It also checks a simple "
    "Volume/Price Divergence (VPD) pattern for extra confirmation."
)

# --------------------------------------------------------------------
# WATCHLIST (EDIT ANYTIME)
# --------------------------------------------------------------------
watchlist = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "AMD", "AMZN", "META", "PLTR", "SOFI",
    "GME", "AMC", "RIOT", "MARA", "MSTR",
    "U", "COIN", "NFLX", "BABA", "DIS",
    "WBD", "BITO",  # BITO = Bitcoin ETF
]

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def fetch_price_data(ticker: str):
    """
    Try to pull intraday data with several fallbacks so we don't die
    when the market is closed or 1m data isn't available.
    """
    attempts = [
        ("2d", "1m"),   # very fresh, 1-minute candles
        ("5d", "5m"),   # fallback: 5-minute candles
        ("1mo", "1h"),  # last resort: 1-hour candles
    ]

    for period, interval in attempts:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
            )
            if not df.empty:
                return df
        except Exception:
            # just try the next combo
            continue

    return None


def detect_bullish_vpd(df: pd.DataFrame,
                       price_col: str = "Close",
                       volume_col: str = "Volume"):
    """
    Very simple Volume/Price Divergence (VPD) detector.

    Idea:
    - Compare the last N candles to the prior window.
    - If price is grinding up while volume ramps up sharply,
      we flag it as "ACCUMULATION (Bullish VPD)".
    - If price is grinding down while volume ramps up sharply,
      we flag it as "DISTRIBUTION (Bearish VPD)".
    """
    if df is None or df.empty:
        return None

    if price_col not in df.columns or volume_col not in df.columns:
        return None

    # Use the last 60 candles as "recent"
    recent = df.tail(60)
    if recent[volume_col].sum() == 0:
        return None

    # Reference window = last 200 candles (or all if smaller)
    ref = df.tail(200)

    # Price change over recent window
    start_price = recent[price_col].iloc[0]
    end_price = recent[price_col].iloc[-1]
    if start_price == 0:
        return None
    price_change_pct = (end_price - start_price) / start_price

    # Volume change: recent avg vs ref avg
    ref_vol_mean = ref[volume_col].mean()
    recent_vol_mean = recent[volume_col].mean()
    if ref_vol_mean == 0:
        return None
    vol_change_ratio = (recent_vol_mean - ref_vol_mean) / ref_vol_mean

    # Thresholds – you can tweak these
    MIN_PRICE_MOVE = 0.01     # 1%
    MIN_VOL_RAMP = 0.5        # +50% vs baseline

    if price_change_pct > MIN_PRICE_MOVE and vol_change_ratio > MIN_VOL_RAMP:
        return "ACCUMULATION (Bullish VPD)"
    elif price_change_pct < -MIN_PRICE_MOVE and vol_change_ratio > MIN_VOL_RAMP:
        return "DISTRIBUTION (Bearish VPD)"
    else:
        return None


def get_signals(ticker: str):
    """
    Pulls data, computes EMA4/EMA9, runs VPD detector,
    and returns a clean dict for the table.
    """
    try:
        df = fetch_price_data(ticker)
        if df is None or df.empty:
            return {
                "Ticker": ticker,
                "Price": None,
                "EMA4": None,
                "EMA9": None,
                "Signal": "No data",
            }

        # Keep it light
        df = df.tail(300)

        # EMAs
        df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

        # VPD label
        vpd_label = detect_bullish_vpd(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # -----------------------------
        # EMA-based signal
        # -----------------------------
        if latest["EMA4"] > latest["EMA9"] and prev["EMA4"] <= prev["EMA9"]:
            ema_signal = "EMA ENTRY (Bullish 4/9 cross up)"
        elif latest["EMA4"] < latest["EMA9"] and prev["EMA4"] >= prev["EMA9"]:
            ema_signal = "EMA EXIT (Bearish 4/9 cross down)"
        else:
            ema_signal = ""

        # -----------------------------
        # VPD-based signal
        # -----------------------------
        if vpd_label is not None:
            vpd_signal = f"VPD {vpd_label}"
        else:
            vpd_signal = ""

        # -----------------------------
        # Combine into final Signal text
        # -----------------------------
        parts = [p for p in (ema_signal, vpd_signal) if p]
        if parts:
            signal = " | ".join(parts)
        else:
            signal = "No clear signal"

        return {
            "Ticker": ticker,
            "Price": round(float(latest["Close"]), 2),
            "EMA4": round(float(latest["EMA4"]), 2),
            "EMA9": round(float(latest["EMA9"]), 2),
            "Signal": signal,
        }

    except Exception as e:
        # Fail gracefully so one bad ticker doesn't kill the app
        return {
            "Ticker": ticker,
            "Price": None,
            "EMA4": None,
            "EMA9": None,
            "Signal": f"Error: {str(e)}",
        }


# --------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------
st.markdown("### 📈 Scanning **{}** tickers...".format(len(watchlist)))

rows = []
for symbol in watchlist:
    row = get_signals(symbol)
    rows.append(row)

signals_df = pd.DataFrame(rows)

st.markdown("## 🔍 Signals")
st.dataframe(signals_df, use_container_width=True)

st.caption(
    "Last updated: {} UTC".format(
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    )
)
