import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Threading Entry/Exit Watcher", layout="wide")

# -----------------------------------------------
# YOUR WATCHLIST (EDIT THIS ANYTIME)
# -----------------------------------------------
watchlist = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "AMD", "AMZN", "META", "PLTR", "SOFI",
    "GME", "AMC", "RIOT", "MARA", "MSTR",
    "U", "COIN", "NFLX", "BABA", "DIS"
]

# -----------------------------------------------
# GET THREADING SIGNALS (4/9 EMA)
# -----------------------------------------------
def get_signals(ticker):
    try:
        # Try 1: 1-minute intraday, last 2 days
    df = yf.Ticker(ticker).history(period="2d", interval="1m")

# If market is closed or no 1m data, fall back to 5m candles
if df.empty:
    df = yf.Ticker(ticker).history(period="5d", interval="5m")

# If still no data, fall back to 1h candles over 1 month
if df.empty:
    df = yf.download(ticker, period="1mo", interval="1h")

# If still empty, give up on this ticker
if df.empty:
    return None

        df["EMA4"] = df["Close"].ewm(span=4).mean()
        df["EMA9"] = df["Close"].ewm(span=9).mean()

        last = df.iloc[-1]

        price = last["Close"]
        ema4 = last["EMA4"]
        ema9 = last["EMA9"]

        if ema4 > ema9:
            signal = "📈 Bullish Threading — Consider Long Entry"
            color = "#1db954"
        elif ema4 < ema9:
            signal = "📉 Bearish Threading — Consider Exit / Put Flow"
            color = "#ff4c4c"
        else:
            signal = "⚪ Neutral — No Clear Setup"
            color = "#888888"

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "EMA4": round(ema4, 2),
            "EMA9": round(ema9, 2),
            "Signal": signal,
            "Color": color
        }

    except:
        return None


# -----------------------------------------------
# BUILD DASHBOARD UI
# -----------------------------------------------
st.title("🔥 Threading Entry/Exit Signal Dashboard")

st.markdown("""
This dashboard uses **real-time EMA 4/9 threading**, matching your day-trading style.
""")

results = []

for stock in watchlist:
    data = get_signals(stock)
    if data:
        results.append(data)

df = pd.DataFrame(results)

# -----------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------
st.subheader(f"🔍 Signals for {len(watchlist)} Stocks")

if df.empty:
    st.warning("No data available.")
else:
    for i, row in df.iterrows():
        st.markdown(
            f"""
            <div style="padding:12px; margin:8px 0; border-radius:10px; background:{row['Color']}33;">
                <h3 style="color:{row['Color']};">{row['Ticker']}</h3>
                <p style="color:white;">
                    📌 Price: <b>${row['Price']}</b><br>
                    🟦 EMA4: {row['EMA4']}<br>
                    🟪 EMA9: {row['EMA9']}<br><br>
                    <b>{row['Signal']}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
