import streamlit as st
import yfinance as yf
import pandas as pd

# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="Lockout Threading Signals", layout="wide")

st.title("📈 Lockout Trading Threading Dashboard")
st.write("Real-time threading signals based on your 4/9 EMA system.")

# -------------------------------
# WATCHLIST (edit anytime)
# -------------------------------
watchlist = [
    "AMC", "GME", "MARA", "RIOT", "SOFI",
    "PLTR", "NVDA", "TSLA", "SPY", "QQQ",
    "U", "AAPL", "AMZN", "META", "MSFT",
    "BITO", "COIN", "MSTR", "UBER", "NFLX"
]

st.sidebar.header("⚙️ Dashboard Settings")
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])

# -------------------------------
# FUNCTION: Pull data and compute EMAs
# -------------------------------
def get_signals(ticker):
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval=interval,
            progress=False
        )

        if df.empty:
            return None

        df["EMA4"] = df["Close"].ewm(span=4).mean()
        df["EMA9"] = df["Close"].ewm(span=9).mean()

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price = latest["Close"]

        # Determine threading signals
        signal = "NEUTRAL"
        color = "white"

        # Bullish thread (EMA4 crossing above EMA9)
        if prev["EMA4"] < prev["EMA9"] and latest["EMA4"] > latest["EMA9"]:
            signal = "🔥 BULLISH THREADING ENTRY"
            color = "green"

        # Bearish thread (EMA4 crossing below EMA9)
        elif prev["EMA4"] > prev["EMA9"] and latest["EMA4"] < latest["EMA9"]:
            signal = "🔻 BEARISH THREADING EXIT"
            color = "red"

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "EMA4": round(latest["EMA4"], 2),
            "EMA9": round(latest["EMA9"], 2),
            "Signal": signal,
            "Color": color
        }

    except:
        return None

# -------------------------------
# PROCESS WATCHLIST
# -------------------------------
results = []

for stock in watchlist:
    data = get_signals(stock)
    if data:
        results.append(data)

df = pd.DataFrame(results)

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
st.subheader(f"🔍 Signals for {len(watchlist)} stocks")

if df.empty:
    st.warning("No data available. Market may be closed or API limit reached.")
else:
    for i, row in df.iterrows():
        st.markdown(
            f"""
            <div style="padding:10px; margin-bottom:8px; border-radius:8px; background-color:#222;">
                <h3 style="color:{row['Color']};">{row['Ticker']} — {row['Signal']}</h3>
                <p style="color:white;">
                    📌 Price: <b>${row['Price']}</b><br>
                    📉 EMA4: {row['EMA4']} | EMA9: {row['EMA9']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
