# ============================================================
# LOCKOUT SIGNALS — SPY HIGH-SPEED COMMAND CENTER v2
# Trend-First • Aggressive • Trader-Readable
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Lockout Signals • SPY Command Center",
    layout="wide"
)

SYMBOL = "SPY"

# ============================================================
# STYLES — CLEAN WALL STREET HUD
# ============================================================
st.markdown("""
<style>
body { background-color:#0b0f14; overflow-x:hidden; }
.block-container { padding-top:6.5rem; max-width:1200px; }

#command {
  position:fixed; top:0; left:0; width:100%; height:60px;
  background:#05080d; border-bottom:2px solid #1f2933;
  z-index:9999; overflow:hidden;
}

.marquee {
  display:flex; white-space:nowrap;
  animation:scroll 16s linear infinite;
  font-size:1.1rem; font-weight:900;
  padding-left:100%;
}
.marquee span { margin-right:3rem; }

@keyframes scroll {
  0% { transform:translateX(0); }
  100% { transform:translateX(-100%); }
}

.green { color:#00ff9c; }
.red { color:#ff4d4d; }
.gray { color:#cfcfcf; }

.center { text-align:center; margin-top:28px; }
.symbol { font-size:1.6rem; opacity:0.8; }
.price { font-size:5.6rem; font-weight:900; letter-spacing:-2px; }
.action { font-size:2.2rem; font-weight:900; margin-top:6px; }
.range { font-size:1.3rem; font-weight:800; margin-top:16px; }
.instruction { font-size:1.15rem; margin-top:14px; }
.why { font-size:0.95rem; opacity:0.6; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA FETCH (NEAR-LIVE)
# ============================================================
@st.cache_data(ttl=10)
def fetch():
    df = yf.Ticker(SYMBOL).history(period="5d", interval="1m")
    return df.dropna()

df = fetch()
if df.empty or len(df) < 60:
    st.stop()

# ============================================================
# INDICATORS — CORE ENGINE
# ============================================================
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA_SLOPE"] = df["EMA9"] - df["EMA9"].shift(3)

tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

mid = df["Close"].rolling(20).mean()
std = df["Close"].rolling(20).std()
df["BBW"] = ((mid + 2*std) - (mid - 2*std)) / mid
df["BBW_SLOPE"] = df["BBW"] - df["BBW"].shift(3)

tr = np.maximum(
    df["High"] - df["Low"],
    np.maximum(abs(df["High"] - df["Close"].shift()),
               abs(df["Low"] - df["Close"].shift()))
)
df["ATR"] = tr.rolling(14).mean()

# ============================================================
# CURRENT STATE
# ============================================================
last = df.iloc[-1]
price = float(last["Close"])
vwap = float(last["VWAP"])
atr = float(last["ATR"])
ema_slope = float(last["EMA_SLOPE"])
vol_expanding = float(last["BBW_SLOPE"]) > 0

# ============================================================
# DIRECTIONAL BIAS
# ============================================================
bullish = price > vwap
bias = "BULLISH" if bullish else "BEARISH"
action = "CALLS" if bullish else "PUTS"
color = "green" if bullish else "red"

# ============================================================
# TRADE STATE LOGIC (AGGRESSIVE)
# ============================================================
state = "WAIT — NO EDGE"

if bullish:
    if ema_slope > 0 and vol_expanding:
        state = "CALLS — TREND CONTINUATION"
    elif ema_slope > 0:
        state = "CALLS — PULLBACK ENTRY"
    if price < vwap:
        state = "EXIT — BIAS LOST"
else:
    if ema_slope < 0 and vol_expanding:
        state = "PUTS — TREND CONTINUATION"
    elif ema_slope < 0:
        state = "PUTS — PULLBACK ENTRY"
    if price > vwap:
        state = "EXIT — BIAS LOST"

# ============================================================
# EXPECTED MOVE — ANCHORED TO PRICE (FIXED)
# ============================================================
if bullish:
    tgt1 = price + atr
    tgt2 = price + 2*atr
    stretch = price + 3*atr
    invalid = vwap
    instruction = "Buy pullbacks. Stay long."
else:
    tgt1 = price - atr
    tgt2 = price - 2*atr
    stretch = price - 3*atr
    invalid = vwap
    instruction = "Sell rips. Stay short."

# ============================================================
# COMMAND RIBBON
# ============================================================
scroll = (
    f"{SYMBOL} | {price:.2f} | {bias} | {state} | "
    f"TARGET {tgt1:.2f} → {tgt2:.2f} | "
    f"STRETCH {stretch:.2f} | INVALID {invalid:.2f}"
)

st.markdown(f"""
<div id="command">
  <div class="marquee {color}">
    <span>{scroll}</span><span>{scroll}</span><span>{scroll}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN HUD
# ============================================================
st.markdown(f"""
<div class="center">
  <div class="symbol">{SYMBOL}</div>
  <div class="price {color}">{price:.2f}</div>
  <div class="action {color}">{state}</div>

  <div class="range">
    EXPECTED MOVE: {tgt1:.2f} → {tgt2:.2f} &nbsp; | &nbsp; STRETCH {stretch:.2f}
  </div>

  <div class="instruction">{instruction} Exit if invalidated.</div>
  <div class="why">VWAP defines bias. EMA slope confirms momentum. Volatility regime supports move.</div>
</div>
""", unsafe_allow_html=True)

st.line_chart(df[["Close", "EMA9", "VWAP"]].tail(200), height=260)

st.caption("Decision-support only. Not financial advice.")