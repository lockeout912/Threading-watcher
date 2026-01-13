# ============================================================
# LOCKOUT SIGNALS — HIGH-SPEED SPY COMMAND CENTER
# Aggressive Trend Hunter • Instruction-First UI
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

# ============================================================
# SETTINGS
# ============================================================
SYMBOL = "SPY"
EMA_LEN = 9
ATR_LEN = 14
BB_LEN = 20
BB_STD = 2
ORB_MINUTES = 5
FLIP_LOOKBACK = 20

# ============================================================
# STYLES — COMMAND CENTER HUD
# ============================================================
st.markdown("""
<style>
body { overflow-x: hidden; background-color: #0b0f14; }
.block-container { padding-top: 7rem; max-width: 1200px; }

#command-center {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 64px;
    z-index: 9999;
    background: #05080d;
    border-bottom: 2px solid #1f2933;
    overflow: hidden;
}

.marquee {
    display: flex;
    white-space: nowrap;
    animation: scroll-left 18s linear infinite;
    font-size: 1.15rem;
    font-weight: 900;
    padding-left: 100%;
}

.marquee span { margin-right: 3rem; }

@keyframes scroll-left {
    0% { transform: translateX(0%); }
    100% { transform: translateX(-100%); }
}

.green { color: #00ff9c; }
.red { color: #ff4d4d; }
.gray { color: #cccccc; }

.center {
    text-align: center;
    margin-top: 30px;
}

.price {
    font-size: 5.5rem;
    font-weight: 900;
    letter-spacing: -2px;
}

.symbol {
    font-size: 1.8rem;
    opacity: 0.8;
}

.action {
    font-size: 2.2rem;
    font-weight: 900;
    margin-top: 10px;
}

.range {
    font-size: 1.3rem;
    font-weight: 800;
    margin-top: 18px;
}

.instruction {
    font-size: 1.15rem;
    margin-top: 18px;
    opacity: 0.9;
}

.why {
    font-size: 0.95rem;
    opacity: 0.65;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=15)
def fetch_data():
    df = yf.Ticker(SYMBOL).history(period="5d", interval="1m")
    return df.dropna()

df = fetch_data()
if df.empty or len(df) < 60:
    st.stop()

# ============================================================
# INDICATORS (ENGINE)
# ============================================================
df["EMA9"] = df["Close"].ewm(span=EMA_LEN, adjust=False).mean()
df["EMA_SLOPE"] = df["EMA9"] - df["EMA9"].shift(3)

tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

mid = df["Close"].rolling(BB_LEN).mean()
std = df["Close"].rolling(BB_LEN).std()
df["BBW"] = ((mid + BB_STD * std) - (mid - BB_STD * std)) / mid
df["BBW_SLOPE"] = df["BBW"] - df["BBW"].shift(3)

tr = np.maximum(
    df["High"] - df["Low"],
    np.maximum(abs(df["High"] - df["Close"].shift()),
               abs(df["Low"] - df["Close"].shift()))
)
df["ATR"] = tr.rolling(ATR_LEN).mean()

# ============================================================
# OPENING RANGE
# ============================================================
today = df.index[-1].date()
day_df = df[df.index.date == today]
orb_high = day_df.iloc[:ORB_MINUTES]["High"].max() if len(day_df) >= ORB_MINUTES else None
orb_low = day_df.iloc[:ORB_MINUTES]["Low"].min() if len(day_df) >= ORB_MINUTES else None

# ============================================================
# STATE LOGIC
# ============================================================
last = df.iloc[-1]
price = float(last["Close"])
vwap = float(last["VWAP"])
atr = float(last["ATR"])

ema_slope = float(last["EMA_SLOPE"])
vol_expanding = float(last["BBW_SLOPE"]) > 0

bias = "BULLISH" if price > vwap else "BEARISH"
action = "CALLS" if bias == "BULLISH" else "PUTS"
color = "green" if bias == "BULLISH" else "red"

state = "WAIT"
if bias == "BULLISH":
    if orb_high and price > orb_high and ema_slope > 0:
        state = "ENTRY ACTIVE"
    elif ema_slope > 0:
        state = "SETUP"
    if price < vwap:
        state = "EXIT"
else:
    if orb_low and price < orb_low and ema_slope < 0:
        state = "ENTRY ACTIVE"
    elif ema_slope < 0:
        state = "SETUP"
    if price > vwap:
        state = "EXIT"

if state == "ENTRY ACTIVE" and not vol_expanding:
    state = "CAUTION"

# ============================================================
# RANGE
# ============================================================
if bias == "BULLISH":
    r1 = vwap + atr
    r2 = vwap + 2 * atr
    stretch = vwap + 3 * atr
    invalid = vwap - atr
else:
    r1 = vwap - 2 * atr
    r2 = vwap - atr
    stretch = vwap - 3 * atr
    invalid = vwap + atr

# ============================================================
# COMMAND CENTER SCROLL
# ============================================================
scroll = (
    f"{SYMBOL} | PRICE {price:.2f} | {bias} | {action} | {state} | "
    f"TARGET {r1:.2f} → {r2:.2f} | STRETCH {stretch:.2f} | "
    f"EXIT IF {'BELOW' if bias=='BULLISH' else 'ABOVE'} {invalid:.2f}"
)

st.markdown(f"""
<div id="command-center">
  <div class="marquee {color}">
    <span>{scroll}</span><span>{scroll}</span><span>{scroll}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CENTER HUD
# ============================================================
st.markdown(f"""
<div class="center">
  <div class="symbol">{SYMBOL}</div>
  <div class="price {color}">{price:.2f}</div>
  <div class="action {color}">{action} — {state}</div>

  <div class="range">
    EXPECTED MOVE: {r1:.2f} → {r2:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; STRETCH {stretch:.2f}
  </div>

  <div class="instruction">
    { "Buy pullbacks. Stay long." if bias=="BULLISH" else "Sell rips. Stay short." }
    Exit if invalidated.
  </div>

  <div class="why">
    VWAP aligned, momentum slope agrees, volatility regime supports move.
  </div>
</div>
""", unsafe_allow_html=True)

st.line_chart(df[["Close", "VWAP", "EMA9"]].tail(200), height=260)

st.caption("Decision-support only. Not financial advice.")