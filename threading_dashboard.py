# ============================================================
# LOCKOUT SIGNALS — AGGRESSIVE SPY COMMAND CENTER
# Full Single-File Streamlit App
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Lockout Signals • SPY Command Center",
    layout="wide"
)

# ============================================================
# GLOBAL SETTINGS
# ============================================================
SYMBOL = "SPY"

EMA_LEN = 9
ATR_LEN = 14
BB_LEN = 20
BB_STD = 2

ORB_MINUTES = 5
FLIP_LOOKBACK = 20

# ============================================================
# GLOBAL STYLES (NO OVERLAYS, FIXED COMMAND CENTER)
# ============================================================
st.markdown("""
<style>
body { overflow-x: hidden; }
.block-container { padding-top: 7rem; max-width: 1250px; }

#command-center {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 64px;
    z-index: 9999;
    background: #0b0f14;
    border-bottom: 2px solid #1f2933;
    overflow: hidden;
}

.marquee {
    display: flex;
    white-space: nowrap;
    animation: scroll-left 22s linear infinite;
    font-size: 1.15rem;
    font-weight: 900;
    padding-left: 100%;
}

.marquee span {
    margin-right: 3rem;
}

@keyframes scroll-left {
    0% { transform: translateX(0%); }
    100% { transform: translateX(-100%); }
}

.green { color: #00ff9c; }
.red { color: #ff4d4d; }
.gray { color: #cccccc; }

.card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 18px;
}

.action {
    font-size: 1.8rem;
    font-weight: 900;
}

.sub {
    font-size: 1.05rem;
    opacity: 0.85;
}

.range {
    font-size: 1.15rem;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=15)
def fetch_data():
    df = yf.Ticker(SYMBOL).history(
        period="5d",
        interval="1m",
        auto_adjust=False
    )
    df = df.dropna()
    return df

df = fetch_data()

if df.empty or len(df) < 60:
    st.warning("Not enough data yet.")
    st.stop()

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

# EMA
df["EMA9"] = df["Close"].ewm(span=EMA_LEN, adjust=False).mean()
df["EMA_SLOPE"] = df["EMA9"] - df["EMA9"].shift(3)

# VWAP
tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
df["VWAP_SLOPE"] = df["VWAP"] - df["VWAP"].shift(3)

# Bollinger Band Width
mid = df["Close"].rolling(BB_LEN).mean()
std = df["Close"].rolling(BB_LEN).std()
df["BBW"] = ((mid + BB_STD * std) - (mid - BB_STD * std)) / mid
df["BBW_SLOPE"] = df["BBW"] - df["BBW"].shift(3)

# ATR
tr = np.maximum(
    df["High"] - df["Low"],
    np.maximum(
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    )
)
df["ATR"] = tr.rolling(ATR_LEN).mean()

# ============================================================
# OPENING RANGE (5 MIN)
# ============================================================
today = df.index[-1].date()
day_df = df[df.index.date == today]

orb_high = None
orb_low = None

if len(day_df) >= ORB_MINUTES:
    orb_high = day_df.iloc[:ORB_MINUTES]["High"].max()
    orb_low = day_df.iloc[:ORB_MINUTES]["Low"].min()

# ============================================================
# VWAP FLIP COUNT (CHOP FILTER)
# ============================================================
recent = df.tail(FLIP_LOOKBACK)
flips = ((recent["Close"] > recent["VWAP"]) !=
         (recent["Close"].shift() > recent["VWAP"].shift())).sum()

# ============================================================
# CURRENT MARKET STATE
# ============================================================
last = df.iloc[-1]

price = float(last["Close"])
vwap = float(last["VWAP"])
atr = float(last["ATR"])

ema_slope = float(last["EMA_SLOPE"])
bbw_slope = float(last["BBW_SLOPE"])

bias = "Bullish" if price > vwap else "Bearish"
action = "CALLS" if bias == "Bullish" else "PUTS"
bias_color = "green" if bias == "Bullish" else "red"

vol_expanding = bbw_slope > 0
momentum_up = ema_slope > 0

# Trend strength
if vol_expanding and momentum_up:
    strength = "HIGH"
elif momentum_up:
    strength = "MED"
else:
    strength = "LOW"

# ============================================================
# SIGNAL STATE MACHINE
# ============================================================
state = "WAIT"

# Chop filter
if flips > 8:
    state = "WAIT"

elif bias == "Bullish":
    if orb_high and price > orb_high and momentum_up:
        state = "ENTRY ACTIVE"
    elif momentum_up:
        state = "SETUP"

    if price < vwap or ema_slope < 0:
        state = "EXIT"

elif bias == "Bearish":
    if orb_low and price < orb_low and ema_slope < 0:
        state = "ENTRY ACTIVE"
    elif ema_slope < 0:
        state = "SETUP"

    if price > vwap or ema_slope > 0:
        state = "EXIT"

# Caution if trend weakens
if state == "ENTRY ACTIVE" and not vol_expanding:
    state = "CAUTION"

# ============================================================
# RANGE CALCULATION
# ============================================================
if bias == "Bullish":
    base_lo = vwap + atr
    base_hi = vwap + 2 * atr
    stretch = vwap + 3 * atr
    invalid = vwap - atr
else:
    base_lo = vwap - 2 * atr
    base_hi = vwap - atr
    stretch = vwap - 3 * atr
    invalid = vwap + atr

# ============================================================
# COMMAND CENTER MESSAGE
# ============================================================
cmd = (
    f"{SYMBOL} | {bias.upper()} TREND | {action} FAVORED | {state} | "
    f"EXPECTED {base_lo:.2f} → {base_hi:.2f} | "
    f"STRETCH {stretch:.2f} | "
    f"EXIT IF {'BELOW' if bias=='Bullish' else 'ABOVE'} {invalid:.2f} | "
    f"STRENGTH {strength}"
)

st.markdown(f"""
<div id="command-center">
    <div class="marquee {bias_color}">
        <span>{cmd}</span>
        <span>{cmd}</span>
        <span>{cmd}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ACTION PANEL
# ============================================================
st.markdown(f"""
<div class="card">
    <div class="action {bias_color}">{action} — {state}</div>
    <div class="sub">Bias: {bias} | Trend Strength: {strength}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RANGE PANEL
# ============================================================
st.markdown(f"""
<div class="card">
    <div class="range">Expected Probable Move</div>
    <div>Base Range: <b>{base_lo:.2f} → {base_hi:.2f}</b></div>
    <div>Stretch (Trend Day): <b>{stretch:.2f}</b></div>
    <div>Exit / Invalidation: <b>{invalid:.2f}</b></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# WHY PANEL
# ============================================================
st.markdown("""
<div class="card">
<b>Why this signal:</b>
<ul>
<li>Price aligned with VWAP (institutional anchor)</li>
<li>Momentum slope confirms direction</li>
<li>Volatility regime supports expansion</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MICRO TREND CHART
# ============================================================
st.line_chart(
    df[["Close", "VWAP", "EMA9"]].tail(240),
    height=260
)

st.caption("Decision-support system only. Not financial advice.")