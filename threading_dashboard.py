# ============================================================
# LOCKOUT SIGNALS — HIGH-SPEED COMMAND CENTER (AGGRESSIVE v3)
# SPY + BTC | staged alerts | regime filter | strength score
# ============================================================

import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Optional autorefresh (works if package exists). If missing, manual refresh still works.
try:
    from streamlit_autorefresh import st_autorefresh  # pip package: streamlit-autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lockout Signals • Command Center", layout="wide")

# ============================================================
# UI / STYLE
# ============================================================
st.markdown("""
<style>
body { background-color:#0b0f14; overflow-x:hidden; }
.block-container { padding-top:6.3rem; max-width:1200px; }

#command {
  position:fixed; top:0; left:0; width:100%; height:62px;
  background:#05080d; border-bottom:2px solid #1f2933;
  z-index:9999; overflow:hidden;
}

.marquee {
  display:flex; white-space:nowrap;
  animation:scroll 15s linear infinite;
  font-size:1.05rem; font-weight:900;
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
.yellow { color:#ffd166; }

.center { text-align:center; margin-top:22px; }
.symbol { font-size:1.5rem; opacity:0.85; letter-spacing:1px; }
.price { font-size:5.6rem; font-weight:950; letter-spacing:-2px; line-height:1.0; }
.action { font-size:2.0rem; font-weight:950; margin-top:8px; }
.subline { font-size:1.05rem; margin-top:10px; opacity:0.92; }
.range { font-size:1.25rem; font-weight:900; margin-top:16px; }
.why { font-size:0.95rem; opacity:0.62; margin-top:6px; }
.kpis { margin-top:18px; }
.smallcap { font-size:0.92rem; opacity:0.65; }

.badge {
  display:inline-block;
  padding:6px 12px;
  border:1px solid #24303a;
  border-radius:999px;
  margin:0 6px;
  font-weight:900;
  font-size:0.95rem;
  background:#0b0f14;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Command Settings")

asset = st.sidebar.selectbox("Asset", ["SPY", "BTC"], index=0)
symbol = "SPY" if asset == "SPY" else "BTC-USD"

orb_minutes = st.sidebar.selectbox("Opening Range", [3, 5, 10, 15], index=1)
show_chart = st.sidebar.checkbox("Show chart", value=True)

refresh_seconds = st.sidebar.selectbox("Refresh (seconds)", [5, 10, 15, 20, 30], index=1)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
if auto_refresh and not HAS_AUTOREFRESH:
    st.sidebar.warning("Auto-refresh needs `streamlit-autorefresh`. Manual refresh still works.")
    auto_refresh = False

manual = st.sidebar.button("🔄 Refresh now")

# If autorefresh is available, use it (best UX).
if auto_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_seconds * 1000, key="refresh")

# ============================================================
# DATA FETCH (near-live)
# ============================================================
@st.cache_data(ttl=8)
def fetch_history(tkr: str) -> pd.DataFrame:
    # 1m bars, last 5 days. (yfinance can lag; this is best-effort.)
    df = yf.Ticker(tkr).history(period="5d", interval="1m")
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

df = fetch_history(symbol)

if manual:
    # Bust cache for a forced refresh
    fetch_history.clear()
    df = fetch_history(symbol)

if df.empty or len(df) < 120:
    st.error("Not enough data yet. Try again in a minute.")
    st.stop()

# ============================================================
# INDICATORS — FAST + AGGRESSIVE
# ============================================================
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_dn = pd.Series(dn, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def roc(series: pd.Series, length: int = 9) -> pd.Series:
    return (series / series.shift(length) - 1.0) * 100.0

# EMA trend backbone
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["EMA_SLOPE"] = df["EMA9"] - df["EMA9"].shift(3)

# VWAP (session-style approximation using cumulative since dataframe start)
tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

# Volatility expansion
mid = df["Close"].rolling(20).mean()
std = df["Close"].rolling(20).std()
df["BBW"] = ((mid + 2*std) - (mid - 2*std)) / (mid + 1e-9)
df["BBW_SLOPE"] = df["BBW"] - df["BBW"].shift(3)

# ATR
tr = np.maximum(
    df["High"] - df["Low"],
    np.maximum(abs(df["High"] - df["Close"].shift()),
               abs(df["Low"] - df["Close"].shift()))
)
df["ATR"] = pd.Series(tr, index=df.index).rolling(14).mean()

# Momentum extras (fast)
df["ROC9"] = roc(df["Close"], 9)
df["RSI14"] = rsi(df["Close"], 14)

# ============================================================
# SESSION / LEVELS
# ============================================================
last = df.iloc[-1]
price = float(last["Close"])
vwap = float(last["VWAP"])
atr = float(last["ATR"]) if pd.notna(last["ATR"]) else float(df["ATR"].dropna().iloc[-1])
ema_slope = float(last["EMA_SLOPE"])
bbw_slope = float(last["BBW_SLOPE"])
roc9 = float(last["ROC9"]) if pd.notna(last["ROC9"]) else 0.0
rsi14 = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0

# Today's slice (for OR + session open)
today = df.index[-1].date()
day_df = df[df.index.date == today]
session_open = float(day_df["Open"].iloc[0]) if len(day_df) > 0 else float(df["Open"].iloc[-1])

orb_hi = None
orb_lo = None
if len(day_df) >= orb_minutes:
    orb_hi = float(day_df.iloc[:orb_minutes]["High"].max())
    orb_lo = float(day_df.iloc[:orb_minutes]["Low"].min())

# Flip count: how often price crosses VWAP recently (chop detector)
def flip_count(close: pd.Series, anchor: pd.Series, lookback: int = 20) -> int:
    x = (close - anchor).tail(lookback)
    s = np.sign(x.fillna(0.0))
    return int(np.sum(s.values[1:] != s.values[:-1]))

flips = flip_count(df["Close"], df["VWAP"], lookback=20)

# ============================================================
# REGIME FILTER (TREND / RANGE / CHOP)
# ============================================================
# Chop if: lots of flips + low volatility expansion
bbw = float(last["BBW"]) if pd.notna(last["BBW"]) else 0.0
vol_expanding = bbw_slope > 0

# No-trade zone around VWAP (aggressive mode still needs a "do not bleed" guardrail)
no_trade_band = 0.15 * atr
near_vwap = abs(price - vwap) <= no_trade_band

if flips >= 8 and (bbw < 0.01 or not vol_expanding) and near_vwap:
    regime = "CHOP"
elif vol_expanding and flips <= 4 and abs(price - vwap) > no_trade_band:
    regime = "TREND"
else:
    regime = "RANGE"

# ============================================================
# STRENGTH SCORE (0–100) — one number to rule behavior
# ============================================================
# Components normalized in an aggressive, simple way
vwap_dist = abs(price - vwap) / (atr + 1e-9)          # 0..?
ema_strength = abs(ema_slope) / (atr + 1e-9)          # slope vs ATR
mom_strength = abs(roc9) / 0.25                       # 0.25% ROC ~ “meaningful” intraday pop (tunable)
vol_strength = 1.0 if vol_expanding else 0.0
flip_penalty = min(flips / 10.0, 1.0)

raw = (
    35 * np.clip(vwap_dist, 0, 2) / 2 +
    25 * np.clip(ema_strength, 0, 1.5) / 1.5 +
    20 * np.clip(mom_strength, 0, 2) / 2 +
    20 * vol_strength
)
score = int(np.clip(raw * (1.0 - 0.55 * flip_penalty), 0, 100))

# ============================================================
# BIAS (CALLS/PUTS) + STAGED STATES (AGGRESSIVE)
# ============================================================
bullish_bias = price > vwap
bias = "BULLISH" if bullish_bias else "BEARISH"
action = "CALLS" if bullish_bias else "PUTS"
color = "green" if bullish_bias else "red"

# Early heads up triggers (fast + aggressive)
# - momentum turns positive/negative OR
# - reclaim/lose VWAP recently OR
# - ROC crosses 0 direction
recent = df.tail(6).copy()
vwap_side = np.sign((recent["Close"] - recent["VWAP"]).fillna(0.0))
just_flipped_bias = np.any(vwap_side.values[1:] != vwap_side.values[:-1])

heads_up = False
if bullish_bias and (roc9 > 0 or ema_slope > 0 or just_flipped_bias):
    heads_up = True
if (not bullish_bias) and (roc9 < 0 or ema_slope < 0 or just_flipped_bias):
    heads_up = True

# Breakout / continuation hints
above_orh = (orb_hi is not None) and (price > orb_hi)
below_orl = (orb_lo is not None) and (price < orb_lo)

# Extension / exhaustion
ext_from_vwap = (price - vwap) / (atr + 1e-9)  # in ATR units (+/-)
extended = (abs(ext_from_vwap) >= 1.5) or (rsi14 >= 72) or (rsi14 <= 28)

# Momentum roll-over for caution (fast)
roc_roll = float(recent["ROC9"].iloc[-1] - recent["ROC9"].iloc[-3]) if len(recent) >= 4 and pd.notna(recent["ROC9"].iloc[-3]) else 0.0
momentum_fading = (bullish_bias and roc_roll < 0) or ((not bullish_bias) and roc_roll > 0)

# --------------------------
# STATE DECISION TREE
# --------------------------
state = "WAIT — NO EDGE"
sub_instruction = "Stand down. No edge."
why_line = "No clear alignment."

if regime == "CHOP" or (near_vwap and flips >= 6):
    state = "WAIT — NO TRADE ZONE"
    sub_instruction = "Do nothing. Chop around VWAP."
    why_line = "VWAP flip risk high."

else:
    # HEADS UP (early)
    if heads_up:
        if bullish_bias:
            state = "HEADS UP — BULLISH (CALLS BIAS)"
            sub_instruction = "Scout calls. Wait for continuation/pullback confirmation."
            why_line = "Bias + early momentum."
        else:
            state = "HEADS UP — BEARISH (PUTS BIAS)"
            sub_instruction = "Scout puts. Wait for continuation/pullback confirmation."
            why_line = "Bias + early momentum."

    # ENTRY ACTIVE thresholds (aggressive)
    # We allow earlier entries when score is decent OR OR break happens
    entry_active = False
    if bullish_bias:
        entry_active = (score >= 55 and (ema_slope > 0 or roc9 > 0)) or above_orh
    else:
        entry_active = (score >= 55 and (ema_slope < 0 or roc9 < 0)) or below_orl

    # TREND CONTINUATION vs PULLBACK ENTRY
    if entry_active:
        if score >= 80 and regime == "TREND":
            state = f"{action} — TREND CONTINUATION"
            sub_instruction = "Trade with trend. Add on clean pullbacks."
            why_line = "Strong alignment + expanding volatility."
        else:
            state = f"{action} — ENTRY ACTIVE"
            sub_instruction = "Entry allowed. Prefer pullbacks / reclaims."
            why_line = "Bias aligned; momentum acceptable."

    # EXTENDED / CAUTION / EXIT overlays
    if extended and entry_active:
        state = f"{action} — EXTENDED (MANAGE RISK)"
        sub_instruction = "Protect profits. Tighten stop. Avoid chasing."
        why_line = "Extended from VWAP / momentum stretched."

    if momentum_fading and entry_active:
        state = f"{action} — CAUTION (MOMENTUM FADING)"
        sub_instruction = "Late entry risk. Scale or wait for reset."
        why_line = "Momentum rollover warning."

    # Hard EXIT when bias breaks (VWAP is truth line)
    if bullish_bias and price < vwap:
        state = "EXIT — BIAS LOST"
        sub_instruction = "Exit calls. Wait for reclaim."
        why_line = "Price lost VWAP."
    if (not bullish_bias) and price > vwap:
        state = "EXIT — BIAS LOST"
        sub_instruction = "Exit puts. Wait for reclaim."
        why_line = "Price reclaimed VWAP."

# ============================================================
# EXPECTED MOVE (FROM HERE) — Likely / Possible / Stretch
# ============================================================
# Aggressive mode: slightly wider “likely” on TREND days
mult = 1.2 if regime == "TREND" else 1.0
likely = atr * mult
possible = atr * (2.0 * mult)
stretch = atr * (3.0 * mult)

if bullish_bias:
    tgt1 = price + likely
    tgt2 = price + possible
    tgt3 = price + stretch
else:
    tgt1 = price - likely
    tgt2 = price - possible
    tgt3 = price - stretch

invalid = vwap  # truth line

# ============================================================
# COMMAND RIBBON (SCROLLING, ALWAYS)
# ============================================================
badge_color = "green" if bullish_bias else "red"
reg_color = "yellow" if regime == "RANGE" else ("gray" if regime == "CHOP" else badge_color)

ribbon = (
    f"{asset} | {price:.2f} | {bias} | {regime} | SCORE {score}/100 | "
    f"{state} | "
    f"LIKELY {tgt1:.2f} | POSS {tgt2:.2f} | STRETCH {tgt3:.2f} | "
    f"INVALID {invalid:.2f}"
)

st.markdown(f"""
<div id="command">
  <div class="marquee {badge_color}">
    <span>{ribbon}</span><span>{ribbon}</span><span>{ribbon}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN HUD (PRICE DOMINANT)
# ============================================================
st.markdown(f"""
<div class="center">
  <div class="symbol">{asset}</div>
  <div class="price {badge_color}">{price:.2f}</div>
  <div class="action {badge_color}">{state}</div>

  <div class="kpis">
    <span class="badge {badge_color}">{bias} — {action}</span>
    <span class="badge {reg_color}">REGIME: {regime}</span>
    <span class="badge gray">SCORE: {score}/100</span>
  </div>

  <div class="range">
    EXPECTED MOVE (FROM HERE): <span class="{badge_color}">LIKELY {tgt1:.2f}</span>
    &nbsp; | &nbsp; POSS {tgt2:.2f}
    &nbsp; | &nbsp; STRETCH {tgt3:.2f}
  </div>

  <div class="subline">{sub_instruction} <b>Invalid:</b> {invalid:.2f}</div>

  <div class="why">{why_line}</div>
</div>
""", unsafe_allow_html=True)

# Small tactical readout (kept light)
c1, c2, c3, c4 = st.columns(4)
c1.metric("VWAP", f"{vwap:.2f}", f"{(price - vwap):+.2f}")
c2.metric("Session Open", f"{session_open:.2f}", f"{(price - session_open):+.2f}")
c3.metric("ATR(14)", f"{atr:.2f}", f"{ext_from_vwap:+.2f}x from VWAP")
c4.metric("Momentum", f"ROC9 {roc9:+.2f}%", f"RSI {rsi14:.0f}")

# Opening range levels (if available)
if orb_hi is not None and orb_lo is not None:
    st.caption(f"OR({orb_minutes}m) High: {orb_hi:.2f} | OR Low: {orb_lo:.2f} | Flips(20): {flips} | Near VWAP band: ±{no_trade_band:.2f}")

# Chart (optional)
if show_chart:
    st.line_chart(df[["Close", "EMA9", "VWAP"]].tail(240), height=260)

st.caption("Decision-support only. Not financial advice.")