import time
from datetime import datetime, time as dtime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lockout Signals • Command Center", layout="wide")


# ============================================================
# THEME (DARK DESKTOP)
# ============================================================
st.markdown("""
<style>
:root{
  --bg:#0a0a0a;
  --panel:#121212;
  --text:#e7edf5;
  --muted:#9db0c6;
  --good:#00ff95;
  --bad:#ff4d4d;
  --warn:#ffc44d;
  --cyan:#4de3ff;
}
html, body, [class*="stApp"] { background-color: var(--bg) !important; color: var(--text) !important; }
section[data-testid="stSidebar"] { background-color: var(--panel) !important; }
.block-container { padding-top: 0.6rem; max-width: 1200px; }
div[data-testid="stMetricValue"] { font-size: 2.0rem; }
div[data-testid="stMetricLabel"] { color: rgba(231,237,245,0.75) !important; }

.ribbon {
  position: sticky; top: 0; z-index: 9999;
  background: rgba(10,10,10,0.92);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 10px 14px;
  margin-bottom: 14px;
  overflow: hidden;
}
.marquee { white-space: nowrap; overflow: hidden; position: relative; }
.marquee span {
  display: inline-block; padding-left: 100%;
  animation: marquee 14s linear infinite;
  font-weight: 900; letter-spacing: 0.6px;
  color: var(--text);
}
@keyframes marquee { 0% { transform: translateX(0%);} 100% { transform: translateX(-100%);} }

.pill{
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  font-weight: 900;
}
.good{ color: var(--good); border-color: rgba(0,255,149,0.35); }
.bad { color: var(--bad);  border-color: rgba(255,77,77,0.35); }
.warn{ color: var(--warn); border-color: rgba(255,196,77,0.35); }
.cyan{ color: var(--cyan); border-color: rgba(77,227,255,0.35); }

.hero{
  text-align:center;
  padding: 18px 10px 12px 10px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.06);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
}
.symbol{ font-size: 34px; font-weight: 1000; color: rgba(231,237,245,0.72); letter-spacing:2px; }
.price{ font-size: 86px; line-height: 1.0; font-weight: 1100; margin: 8px 0 10px; }
.action{ font-size: 44px; font-weight: 1100; margin: 6px 0 10px; letter-spacing:1px; }

.krow{
  display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-top: 10px;
}
.hr{ height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }

.targets{ text-align:center; margin-top: 10px; font-weight: 1000; letter-spacing: 0.8px; }
.targets .label{ color: rgba(231,237,245,0.75); font-weight: 1000; }
.targets .val{ font-size: 28px; font-weight: 1100; margin-top: 4px; }

.subline{ font-size: 18px; color: var(--muted); font-weight: 800; margin-top: 6px; }
.whyline{ font-size: 16px; color: rgba(231,237,245,0.78); margin-top: 6px; }
.smallnote{ text-align:center; color: rgba(231,237,245,0.55); font-size: 13px; margin-top: 10px; }

table { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ASSETS UNIVERSE (USED FOR TOP MOVERS + DROPDOWN)
# ============================================================
ASSETS = {
    # Core
    "SPY": "SPY",
    "QQQ": "QQQ",

    # Crypto (still uses 5m/1m to match your request)
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "XLM": "XLM-USD",
    "DOGE": "DOGE-USD",

    # High liquidity / mega cap
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "META": "META",
    "AMZN": "AMZN",
    "GOOG": "GOOG",
    "NFLX": "NFLX",
    "TSLA": "TSLA",

    # Financial / Energy / Staples
    "JPM": "JPM",
    "BAC": "BAC",
    "WFC": "WFC",
    "XOM": "XOM",
    "OXY": "OXY",
    "KO": "KO",
    "PFE": "PFE",
    "NKE": "NKE",
    "DIS": "DIS",
    "BA": "BA",

    # Your momentum list
    "GME": "GME",
    "AMC": "AMC",
    "PLTR": "PLTR",
    "SOFI": "SOFI",
    "AMD": "AMD",
    "OPEN": "OPEN",
    "ASTS": "ASTS",
    "U": "U",
    "HYMC": "HYMC",
    "BITO": "BITO",
    "RIOT": "RIOT",
    "MARA": "MARA",
    "MSTR": "MSTR",
    "MSTU": "MSTU",
    "IREN": "IREN",
    "CLSK": "CLSK",
    "NOK": "NOK",
}


# ============================================================
# MARKET STATUS
# ============================================================
def market_status_for(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "24/7"

    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern).time()

    if dtime(9, 30) <= now <= dtime(16, 0):
        return "MARKET OPEN"
    if dtime(4, 0) <= now < dtime(9, 30):
        return "PRE-MARKET"
    return "AFTER HOURS"


# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=12)
def fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(
            period=period, interval=interval, prepost=True,
            actions=False, auto_adjust=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "ts"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "ts"})
        else:
            df = df.rename(columns={df.columns[0]: "ts"})
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20)
def fetch_batch_5m(symbols: list[str]) -> dict:
    """
    Batch 5m pull for movers. Returns dict {symbol: df}
    """
    out = {}
    for s in symbols:
        out[s] = fetch_intraday(s, "5m", "1d")
    return out


# ============================================================
# INDICATORS
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d["Volume"].fillna(0)
    pv = (tp * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return (pv / vv).ffill()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    d = df.copy()
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        (d["High"] - d["Low"]).abs(),
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def slope(series: pd.Series, lookback: int = 8) -> float:
    if series is None or len(series) < lookback + 1:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-lookback - 1])


def prep(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["EMA9"] = ema(d["Close"], 9)
    d["EMA21"] = ema(d["Close"], 21)
    d["VWAP"] = vwap(d)
    return d


# ============================================================
# CHOP FILTER (FAST + EFFECTIVE)
# ============================================================
def chop_score(df_1m: pd.DataFrame) -> float:
    """
    0..100 (higher = choppier). Uses:
    - VWAP cross frequency
    - wickiness (wicks >> bodies)
    """
    if df_1m is None or df_1m.empty or len(df_1m) < 60:
        return 50.0

    d = df_1m.tail(60).copy()
    px = d["Close"]
    vw = d["VWAP"]

    # vwap crosses
    side = (px > vw).astype(int)
    crosses = int((side.diff().abs() > 0).sum())

    # wick ratio
    body = (d["Close"] - d["Open"]).abs()
    wick = (d["High"] - d["Low"]).abs() - body
    wick_ratio = float((wick / (body.replace(0, np.nan))).median())
    wick_ratio = 3.0 if np.isnan(wick_ratio) else min(3.0, wick_ratio)

    # scale to 0..100
    cross_component = min(60.0, crosses * 4.0)     # ~15 crosses => maxed
    wick_component = min(40.0, wick_ratio / 3.0 * 40.0)

    return float(min(100.0, cross_component + wick_component))


# ============================================================
# REGIME (TREND / RANGE / BREAKOUT-VOL)
# ============================================================
def compute_regime(df_5m: pd.DataFrame) -> str:
    if df_5m is None or df_5m.empty or len(df_5m) < 40:
        return "RANGE"
    d = df_5m.copy()
    d["ATR"] = atr(d, 14)
    a = d["ATR"].iloc[-1]
    if pd.isna(a) or a <= 0:
        return "RANGE"

    # ATR expansion vs median
    med = float(d["ATR"].rolling(30).median().iloc[-1]) if len(d) >= 30 else float(d["ATR"].median())
    if med <= 0:
        med = float(a)

    exp = float(a / med)

    # quick breakout heuristic: expanding ATR AND strong distance from VWAP
    dist = abs(float(d["Close"].iloc[-1]) - float(d["VWAP"].iloc[-1]))
    if exp >= 1.35 and dist >= 0.35 * float(a):
        return "VOLATILE BREAKOUT"
    if exp >= 1.15:
        return "TREND"
    return "RANGE"


# ============================================================
# BIAS (5m brain)
# ============================================================
def compute_bias(df_5m: pd.DataFrame) -> tuple[str, int]:
    """
    Returns (BIAS, SCORE 0..100)
    Bias uses VWAP + EMA slope + spacing.
    """
    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return "NEUTRAL", 0

    d = df_5m.copy()
    px = float(d["Close"].iloc[-1])
    vw = float(d["VWAP"].iloc[-1])
    e9 = float(d["EMA9"].iloc[-1])
    e21 = float(d["EMA21"].iloc[-1])

    s9 = slope(d["EMA9"], 8)
    sv = slope(d["VWAP"], 8)

    spacing = e9 - e21
    above = px > vw
    below = px < vw

    bull = above and (spacing >= 0) and (s9 >= 0) and (sv >= 0)
    bear = below and (spacing <= 0) and (s9 <= 0) and (sv <= 0)

    bias = "NEUTRAL"
    if bull:
        bias = "BULLISH"
    elif bear:
        bias = "BEARISH"

    # Score (simple confidence)
    score = 0
    score += 40 if bias != "NEUTRAL" else 10
    score += 20 if abs(spacing) > 0 else 5
    score += 20 if abs(s9) > 0 else 5
    score += 20 if abs(px - vw) > 0 else 5
    score = max(0, min(100, score))

    return bias, score


# ============================================================
# SIGNAL ENGINE (1m trigger) — Aggressive but sane + Full Send
# ============================================================
def classify_signal(df_1m: pd.DataFrame, df_5m: pd.DataFrame, bias: str, mode: str) -> tuple[str, str, str, float | None]:
    """
    Returns: (STATE, SIDE, WHY, INVALIDATION)
    States: HEADS UP / ENTRY ACTIVE / CAUTION / EXIT / WAIT
    """
    if df_1m is None or df_1m.empty or df_5m is None or df_5m.empty or len(df_1m) < 40:
        return "WAIT — NO DATA", "WAIT", "Not enough candles yet.", None

    d1 = df_1m.copy()
    px = float(d1["Close"].iloc[-1])
    vw = float(d1["VWAP"].iloc[-1])
    e9 = float(d1["EMA9"].iloc[-1])
    e21 = float(d1["EMA21"].iloc[-1])

    s9 = slope(d1["EMA9"], 6)

    # chop score (blocks entries)
    chop = chop_score(d1)

    # Pullback/hold logic
    recent = d1.tail(6).copy()
    closes = recent["Close"].astype(float).tolist()

    def held_above(level: float, n: int):
        return all(c > level for c in closes[-n:])

    def held_below(level: float, n: int):
        return all(c < level for c in closes[-n:])

    # MODE thresholds
    full_send = (mode == "FULL SEND")
    hold_n = 1 if full_send else 2
    chop_limit_entry = 78 if full_send else 62   # full send tolerates more chop (but still not insane)

    # Directional conditions
    long_ok = (bias in ["BULLISH", "NEUTRAL"]) and (px >= vw) and (e9 >= e21)
    short_ok = (bias in ["BEARISH", "NEUTRAL"]) and (px <= vw) and (e9 <= e21)

    # Heads up triggers (more frequent)
    heads_up_long = long_ok and (px >= e9) and (s9 >= 0)
    heads_up_short = short_ok and (px <= e9) and (s9 <= 0)

    # Entry Active triggers (aggressive but filters chop)
    entry_long = (bias == "BULLISH") and (px >= vw) and held_above(vw, hold_n) and (chop <= chop_limit_entry)
    entry_short = (bias == "BEARISH") and (px <= vw) and held_below(vw, hold_n) and (chop <= chop_limit_entry)

    # Caution (momentum fading)
    caution_long = (bias == "BULLISH") and (px < e9) and (s9 < 0)
    caution_short = (bias == "BEARISH") and (px > e9) and (s9 > 0)

    # Exit (lost VWAP + flip)
    exit_long = (bias == "BULLISH") and (px < vw) and (e9 < e21)
    exit_short = (bias == "BEARISH") and (px > vw) and (e9 > e21)

    # Invalidation
    invalid = vw

    if entry_long:
        return "ENTRY ACTIVE", "CALLS", f"Above VWAP + trend aligned. Chop:{int(chop)}", invalid
    if entry_short:
        return "ENTRY ACTIVE", "PUTS", f"Below VWAP + trend aligned. Chop:{int(chop)}", invalid

    if caution_long:
        return "CAUTION", "CALLS", f"Momentum fading (lost EMA9). Chop:{int(chop)}", invalid
    if caution_short:
        return "CAUTION", "PUTS", f"Momentum fading (lost EMA9). Chop:{int(chop)}", invalid

    if exit_long:
        return "EXIT / RESET", "CALLS", f"Lost VWAP + momentum flipped. Chop:{int(chop)}", invalid
    if exit_short:
        return "EXIT / RESET", "PUTS", f"Reclaimed VWAP + momentum flipped. Chop:{int(chop)}", invalid

    if heads_up_long and chop <= 85:
        return "HEADS UP", "CALLS", f"Bias building (watch VWAP hold). Chop:{int(chop)}", invalid
    if heads_up_short and chop <= 85:
        return "HEADS UP", "PUTS", f"Bias building (watch VWAP hold). Chop:{int(chop)}", invalid

    # If choppy, explicitly warn
    if chop >= 75:
        return "WAIT — CHOP", "WAIT", f"Chop too high. Chop:{int(chop)}", invalid

    return "WAIT — NO EDGE", ("CALLS" if bias == "BULLISH" else "PUTS" if bias == "BEARISH" else "WAIT"), f"No clean VWAP hold/reject yet. Chop:{int(chop)}", invalid


# ============================================================
# EXPECTED MOVE (ATR * REGIME MULTIPLIER) — wider & more realistic
# ============================================================
def expected_move_targets(price: float, bias: str, atr5: float, regime: str) -> tuple[float, float, float]:
    if atr5 <= 0 or np.isnan(atr5):
        atr5 = max(0.5, price * 0.0012)

    if regime == "VOLATILE BREAKOUT":
        mult = (1.20, 2.20, 3.20)
    elif regime == "TREND":
        mult = (1.00, 1.80, 2.60)
    else:
        mult = (0.60, 1.10, 1.70)

    move1, move2, move3 = atr5 * mult[0], atr5 * mult[1], atr5 * mult[2]

    if bias == "BEARISH":
        return price - move1, price - move2, price - move3
    # default bullish/neutral => upward targets for readability
    return price + move1, price + move2, price + move3


# ============================================================
# SIDEBAR: TOP MOVERS (ALWAYS VISIBLE)
# ============================================================
def build_top_movers_panel():
    st.sidebar.markdown("## 📈 Top Movers (Universe)")

    symbols = list(ASSETS.values())
    batch = fetch_batch_5m(symbols)

    rows = []
    for label, sym in ASSETS.items():
        df = batch.get(sym, pd.DataFrame())
        if df is None or df.empty or len(df) < 3:
            continue

        # session open = first candle open of 5m "1d"
        o = float(df["Open"].iloc[0])
        last = float(df["Close"].iloc[-1])
        if o <= 0:
            continue
        pct = (last - o) / o * 100.0
        rows.append({"Ticker": label, "Symbol": sym, "%": pct, "Last": last})

    if not rows:
        st.sidebar.info("No movers data yet.")
        return

    movers = pd.DataFrame(rows).sort_values("%", ascending=False)

    up = movers.head(10).copy()
    down = movers.tail(10).sort_values("%", ascending=True).copy()

    def fmt(df):
        df = df.copy()
        df["%"] = df["%"].map(lambda x: f"{x:+.2f}%")
        df["Last"] = df["Last"].map(lambda x: f"{x:,.2f}")
        return df[["Ticker", "%", "Last"]]

    st.sidebar.markdown("**Top 10 Up**")
    st.sidebar.dataframe(fmt(up), use_container_width=True, height=260)

    st.sidebar.markdown("**Top 10 Down**")
    st.sidebar.dataframe(fmt(down), use_container_width=True, height=260)

    st.sidebar.caption("Movers = % change from session open (5m).")


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown("# 🎛️ Controls")

asset_choice = st.sidebar.selectbox("Asset", list(ASSETS.keys()), index=0)
symbol = ASSETS[asset_choice]

mode = st.sidebar.radio("Mode", ["AGGRESSIVE", "FULL SEND"], index=0)

auto = st.sidebar.toggle("Auto-refresh", value=True)
refresh_s = st.sidebar.selectbox("Refresh seconds", [5, 10, 15, 20, 30], index=1)

refresh_now = st.sidebar.button("🔄 Refresh now", use_container_width=True)

# Build movers panel above the fold
build_top_movers_panel()

st.sidebar.markdown("---")
st.sidebar.caption("Sponsor logo slot reserved (we’ll add after engine lock).")


# ============================================================
# FETCH MAIN ASSET DATA (5m brain + 1m trigger)
# ============================================================
df_1m_raw = fetch_intraday(symbol, "1m", "2d")
df_5m_raw = fetch_intraday(symbol, "5m", "5d")

df_1m = prep(df_1m_raw)
df_5m = prep(df_5m_raw)

if df_1m is None or df_1m.empty or df_5m is None or df_5m.empty:
    st.warning("No data available yet for this asset. Try refresh.")
    if refresh_now:
        st.cache_data.clear()
        st.rerun()
    st.stop()

df_5m["ATR"] = atr(df_5m, 14)

# ============================================================
# COMPUTE ENGINE OUTPUTS
# ============================================================
mkt_status = market_status_for(symbol)

price = float(df_1m["Close"].iloc[-1])
vw = float(df_1m["VWAP"].iloc[-1])

bias, score = compute_bias(df_5m)
regime = compute_regime(df_5m)

state, side, why, invalid = classify_signal(df_1m, df_5m, bias, mode)

atr5 = float(df_5m["ATR"].iloc[-1]) if pd.notna(df_5m["ATR"].iloc[-1]) else 0.0
t1, t2, t3 = expected_move_targets(price, bias if bias != "NEUTRAL" else ("BULLISH" if side == "CALLS" else "BEARISH"), atr5, regime)


# ============================================================
# UI HELPERS
# ============================================================
def cls_for_bias(b):
    return "good" if b == "BULLISH" else "bad" if b == "BEARISH" else "warn"

def cls_for_state(s):
    if "ENTRY ACTIVE" in s:
        return "good"
    if "HEADS UP" in s:
        return "cyan"
    if "CAUTION" in s:
        return "warn"
    if "EXIT" in s:
        return "bad"
    if "CHOP" in s:
        return "warn"
    return "warn"

def cls_for_market(ms):
    if ms == "MARKET OPEN":
        return "good"
    if ms == "PRE-MARKET":
        return "warn"
    if ms == "24/7":
        return "cyan"
    return "warn"


# ============================================================
# COMMAND RIBBON (SCROLLING)
# ============================================================
bias_txt = f"{bias} — {'CALLS' if bias=='BULLISH' else 'PUTS' if bias=='BEARISH' else 'WAIT'}"
inv_txt = f"INVALID:{invalid:.2f}" if invalid is not None else "INVALID:—"
rng_txt = f"RANGE: {t1:.2f} | {t2:.2f} | {t3:.2f}"
hud = (
    f"NOW: {state} • SIDE: {side} • BIAS: {bias_txt} • REGIME: {regime} • "
    f"PRICE: {price:.2f} • VWAP: {vw:.2f} • {rng_txt} • {inv_txt} • "
    f"SCORE: {score}/100 • MODE: {mode} • MARKET: {mkt_status} • "
)
st.markdown(f"""
<div class="ribbon">
  <div class="marquee">
    <span>{hud}{hud}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN HERO CARD
# ============================================================
price_class = cls_for_bias(bias) if bias != "NEUTRAL" else "warn"
state_class = cls_for_state(state)
market_class = cls_for_market(mkt_status)

st.title("Lockout Signals • Command Center")

st.markdown(f"""
<div class="hero">
  <div class="symbol">{asset_choice} • 5m Brain / 1m Trigger</div>
  <div class="price {price_class}">{price:,.2f}</div>
  <div class="action {state_class}">{state} — {side}</div>

  <div class="krow">
    <span class="pill {cls_for_bias(bias)}">{bias_txt}</span>
    <span class="pill {market_class}">{mkt_status}</span>
    <span class="pill cyan">REGIME: {regime}</span>
    <span class="pill">SCORE: {score}/100</span>
    <span class="pill">MODE: {mode}</span>
  </div>

  <div class="hr"></div>

  <div class="targets">
    <div class="label">EXPECTED MOVE (WIDER • REGIME-AWARE)</div>
    <div class="val {cls_for_bias(bias if bias!='NEUTRAL' else 'BULLISH')}">
      LIKELY {t1:,.2f} &nbsp; | &nbsp; POSS {t2:,.2f} &nbsp; | &nbsp; STRETCH {t3:,.2f}
    </div>
  </div>

  <div class="subline">{inv_txt} • VWAP {vw:.2f} • ATR(5m) {atr5:.2f}</div>
  <div class="whyline">{why}</div>

  <div class="smallnote">Decision-support only. Not financial advice. Trade your plan.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# QUICK READOUT + MINI CHART
# ============================================================
st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Bias", bias, "")
with c2:
    st.metric("VWAP", f"{vw:.2f}", "")
with c3:
    st.metric("ATR(5m)", f"{atr5:.2f}", "")
with c4:
    st.metric("Chop", f"{int(chop_score(df_1m))}/100", "")

st.write("")
chart_df = df_1m.tail(180).copy().set_index("ts")[["Close", "EMA9", "VWAP"]].dropna()
st.line_chart(chart_df, height=280, use_container_width=True)

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


# ============================================================
# REFRESH
# ============================================================
if refresh_now:
    st.cache_data.clear()
    st.rerun()

if auto:
    time.sleep(int(refresh_s))
    st.rerun()