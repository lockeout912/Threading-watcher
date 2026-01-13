# app.py — Lockout Signals • Command Center
# Single-file Streamlit app. Copy/paste whole file.

import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# Optional auto-refresh support
_AUTORF_AVAILABLE = True
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    _AUTORF_AVAILABLE = False

# yfinance for free-ish quotes (not true tick-level; best effort)
import yfinance as yf

# Python 3.9+ timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# Page / Theme
# =========================
st.set_page_config(
    page_title="Lockout Signals • Command Center",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
/* ===== Global Dark ===== */
html, body, [data-testid="stAppViewContainer"] {
  background: #0b0f14 !important;
  color: rgba(255,255,255,.92) !important;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { background: #0a0e13 !important; }

/* ===== Responsive typography ===== */
:root{
  --price-size: clamp(52px, 10vw, 110px);
  --action-size: clamp(26px, 6vw, 56px);
  --subhead-size: clamp(16px, 3.6vw, 26px);
  --chip-size: clamp(12px, 2.6vw, 16px);
  --small-size: clamp(12px, 2.6vw, 14px);
}

/* ===== Command Card ===== */
.cc-card{
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.10);
  background: radial-gradient(1200px 400px at 50% 0%, rgba(255,255,255,.06), rgba(255,255,255,.02));
  padding: 22px 18px;
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
.k_subhead{
  font-size: var(--subhead-size);
  text-align: center;
  letter-spacing: .8px;
  opacity: .85;
  margin-bottom: 8px;
}
.k_price{
  font-size: var(--price-size);
  line-height: 1.0;
  text-align: center;
  font-weight: 800;
  margin: 6px 0 6px 0;
}
.k_action{
  font-size: var(--action-size);
  line-height: 1.05;
  text-align: center;
  font-weight: 800;
  letter-spacing: 1.2px;
  margin: 4px 0 12px 0;
}
.k_small{
  font-size: var(--small-size);
  opacity: .80;
  text-align: center;
}

/* ===== Chips ===== */
.k_chips{
  display:flex; gap:10px; flex-wrap:wrap;
  justify-content:center; align-items:center;
  margin-top: 10px; margin-bottom: 14px;
}
.k_chip{
  font-size: var(--chip-size);
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.16);
  background: rgba(255,255,255,.04);
  white-space: nowrap;
}

/* ===== Marquee (command feed) ===== */
.marquee-wrap{
  width: 100%;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.03);
  padding: 10px 0;
  margin: 10px 0 18px 0;
}
.marquee{
  display: inline-block;
  white-space: nowrap;
  animation: scroll-left 18s linear infinite;
  font-size: clamp(12px, 2.4vw, 14px);
  letter-spacing: .6px;
  padding-left: 100%;
}
@keyframes scroll-left{
  0% { transform: translateX(0); }
  100% { transform: translateX(-100%); }
}
.marq-good { color: #34ff9a; }
.marq-warn { color: #ffcc66; }
.marq-bad  { color: #ff5b6e; }
.marq-neutral { color: rgba(255,255,255,.82); }

/* ===== Compact KPI row ===== */
.kpi-row{
  display:flex; gap:14px; flex-wrap:wrap;
  justify-content:space-between;
  margin-top: 14px;
}
.kpi{
  flex: 1 1 160px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.03);
  padding: 12px 12px;
}
.kpi .label{
  font-size: 12px;
  opacity: .75;
  letter-spacing: .8px;
}
.kpi .value{
  font-size: 22px;
  font-weight: 800;
  margin-top: 6px;
}
.kpi .delta{
  font-size: 12px;
  opacity: .75;
  margin-top: 2px;
}

/* ===== Mobile padding ===== */
@media (max-width: 600px){
  .block-container { padding-top: 1.2rem !important; padding-left: .8rem !important; padding-right: .8rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Utilities
# =========================
def is_crypto_symbol(asset_key: str) -> bool:
    return asset_key.startswith("CRYPTO:")


def yf_symbol(asset_key: str) -> str:
    # Map internal keys to yfinance tickers
    if asset_key.startswith("CRYPTO:"):
        return asset_key.split("CRYPTO:", 1)[1]  # e.g. BTC-USD
    return asset_key


def now_et() -> dt.datetime:
    if ZoneInfo is None:
        return dt.datetime.utcnow()
    return dt.datetime.now(ZoneInfo("America/New_York"))


def market_status(asset_key: str) -> str:
    # Crypto = always open
    if is_crypto_symbol(asset_key):
        return "24/7 OPEN"

    t = now_et()
    # Weekend
    if t.weekday() >= 5:
        return "CLOSED"

    # US equity session (rough, no holiday calendar)
    # Pre: 04:00–09:30, Regular: 09:30–16:00, After: 16:00–20:00
    hhmm = t.hour * 60 + t.minute
    pre = 4 * 60
    open_ = 9 * 60 + 30
    close_ = 16 * 60
    after = 20 * 60

    if pre <= hhmm < open_:
        return "PRE-MARKET"
    if open_ <= hhmm < close_:
        return "MARKET OPEN"
    if close_ <= hhmm < after:
        return "AFTER HOURS"
    return "CLOSED"


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=n, adjust=False).mean()


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(n).sum()
    hi = df["High"].rolling(n).max()
    lo = df["Low"].rolling(n).min()
    rng = (hi - lo).replace(0, np.nan)
    chop = 100 * np.log10(tr_sum / rng) / np.log10(n)
    return chop


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    # VWAP based on typical price and volume
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].copy()
    # If volume missing (crypto sometimes), fake volume = 1
    vol = vol.replace(0, np.nan).fillna(1.0)
    cum = (tp * vol).cumsum()
    cumv = vol.cumsum()
    return cum / cumv


def fmt_price(x: float, decimals: int = 2) -> str:
    if np.isnan(x):
        return "—"
    return f"{x:,.{decimals}f}"


def color_for_action(action: str) -> str:
    a = action.upper()
    if "ENTRY" in a or "ACTIVE" in a:
        return "#34ff9a"
    if "HEADS" in a:
        return "#58d7ff"
    if "CAUTION" in a:
        return "#ffcc66"
    if "EXIT" in a or "RESET" in a:
        return "#ff5b6e"
    if "PUT" in a or "BEAR" in a:
        return "#ff5b6e"
    if "CALL" in a or "BULL" in a:
        return "#34ff9a"
    return "rgba(255,255,255,.85)"


def tone_class(bias: str, action: str) -> str:
    # feed color: action dominates, bias secondary
    a = action.upper()
    if "EXIT" in a or "RESET" in a:
        return "bad"
    if "CAUTION" in a:
        return "warn"
    if "ENTRY" in a or "ACTIVE" in a:
        return "good"
    if "HEADS" in a:
        return "warn"
    b = bias.upper()
    if "BULL" in b:
        return "good"
    if "BEAR" in b:
        return "bad"
    return "neutral"


def command_feed(text: str, cls: str):
    st.markdown(
        f"""
<div class="marquee-wrap">
  <div class="marquee marq-{cls}">
    {text} &nbsp; • &nbsp; {text} &nbsp; • &nbsp; {text}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# Data Fetching
# =========================
@st.cache_data(ttl=12)  # fast refresh; rely on TTL
def fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    t = yf.Ticker(symbol)
    df = t.history(interval=interval, period=period)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Datetime").reset_index()
    # Normalize columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=["Close"])
    return df


def get_last_price(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    if df_1m is not None and not df_1m.empty:
        return float(df_1m["Close"].iloc[-1]), df_1m["Datetime"].iloc[-1]
    if df_5m is not None and not df_5m.empty:
        return float(df_5m["Close"].iloc[-1]), df_5m["Datetime"].iloc[-1]
    return np.nan, pd.NaT


# =========================
# Signal Engine
# =========================
@dataclass
class EngineOut:
    price: float
    last_time: str
    bias: str
    direction: str
    regime: str
    action: str
    score: int
    vwap_5m: float
    atr_5m: float
    chop: float
    likely: float
    poss: float
    stretch: float
    invalid: float
    why: str


def compute_engine(df_1m: pd.DataFrame, df_5m: pd.DataFrame, mode: str) -> EngineOut:
    # Defaults
    price = np.nan
    last_time = "—"
    if df_5m is None:
        df_5m = pd.DataFrame()
    if df_1m is None:
        df_1m = pd.DataFrame()

    price, tstamp = get_last_price(df_1m, df_5m)
    if pd.notna(tstamp):
        last_time = str(tstamp)

    # Not enough data
    if df_5m.empty or len(df_5m) < 20:
        return EngineOut(
            price=price,
            last_time=last_time,
            bias="NEUTRAL",
            direction="WAIT",
            regime="RANGE",
            action="WAIT — NO EDGE",
            score=0,
            vwap_5m=np.nan,
            atr_5m=np.nan,
            chop=np.nan,
            likely=np.nan,
            poss=np.nan,
            stretch=np.nan,
            invalid=np.nan,
            why="Not enough 5m session data yet — let more candles print.",
        )

    # Build indicators (5m brain)
    d5 = df_5m.copy()
    d5["EMA9"] = ema(d5["Close"], 9)
    d5["EMA21"] = ema(d5["Close"], 21)
    d5["VWAP"] = vwap_intraday(d5)
    d5["ATR"] = atr(d5, 14)
    d5["CHOP"] = choppiness(d5, 14)

    vwap5 = float(d5["VWAP"].iloc[-1])
    atr5 = float(d5["ATR"].iloc[-1])
    chop5 = float(d5["CHOP"].iloc[-1])

    # Regime
    if np.isnan(chop5):
        regime = "MIXED"
    elif chop5 >= 60:
        regime = "RANGE"
    elif chop5 <= 45:
        regime = "TREND"
    else:
        regime = "MIXED"

    # Trend / slope
    ema9_now = float(d5["EMA9"].iloc[-1])
    ema21_now = float(d5["EMA21"].iloc[-1])
    ema9_prev = float(d5["EMA9"].iloc[-4]) if len(d5) >= 4 else ema9_now
    slope9 = ema9_now - ema9_prev

    # Bias logic (more “trader” less “professor”)
    bull_stack = (ema9_now > ema21_now) and (price > vwap5)
    bear_stack = (ema9_now < ema21_now) and (price < vwap5)

    if bull_stack and slope9 > 0:
        bias = "BULLISH"
        direction = "CALLS"
    elif bear_stack and slope9 < 0:
        bias = "BEARISH"
        direction = "PUTS"
    else:
        bias = "NEUTRAL"
        direction = "WAIT"

    # 1m trigger indicators (for entries/heads up)
    trigger_ok = False
    heads_up = False
    if not df_1m.empty and len(df_1m) >= 30:
        d1 = df_1m.copy()
        d1["EMA9"] = ema(d1["Close"], 9)
        d1["VWAP"] = vwap_intraday(d1)

        c = float(d1["Close"].iloc[-1])
        c1 = float(d1["Close"].iloc[-2])
        ema9_1 = float(d1["EMA9"].iloc[-1])
        vwap1 = float(d1["VWAP"].iloc[-1])

        # Reclaim / reject logic (fast)
        if direction == "CALLS":
            heads_up = (c > ema9_1 and c1 <= ema9_1) or (c > vwap1 and c1 <= vwap1)
            trigger_ok = (c > ema9_1 and c > vwap1)
        elif direction == "PUTS":
            heads_up = (c < ema9_1 and c1 >= ema9_1) or (c < vwap1 and c1 >= vwap1)
            trigger_ok = (c < ema9_1 and c < vwap1)
        else:
            # neutral: still allow heads up if strong reclaim/reject
            heads_up = abs(c - vwap1) <= max(atr5 * 0.25, 0.05)

    # Score (aggressive)
    score = 0

    # Bias contribution
    if bias == "BULLISH":
        score += 32
    elif bias == "BEARISH":
        score += 32
    else:
        score += 12

    # Alignment / momentum
    if direction != "WAIT":
        if (direction == "CALLS" and price > ema9_now) or (direction == "PUTS" and price < ema9_now):
            score += 18
        if (direction == "CALLS" and slope9 > 0) or (direction == "PUTS" and slope9 < 0):
            score += 12

    # Trigger
    if trigger_ok:
        score += 22
    elif heads_up:
        score += 10

    # Chop penalty (we still trade, but we’re not idiots)
    if not np.isnan(chop5):
        if chop5 >= 65:
            score -= 22
        elif chop5 >= 58:
            score -= 14
        elif chop5 >= 52:
            score -= 8

    score = int(max(0, min(100, score)))

    # Thresholds
    if mode == "FULL SEND":
        entry_th = 45
        caution_th = 32
    else:  # AGGRESSIVE
        entry_th = 55
        caution_th = 40

    # Invalidation level (simple, fast)
    if direction == "CALLS":
        invalid = min(vwap5, ema21_now) - max(atr5 * 0.25, 0.05)
    elif direction == "PUTS":
        invalid = max(vwap5, ema21_now) + max(atr5 * 0.25, 0.05)
    else:
        invalid = vwap5

    # Expected move “FROM HERE” (ATR-based)
    # Wider when FULL SEND, slightly tighter when AGGRESSIVE
    if np.isnan(atr5) or atr5 <= 0:
        atr5 = max(abs(price) * 0.001, 0.25)  # fallback

    if mode == "FULL SEND":
        m1, m2, m3 = 1.20, 2.00, 3.00
    else:
        m1, m2, m3 = 1.00, 1.70, 2.60

    if direction == "CALLS":
        likely = price + atr5 * m1
        poss = price + atr5 * m2
        stretch = price + atr5 * m3
    elif direction == "PUTS":
        likely = price - atr5 * m1
        poss = price - atr5 * m2
        stretch = price - atr5 * m3
    else:
        # Neutral: show both sides around price
        likely = price - atr5 * 0.9
        poss = price + atr5 * 0.9
        stretch = price + atr5 * 1.6

    # Action logic (more signals)
    why = ""
    if direction == "WAIT":
        if heads_up:
            action = "HEADS UP"
            why = "Neutral bias, but price is near key levels — watch for reclaim/reject."
        else:
            action = "WAIT — NO EDGE"
            why = "No clean alignment yet."
    else:
        # exit if invalid breached
        if (direction == "CALLS" and price < invalid) or (direction == "PUTS" and price > invalid):
            action = "EXIT / RESET"
            why = "Invalidation breached. Step away until alignment returns."
        else:
            if score >= entry_th and trigger_ok:
                action = f"ENTRY ACTIVE — {direction}"
                why = "Alignment + trigger confirmed. Trade the direction; manage risk."
            elif score >= caution_th and (trigger_ok or heads_up):
                action = f"CAUTION — {direction}"
                why = "Direction favored, but edge is thinner. Small size / wait for cleaner reclaim."
            elif heads_up:
                action = "HEADS UP"
                why = "Early conditions forming. Wait for confirmation."
            else:
                action = f"{direction} — WAIT"
                why = "Direction bias exists but trigger not confirmed."

    return EngineOut(
        price=price,
        last_time=last_time,
        bias=bias,
        direction=direction,
        regime=regime,
        action=action,
        score=score,
        vwap_5m=vwap5,
        atr_5m=atr5,
        chop=chop5,
        likely=likely,
        poss=poss,
        stretch=stretch,
        invalid=invalid,
        why=why,
    )


# =========================
# Universe / Assets
# =========================
# Your requested tickers + a “top traded” core list baked in.
ASSETS = {
    # Core
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "BITO": "BITO",
    "MSTR": "MSTR",
    "MSTU": "MSTU",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "AMD": "AMD",
    "PLTR": "PLTR",
    "SOFI": "SOFI",
    "GME": "GME",
    "AMC": "AMC",
    "RIOT": "RIOT",
    "MARA": "MARA",
    "CLSK": "CLSK",
    "IREN": "IREN",
    "NOK": "NOK",
    "U": "U",
    "ASTS": "ASTS",
    "OPEN": "OPEN",
    "HYMC": "HYMC",
    "XOM": "XOM",
    "OXY": "OXY",

    # Big “top-traded” defaults (common daily volume names)
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "META": "META",
    "GOOGL": "GOOGL",
    "NFLX": "NFLX",
    "AVGO": "AVGO",
    "INTC": "INTC",
    "BAC": "BAC",
    "NIO": "NIO",

    # Crypto
    "BTC": "CRYPTO:BTC-USD",
    "ETH": "CRYPTO:ETH-USD",
    "XRP": "CRYPTO:XRP-USD",
    "XLM": "CRYPTO:XLM-USD",
    "SOL": "CRYPTO:SOL-USD",
    "DOGE": "CRYPTO:DOGE-USD",
}


UNIVERSE_KEYS = list(ASSETS.keys())


# =========================
# Sidebar Controls
# =========================
st.sidebar.markdown("## Controls")

asset_label = st.sidebar.selectbox("Asset", UNIVERSE_KEYS, index=UNIVERSE_KEYS.index("SPY"))
asset_key = ASSETS[asset_label]
symbol = yf_symbol(asset_key)

mode = st.sidebar.radio("Mode", ["AGGRESSIVE", "FULL SEND"], index=0)

auto = st.sidebar.toggle("Auto-refresh", value=True)
refresh_seconds = st.sidebar.selectbox("Refresh seconds", [10, 15, 20, 30, 45, 60], index=0)

refresh_now = st.sidebar.button("🔁 Refresh now", use_container_width=True)

# Auto-refresh trigger
if auto and _AUTORF_AVAILABLE:
    st_autorefresh(interval=refresh_seconds * 1000, key="autorf")
elif auto and not _AUTORF_AVAILABLE:
    st.sidebar.caption("Auto-refresh not available (missing streamlit-autorefresh). Use Refresh now.")

# Manual refresh rerun
if refresh_now:
    st.cache_data.clear()
    st.rerun()


# =========================
# Header
# =========================
st.title("Lockout Signals • Command Center")


# =========================
# Fetch Data
# =========================
# 5m "brain" — 1d is enough for session VWAP + ATR.
df_5m = fetch_intraday(symbol, interval="5m", period="1d")

# 1m "trigger" — 1d sometimes fails; fallback to 5d
df_1m = fetch_intraday(symbol, interval="1m", period="1d")
if df_1m.empty:
    df_1m = fetch_intraday(symbol, interval="1m", period="5d")

out = compute_engine(df_1m, df_5m, mode=mode)

mkt = market_status(asset_key)

# =========================
# Top Movers Sidebar (Universe)
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### Top Movers (Universe)")

def movers_table(universe_keys: list[str]) -> pd.DataFrame:
    rows = []
    # Pull 5m for each symbol (fast-ish; TTL cache helps)
    for k in universe_keys:
        sym = yf_symbol(ASSETS[k])
        try:
            d = fetch_intraday(sym, interval="5m", period="1d")
            if d.empty:
                continue
            first = float(d["Close"].iloc[0])
            last = float(d["Close"].iloc[-1])
            if first <= 0:
                continue
            pct = (last / first - 1.0) * 100.0
            rows.append([k, pct, last])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Ticker", "%", "Last"])
    df = pd.DataFrame(rows, columns=["Ticker", "%", "Last"]).sort_values("%", ascending=False)
    df["%"] = df["%"].map(lambda x: f"{x:+.2f}%")
    df["Last"] = df["Last"].map(lambda x: f"{x:,.2f}")
    return df

movers = movers_table(UNIVERSE_KEYS)

col_up, col_dn = st.sidebar.columns(2)
with col_up:
    st.caption("Top 10 Up")
    st.dataframe(movers.head(10), use_container_width=True, hide_index=True)
with col_dn:
    st.caption("Top 10 Down")
    st.dataframe(movers.tail(10).sort_values("%", ascending=True), use_container_width=True, hide_index=True)


# =========================
# Command Feed (always on)
# =========================
feed = (
    f"ACTION: {out.action} • BIAS: {out.bias} — {out.direction} • "
    f"STATUS: {mkt} • REGIME: {out.regime} • SCORE: {out.score}/100 • "
    f"INVALID: {fmt_price(out.invalid)} • PRICE: {fmt_price(out.price)}"
)
command_feed(feed, tone_class(out.bias, out.action))


# =========================
# Main Command Card
# =========================
subhead = f"{asset_label} • 5m Brain / 1m Trigger"
action_color = color_for_action(out.action)

st.markdown(
    f"""
<div class="cc-card">
  <div class="k_subhead">{subhead}</div>

  <div class="k_price" style="color: {action_color};">{fmt_price(out.price)}</div>
  <div class="k_action" style="color: {action_color};">{out.action}</div>

  <div class="k_chips">
    <div class="k_chip">{out.bias} — {out.direction}</div>
    <div class="k_chip">{mkt}</div>
    <div class="k_chip">REGIME: {out.regime}</div>
    <div class="k_chip">SCORE: {out.score}/100</div>
    <div class="k_chip">MODE: {mode}</div>
  </div>

  <div class="k_small"><b>EXPECTED MOVE (FROM HERE)</b></div>
  <div class="k_small" style="margin-top:6px;">
    <span style="color:#34ff9a;"><b>LIKELY {fmt_price(out.likely)}</b></span>
    &nbsp; | &nbsp;
    <span style="color:rgba(255,255,255,.92);"><b>POSS {fmt_price(out.poss)}</b></span>
    &nbsp; | &nbsp;
    <span style="color:#34ff9a;"><b>STRETCH {fmt_price(out.stretch)}</b></span>
  </div>

  <div class="k_small" style="margin-top:10px;">
    Invalid: <b>{fmt_price(out.invalid)}</b> • Last update: <b>{out.last_time}</b>
  </div>

  <div class="k_small" style="margin-top:10px;">
    {out.why}
  </div>

  <div class="kpi-row">
    <div class="kpi">
      <div class="label">BIAS</div>
      <div class="value">{out.bias}</div>
      <div class="delta">Direction: {out.direction}</div>
    </div>
    <div class="kpi">
      <div class="label">VWAP (5m)</div>
      <div class="value">{fmt_price(out.vwap_5m)}</div>
      <div class="delta">Anchor level for reclaim/reject</div>
    </div>
    <div class="kpi">
      <div class="label">ATR (5m)</div>
      <div class="value">{fmt_price(out.atr_5m)}</div>
      <div class="delta">Used for “Expected Move”</div>
    </div>
    <div class="kpi">
      <div class="label">CHOP</div>
      <div class="value">{("—" if np.isnan(out.chop) else f"{out.chop:.0f}/100")}</div>
      <div class="delta">Higher = chop/range</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Decision-support only. Not financial advice. Trade your plan.")


# =========================
# Optional mini chart (kept light)
# =========================
with st.expander("Show chart (Close / EMA9 / VWAP)", expanded=False):
    if df_1m is not None and not df_1m.empty and len(df_1m) >= 20:
        d1 = df_1m.copy()
        d1["EMA9"] = ema(d1["Close"], 9)
        d1["VWAP"] = vwap_intraday(d1)
        d1 = d1.tail(240)  # keep it snappy

        chart_df = d1.set_index("Datetime")[["Close", "EMA9", "VWAP"]]
        st.line_chart(chart_df)
    elif df_5m is not None and not df_5m.empty:
        d5 = df_5m.copy()
        d5["EMA9"] = ema(d5["Close"], 9)
        d5["VWAP"] = vwap_intraday(d5)
        chart_df = d5.set_index("Datetime")[["Close", "EMA9", "VWAP"]].tail(200)
        st.line_chart(chart_df)
    else:
        st.info("No chart data available yet.")


# =========================
# Debug panel (optional)
# =========================
with st.expander("Engine debug (optional)", expanded=False):
    st.write("Symbol:", symbol)
    st.write("Data points 5m:", len(df_5m))
    st.write("Data points 1m:", len(df_1m))
    st.write(out)