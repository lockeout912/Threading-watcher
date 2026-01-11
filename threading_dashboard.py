import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time, timezone, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lockout Signals • SPY + BTC", layout="wide")

# ============================================================
# SAFE STYLES
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1100px; }
.small-muted { opacity: 0.75; font-size: 0.85rem; }
.big-price { font-size: 3.1rem; font-weight: 900; line-height: 1.05; margin-top: -6px; }
.hr { height: 1px; background: rgba(255,255,255,0.14); margin: 0.75rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
ASSETS = {
    "SPY": {"ticker": "SPY", "type": "equity"},
    "BTC": {"ticker": "BTC-USD", "type": "crypto"},
}

INTERVAL = "5m"
FETCH_PERIOD = "5d"

BB_PERIOD = 20
BB_STD = 2
EMA_PERIOD = 9
ATR_PERIOD = 14
RVOL_LOOKBACK = 20

VWAP_CONFIRM_BARS = 2
RESET_MINUTES = 15

HEADSUP_VWAP_ATR_BAND = 0.50
RVOL_HEADSUP_MIN = 1.00
RVOL_ENTRY_MIN = 1.30

BANDWIDTH_EXPAND_BARS = 3
BANDWIDTH_MIN_LIFT = 1.05

# ============================================================
# HELPERS
# ============================================================
def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def fmt(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:,.2f}"

def fmt_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:+.2f}%"

def fmt_delta(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:,.2f}"

def pct_change(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return (a - b) / b * 100.0

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def bias_label(bias: str) -> str:
    if bias == "Bullish":
        return "BULLISH → CALLS"
    if bias == "Bearish":
        return "BEARISH → PUTS"
    return "NEUTRAL → STAND DOWN"

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=25)
def fetch_intraday_yf(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=FETCH_PERIOD, interval=INTERVAL)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=5)
def fetch_spy_quote_yf(ticker: str) -> dict:
    """
    Best-effort 'quote' via yfinance. Not true real-time.
    Returns: {price, fetched_at_utc, source}
    """
    t0 = now_utc()
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            p = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            p = safe_float(p)
            if not np.isnan(p):
                return {"price": p, "fetched_at_utc": t0, "source": "yfinance.fast_info"}
        info = getattr(t, "info", {}) or {}
        p = safe_float(info.get("regularMarketPrice"))
        if not np.isnan(p):
            return {"price": p, "fetched_at_utc": t0, "source": "yfinance.info"}
    except Exception:
        pass
    return {"price": np.nan, "fetched_at_utc": t0, "source": "yfinance.unavailable"}

@st.cache_data(ttl=2)
def fetch_btc_quote_coinbase() -> dict:
    """
    Coinbase public endpoint (fast).
    Returns: {price, fetched_at_utc, source}
    """
    t0 = now_utc()
    url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    try:
        r = requests.get(url, timeout=2.5, headers={"User-Agent": "LockoutSignals/1.0"})
        if r.status_code == 200:
            j = r.json()
            p = safe_float(j.get("price"))
            if not np.isnan(p):
                return {"price": p, "fetched_at_utc": t0, "source": "coinbase.exchange.ticker"}
    except Exception:
        pass

    url2 = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    try:
        r = requests.get(url2, timeout=2.5, headers={"User-Agent": "LockoutSignals/1.0"})
        if r.status_code == 200:
            j = r.json()
            amt = (((j or {}).get("data") or {}).get("amount"))
            p = safe_float(amt)
            if not np.isnan(p):
                return {"price": p, "fetched_at_utc": t0, "source": "coinbase.v2.spot"}
    except Exception:
        pass

    return {"price": np.nan, "fetched_at_utc": t0, "source": "coinbase.unavailable"}

# ============================================================
# SESSION FILTER
# ============================================================
def to_tz(dt_index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    idx = dt_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

def filter_session(df: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df, ""

    df = df.copy()
    if asset_type == "equity":
        df.index = to_tz(df.index, "America/New_York")
        session_date = df.index[-1].date()
        session_id = session_date.isoformat()
        start = datetime.combine(session_date, time(9, 30)).replace(tzinfo=df.index.tz)
        end = datetime.combine(session_date, time(16, 0)).replace(tzinfo=df.index.tz)
        return df[(df.index >= start) & (df.index <= end)].copy(), session_id

    df.index = to_tz(df.index, "UTC")
    session_date = df.index[-1].date()
    session_id = session_date.isoformat()
    start = datetime.combine(session_date, time(0, 0)).replace(tzinfo=df.index.tz)
    end = datetime.combine(session_date, time(23, 59)).replace(tzinfo=df.index.tz)
    return df[(df.index >= start) & (df.index <= end)].copy(), session_id

# ============================================================
# INDICATORS
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty or df.shape[0] < max(BB_PERIOD, ATR_PERIOD, RVOL_LOOKBACK, EMA_PERIOD) + 5:
        return df

    df["EMA9"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA9_slope"] = df["EMA9"] - df["EMA9"].shift(3)

    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["TR"] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    pv = tp * vol
    cum_vol = vol.cumsum().replace(0, np.nan)
    df["VWAP"] = pv.cumsum() / cum_vol

    vol_ma = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / vol_ma

    mid = df["Close"].rolling(BB_PERIOD).mean()
    sd = df["Close"].rolling(BB_PERIOD).std(ddof=0)
    upper = mid + BB_STD * sd
    lower = mid - BB_STD * sd
    df["BB_bw"] = (upper - lower) / mid.replace(0, np.nan)
    return df

def bandwidth_state(df: pd.DataFrame) -> tuple[str, float]:
    if df.empty or "BB_bw" not in df.columns:
        return "Chop", np.nan
    bw = df["BB_bw"].dropna()
    if bw.shape[0] < 30:
        return "Chop", safe_float(df["BB_bw"].iloc[-1])
    bw_now = safe_float(bw.iloc[-1])
    last_vals = bw.tail(BANDWIDTH_EXPAND_BARS).values
    rising = all(last_vals[i] > last_vals[i - 1] for i in range(1, len(last_vals)))
    recent_min = safe_float(bw.tail(20).min())
    lifted = bw_now >= (recent_min * BANDWIDTH_MIN_LIFT)
    return ("Trend" if (rising and lifted) else "Chop"), bw_now

def vwap_confirmed_side(df: pd.DataFrame) -> str:
    if df.empty or "VWAP" not in df.columns:
        return "Mixed/None"
    closes = df["Close"].tail(VWAP_CONFIRM_BARS)
    vwaps = df["VWAP"].tail(VWAP_CONFIRM_BARS)
    if closes.shape[0] < VWAP_CONFIRM_BARS:
        return "Mixed/None"
    if (closes > vwaps).all():
        return "Above"
    if (closes < vwaps).all():
        return "Below"
    return "Mixed/None"

def momentum_state(df: pd.DataFrame) -> str:
    if df.empty or "EMA9_slope" not in df.columns:
        return "Mixed"
    s = safe_float(df["EMA9_slope"].iloc[-1])
    if np.isnan(s):
        return "Mixed"
    return "Up" if s > 0 else ("Down" if s < 0 else "Flat")

def compute_levels(df: pd.DataFrame, direction: str) -> dict:
    last = df.iloc[-1]
    current = safe_float(last["Close"])
    vwap = safe_float(last.get("VWAP", np.nan))
    atr = safe_float(last.get("ATR", np.nan))
    if np.isnan(atr) or atr <= 0:
        atr = max(current * 0.003, 0.50)

    if direction == "Bullish":
        entry = vwap
        target_low = entry + 1.5 * atr
        target_high = entry + 3.0 * atr
        exit_if = entry - 1.25 * atr
    else:
        entry = vwap
        target_low = entry - 3.0 * atr
        target_high = entry - 1.5 * atr
        exit_if = entry + 1.25 * atr

    lo = float(min(target_low, target_high))
    hi = float(max(target_low, target_high))
    usd_low = lo - current
    usd_high = hi - current
    pct_low = (usd_low / current) * 100 if current else np.nan
    pct_high = (usd_high / current) * 100 if current else np.nan

    return {
        "current": current, "vwap": vwap, "atr": atr,
        "entry": entry, "target_low": lo, "target_high": hi, "exit_if": exit_if,
        "usd_low": usd_low, "usd_high": usd_high, "pct_low": pct_low, "pct_high": pct_high
    }

# ============================================================
# STATE MACHINE
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = {}
if "quote_state" not in st.session_state:
    st.session_state.quote_state = {}

def get_state(key: str) -> dict:
    if key not in st.session_state.state:
        st.session_state.state[key] = {
            "status": "STAND DOWN",
            "active_direction": None,
            "reset_until": None,
            "last_session_id": None,
            "last_exit_reason": None
        }
    return st.session_state.state[key]

def set_state(key: str, **kwargs):
    s = get_state(key)
    for k, v in kwargs.items():
        s[k] = v
    st.session_state.state[key] = s

def status_box(sts: str):
    if sts == "ENTRY ACTIVE": return st.success
    if sts in ("CAUTION", "HEADS UP"): return st.warning
    if sts == "EXIT / RESET": return st.error
    return st.info

def decision_text(sts: str, direction: str | None, exit_reason: str | None):
    if sts == "ENTRY ACTIVE" and direction == "Bullish":
        return "BUY CALLS (still valid)", "Entry is live. Stay long while price holds ABOVE VWAP. Respect Exit-If."
    if sts == "ENTRY ACTIVE" and direction == "Bearish":
        return "BUY PUTS (still valid)", "Entry is live. Stay short while price holds BELOW VWAP. Respect Exit-If."
    if sts == "CAUTION":
        return "CAUTION (late entry risk)", "Edge is weakening. If entering now, size down or wait for re-confirmation."
    if sts == "HEADS UP":
        return "HEADS UP (not confirmed)", "Setup building. Do NOT enter until ENTRY ACTIVE prints."
    if sts == "EXIT / RESET":
        why = f" — {exit_reason}" if exit_reason else ""
        return f"EXIT / STAND DOWN{why}", "Move is done. No chasing. Wait for the next cycle."
    if sts == "RESET":
        return "RESET (cooldown)", "We’re cooling off to avoid chop whipsaws."
    if sts == "WAITING":
        return "WAITING (setup forming)", "Conditions are close, but confirmation isn’t clean yet."
    return "STAND DOWN (chop)", "No edge right now. Preserve capital and wait."

# ============================================================
# UI CONTROLS
# ============================================================
st.title("Lockout Signals • SPY + BTC")

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.0, 1.1, 1.1])
with ctrl1:
    asset_key = st.selectbox("Asset", list(ASSETS.keys()), index=0)
with ctrl2:
    st.button("🔄 Refresh now")
with ctrl3:
    auto = st.toggle("Auto-refresh", value=True)
with ctrl4:
    refresh_s = st.selectbox("Refresh rate", [5, 10, 20, 60], index=1, format_func=lambda x: f"{x}s")

if auto:
    try:
        st.autorefresh(interval=int(refresh_s * 1000), key="autorefresh")
    except Exception:
        st.caption("Auto-refresh not supported on this Streamlit version; use Refresh now.")

asset = ASSETS[asset_key]
ticker = asset["ticker"]
atype = asset["type"]

# ============================================================
# LOAD CANDLES (for indicators)
# ============================================================
df_raw = fetch_intraday_yf(ticker)
df_sess, session_id = filter_session(df_raw, atype)
df = compute_indicators(df_sess)

if df.empty or df.shape[0] < 30:
    st.warning("Not enough session data yet — let more 5-minute candles print.")
    st.stop()

# Candle-close (confirmed)
candle_close = safe_float(df["Close"].iloc[-1])
last_candle_time = df.index[-1]
session_open = safe_float(df["Open"].iloc[0]) if df.shape[0] else np.nan

# ============================================================
# LIVE QUOTE (as live as possible for $0)
# ============================================================
if asset_key == "BTC":
    quote = fetch_btc_quote_coinbase()
else:
    quote = fetch_spy_quote_yf("SPY")

st.session_state.quote_state[asset_key] = quote

current = quote["price"]
if np.isnan(current):
    current = candle_close  # fallback

freshness_sec = (now_utc() - quote["fetched_at_utc"]).total_seconds()

# ============================================================
# ENGINE VALUES
# ============================================================
last = df.iloc[-1]
vwap = safe_float(last.get("VWAP", np.nan))
atr = safe_float(last.get("ATR", np.nan))
rvol = safe_float(last.get("RVOL", np.nan))

mkt_state, bw_now = bandwidth_state(df)
vwap_side = vwap_confirmed_side(df)
mom = momentum_state(df)

# Bias vs VWAP
bias = "Mixed"
if not np.isnan(current) and not np.isnan(vwap):
    bias = "Bullish" if current >= vwap else "Bearish"
bias_text = bias_label(bias)

# HOD/LOD
hod = safe_float(df["High"].max())
lod = safe_float(df["Low"].min())
dist_to_hod = hod - current if not np.isnan(hod) else np.nan
dist_to_lod = current - lod if not np.isnan(lod) else np.nan

# Move meters
delta_last = current - candle_close
pct_last = pct_change(current, candle_close)
delta_open = current - session_open
pct_open = pct_change(current, session_open)

# Heads Up logic (early build — not tradeable)
vol_prev = safe_float(df["Volume"].iloc[-2]) if df.shape[0] >= 2 else np.nan
vol_now = safe_float(df["Volume"].iloc[-1])
near_vwap = (not np.isnan(atr) and not np.isnan(current) and not np.isnan(vwap) and abs(current - vwap) <= (HEADSUP_VWAP_ATR_BAND * atr))

momentum_improving = False
if "EMA9_slope" in df.columns and df["EMA9_slope"].shape[0] >= 5:
    s_now = safe_float(df["EMA9_slope"].iloc[-1])
    s_prev = safe_float(df["EMA9_slope"].iloc[-3])
    momentum_improving = (not np.isnan(s_now) and not np.isnan(s_prev) and s_now > s_prev)

participation_rising = (not np.isnan(vol_now) and not np.isnan(vol_prev) and vol_now > vol_prev and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN))
bw_series = df.get("BB_bw", pd.Series(dtype=float)).dropna()
bw_improving = (bw_series.shape[0] >= 3 and safe_float(bw_series.iloc[-1]) > safe_float(bw_series.iloc[-2]))

heads_up = (mkt_state == "Chop" and near_vwap and momentum_improving and participation_rising and bw_improving)

# Confirmed entries (tradeable)
confirmed_long = (
    bias == "Bullish"
    and mkt_state == "Trend"
    and vwap_side == "Above"
    and mom == "Up"
    and (not np.isnan(rvol) and rvol >= RVOL_ENTRY_MIN)
)

confirmed_short = (
    bias == "Bearish"
    and mkt_state == "Trend"
    and vwap_side == "Below"
    and mom == "Down"
    and (not np.isnan(rvol) and rvol >= RVOL_ENTRY_MIN)
)

levels_long = compute_levels(df, "Bullish")
levels_short = compute_levels(df, "Bearish")

def exit_triggered(active_dir: str) -> tuple[bool, str]:
    if active_dir == "Bullish":
        if vwap_side == "Below":
            return True, "VWAP flip confirmed"
        if current <= levels_long["exit_if"]:
            return True, "Emergency ATR stop"
    if active_dir == "Bearish":
        if vwap_side == "Above":
            return True, "VWAP flip confirmed"
        if current >= levels_short["exit_if"]:
            return True, "Emergency ATR stop"
    return False, ""

def caution_for_later(direction: str) -> bool:
    if direction == "Bullish":
        still = current >= vwap
        weak = 0
        if mkt_state != "Trend": weak += 1
        if mom != "Up": weak += 1
        if np.isnan(rvol) or rvol < RVOL_ENTRY_MIN: weak += 1
        if vwap_side != "Above": weak += 1
        return bool(still and weak >= 2)
    if direction == "Bearish":
        still = current <= vwap
        weak = 0
        if mkt_state != "Trend": weak += 1
        if mom != "Down": weak += 1
        if np.isnan(rvol) or rvol < RVOL_ENTRY_MIN: weak += 1
        if vwap_side != "Below": weak += 1
        return bool(still and weak >= 2)
    return False

# Session reset detection
s = get_state(asset_key)
if session_id and s.get("last_session_id") != session_id:
    set_state(asset_key, status="STAND DOWN", active_direction=None, reset_until=None, last_session_id=session_id, last_exit_reason=None)
s = get_state(asset_key)

# State machine
in_reset = s["reset_until"] is not None and now_utc() < s["reset_until"]
if in_reset:
    set_state(asset_key, status="RESET")
else:
    active = s.get("active_direction")

    if active in ("Bullish", "Bearish"):
        do_exit, reason = exit_triggered(active)
        if do_exit:
            set_state(asset_key, status="EXIT / RESET", active_direction=None,
                      reset_until=now_utc() + timedelta(minutes=RESET_MINUTES),
                      last_exit_reason=reason)
        else:
            if active == "Bullish":
                if confirmed_long:
                    set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bullish")
                elif caution_for_later("Bullish"):
                    set_state(asset_key, status="CAUTION", active_direction="Bullish")
                else:
                    set_state(asset_key, status="WAITING", active_direction="Bullish")
            else:
                if confirmed_short:
                    set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bearish")
                elif caution_for_later("Bearish"):
                    set_state(asset_key, status="CAUTION", active_direction="Bearish")
                else:
                    set_state(asset_key, status="WAITING", active_direction="Bearish")
    else:
        if confirmed_long:
            set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bullish")
        elif confirmed_short:
            set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bearish")
        else:
            if heads_up:
                set_state(asset_key, status="HEADS UP", active_direction=None)
            elif mkt_state == "Trend":
                set_state(asset_key, status="WAITING", active_direction=None)
            else:
                set_state(asset_key, status="STAND DOWN", active_direction=None)

s = get_state(asset_key)
status = s["status"]
active_dir = s.get("active_direction")
exit_reason = s.get("last_exit_reason")

# Choose direction for level display even when neutral
direction = active_dir if active_dir in ("Bullish", "Bearish") else ("Bullish" if bias == "Bullish" else "Bearish")
levels = compute_levels(df, "Bullish" if direction == "Bullish" else "Bearish")

entry_line = f"Above {fmt(levels['entry'])}" if direction == "Bullish" else f"Below {fmt(levels['entry'])}"
exit_line  = f"Below {fmt(levels['exit_if'])}" if direction == "Bullish" else f"Above {fmt(levels['exit_if'])}"

range_label = "Upside Range" if direction == "Bullish" else "Downside Range"

title, instruction = decision_text(status, active_dir, exit_reason)

# ============================================================
# TABS
# ============================================================
tab_overview, tab_engine, tab_raw = st.tabs(["📍 Overview", "🧠 Engine", "📊 Raw Table"])

with tab_overview:
    left, right = st.columns([1.35, 1.0])

    with left:
        st.markdown(f"**{asset_key} • {ticker} • {INTERVAL} • Session {session_id}**")
        st.markdown(f'<div class="big-price">{fmt(current)}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="small-muted">'
            f'Quote source: {quote["source"]} • Quote freshness: {freshness_sec:.1f}s<br>'
            f'Candle Close: {fmt(candle_close)} • Last candle time: {last_candle_time}'
            f'</div>',
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)
        m1.metric("Move vs last candle", fmt_delta(delta_last), fmt_pct(pct_last))
        m2.metric("Move vs session open", fmt_delta(delta_open), fmt_pct(pct_open))

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("Micro Trend")
        spark = df.tail(160).copy()
        cols = [c for c in ["Close", "VWAP", "EMA9"] if c in spark.columns]
        spark = spark[cols].dropna()
        if spark.empty:
            st.info("Sparkline not available yet.")
        else:
            st.line_chart(spark, height=240)

    with right:
        st.markdown("### Decision Ribbon")
        # My preferred layout: Bias → Status → Action
        st.write(f"**Directional Bias:** `{bias_text}`")
        st.write(f"**Status:** `{status}`")
        st.write(f"**Action:** {title}")
        status_box(status)(instruction)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Entry Rule", entry_line)
        k2.metric(range_label, f"{fmt(levels['target_low'])} → {fmt(levels['target_high'])}")
        k3.metric("Exit-If", exit_line)

        st.caption(
            f"{range_label}: {fmt_pct(levels['pct_low'])} to {fmt_pct(levels['pct_high'])}  "
            f"({fmt_delta(levels['usd_low'])} to {fmt_delta(levels['usd_high'])})"
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        h1.metric("Session HOD", fmt(hod), f"{fmt_delta(-dist_to_hod)} from price" if not np.isnan(dist_to_hod) else "—")
        h2.metric("Session LOD", fmt(lod), f"{fmt_delta(dist_to_lod)} above LOD" if not np.isnan(dist_to_lod) else "—")

with tab_engine:
    st.subheader("Engine Readout (why the ribbon says what it says)")
    st.write(f"**Directional Bias:** `{bias_text}`")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Bias (vs VWAP)", bias)
    e2.metric("Regime", mkt_state)
    e3.metric("VWAP Confirm", vwap_side)
    e4.metric("Momentum (EMA9 slope)", mom)

    e5, e6, e7, e8 = st.columns(4)
    e5.metric("VWAP", fmt(vwap))
    e6.metric("ATR", fmt(atr))
    e7.metric("RVOL", "—" if np.isnan(rvol) else f"{rvol:.2f}")
    e8.metric("BB Width", "—" if np.isnan(bw_now) else f"{bw_now:.4f}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.write("**Signal Glossary**")
    st.write("- **HEADS UP** = early build (do not enter)")
    st.write("- **ENTRY ACTIVE** = tradable edge now")
    st.write("- **CAUTION** = late entry risk / weakening")
    st.write("- **EXIT/RESET** = move done, don’t chase")
    st.write("- **RESET** = cooldown to avoid chop")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.write("**Reality Check (data latency)**")
    st.write(f"- Quote freshness right now: **{freshness_sec:.1f}s** (refresh setting + network + data source)")
    st.write("- **BTC:** Coinbase is typically near-live.")
    st.write("- **SPY:** yfinance can lag. We show freshness so you always know what you’re looking at.")

with tab_raw:
    st.subheader("Raw Session Table (last 80 candles)")
    raw = df.tail(80).copy()
    keep = ["Open", "High", "Low", "Close", "Volume", "VWAP", "EMA9", "ATR", "RVOL", "BB_bw"]
    keep = [c for c in keep if c in raw.columns]
    st.dataframe(raw[keep], use_container_width=True)

st.caption("Not financial advice. Decision-support engine only.")