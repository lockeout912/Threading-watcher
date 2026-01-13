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

st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1100px; }
.small-muted { opacity: 0.75; font-size: 0.85rem; }
.big-price { font-size: 3.1rem; font-weight: 900; line-height: 1.05; margin-top: -6px; }
.hr { height: 1px; background: rgba(255,255,255,0.14); margin: 0.75rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS (TUNED)
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

# Old conservative behavior: 2-bar confirm. We’ll reduce early session.
VWAP_CONFIRM_BARS = 1

# Cooldown stays
RESET_MINUTES = 10

# Early heads-up / bias signal gates (LOOSER)
RVOL_HEADSUP_MIN = 0.85     # was 1.0
RVOL_ENTRY_MIN = 1.05       # was 1.3

# How far from VWAP is “too extended” (CAUTION)
EXTENSION_ATR = 1.25

# Bollinger bandwidth settings remain
BANDWIDTH_EXPAND_BARS = 2
BANDWIDTH_MIN_LIFT = 1.02

# Opening drive window (equities)
OPEN_DRIVE_MINUTES = 60

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
@st.cache_data(ttl=20)
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
    if df.empty:
        return df

    df["EMA9"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA9_slope"] = df["EMA9"] - df["EMA9"].shift(2)  # faster slope read

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
    if bw.shape[0] < 8:
        return "Chop", safe_float(df["BB_bw"].iloc[-1])
    bw_now = safe_float(bw.iloc[-1])
    last_vals = bw.tail(BANDWIDTH_EXPAND_BARS).values
    rising = all(last_vals[i] > last_vals[i - 1] for i in range(1, len(last_vals)))
    recent_min = safe_float(bw.tail(20).min()) if bw.shape[0] >= 20 else safe_float(bw.min())
    lifted = bw_now >= (recent_min * BANDWIDTH_MIN_LIFT) if not np.isnan(recent_min) else False
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
        target_low = entry + 1.0 * atr
        target_high = entry + 2.5 * atr
        exit_if = entry - 1.0 * atr
    else:
        entry = vwap
        target_low = entry - 2.5 * atr
        target_high = entry - 1.0 * atr
        exit_if = entry + 1.0 * atr

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
    if sts in ("CAUTION", "OPEN DRIVE", "HEADS UP"): return st.warning
    if sts == "EXIT / RESET": return st.error
    return st.info

def decision_text(sts: str, direction: str | None, exit_reason: str | None):
    if sts == "OPEN DRIVE" and direction == "Bullish":
        return "BULLISH OPEN DRIVE (CALLS favored)", "Early-session trend impulse. Prefer CALL setups. Wait for pullback or ENTRY ACTIVE to size up."
    if sts == "OPEN DRIVE" and direction == "Bearish":
        return "BEARISH OPEN DRIVE (PUTS favored)", "Early-session dump impulse. Prefer PUT setups. Wait for bounce or ENTRY ACTIVE to size up."
    if sts == "ENTRY ACTIVE" and direction == "Bullish":
        return "BUY CALLS (ENTRY ACTIVE)", "Tradable edge is live. Stay long while price holds ABOVE VWAP. Respect Exit-If."
    if sts == "ENTRY ACTIVE" and direction == "Bearish":
        return "BUY PUTS (ENTRY ACTIVE)", "Tradable edge is live. Stay short while price holds BELOW VWAP. Respect Exit-If."
    if sts == "CAUTION":
        return "CAUTION (late entry risk)", "Edge is weakening or extended. Size down or wait for re-confirmation."
    if sts == "HEADS UP":
        return "HEADS UP (building)", "Setup forming. Direction bias is present, but confirmation isn’t clean yet."
    if sts == "EXIT / RESET":
        why = f" — {exit_reason}" if exit_reason else ""
        return f"EXIT / STAND DOWN{why}", "Move is done. No chasing. Wait for the next cycle."
    if sts == "RESET":
        return "RESET (cooldown)", "Cooling off to avoid chop whipsaws."
    if sts == "WAITING":
        return "WAITING (trend forming)", "Close, but we need one more clean confirmation."
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
# LOAD DATA
# ============================================================
df_raw = fetch_intraday_yf(ticker)
df_sess, session_id = filter_session(df_raw, atype)
df = compute_indicators(df_sess)

# Loosen early requirement: we can speak by candle #3 now
if df.empty or df.shape[0] < 3:
    st.warning("Not enough session data yet — let a couple 5-minute candles print.")
    st.stop()

candle_close = safe_float(df["Close"].iloc[-1])
last_candle_time = df.index[-1]
session_open = safe_float(df["Open"].iloc[0]) if df.shape[0] else np.nan

# ============================================================
# LIVE QUOTE
# ============================================================
if asset_key == "BTC":
    quote = fetch_btc_quote_coinbase()
else:
    quote = fetch_spy_quote_yf("SPY")

st.session_state.quote_state[asset_key] = quote

current = quote["price"]
if np.isnan(current):
    current = candle_close

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

bias = "Mixed"
if not np.isnan(current) and not np.isnan(vwap):
    bias = "Bullish" if current >= vwap else "Bearish"
bias_text = bias_label(bias)

# OPEN DRIVE detection (equities only)
is_open_drive_window = False
if atype == "equity":
    try:
        ts = df.index[-1]  # NY tz already
        market_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
        is_open_drive_window = (ts <= (market_open + timedelta(minutes=OPEN_DRIVE_MINUTES)))
    except Exception:
        is_open_drive_window = False

# Impulse detection (range expansion + volume)
range_now = safe_float(last["High"] - last["Low"])
atr_use = atr if not np.isnan(atr) and atr > 0 else max(candle_close * 0.003, 0.50)
impulse = range_now >= (0.60 * atr_use)
vol_now = safe_float(last.get("Volume", np.nan))
vol_ma = safe_float(df["Volume"].rolling(10).mean().iloc[-1]) if "Volume" in df.columns and df.shape[0] >= 10 else np.nan
vol_surge = (not np.isnan(vol_now) and (np.isnan(vol_ma) or vol_now >= 1.10 * vol_ma))

# Extended?
extended = (not np.isnan(vwap) and not np.isnan(atr_use) and abs(current - vwap) >= (EXTENSION_ATR * atr_use))

# "Trend-like" loosening: allow Trend if bandwidth rising and VWAP side clean
trend_like = (mkt_state == "Trend") or (vwap_side in ("Above", "Below") and mom in ("Up", "Down") and (not np.isnan(bw_now)))

# ENTRY CONFIRM (LOOSER)
confirmed_long = (
    bias == "Bullish"
    and trend_like
    and vwap_side == "Above"
    and mom == "Up"
    and (np.isnan(rvol) or rvol >= RVOL_ENTRY_MIN)
)

confirmed_short = (
    bias == "Bearish"
    and trend_like
    and vwap_side == "Below"
    and mom == "Down"
    and (np.isnan(rvol) or rvol >= RVOL_ENTRY_MIN)
)

# OPEN DRIVE (BIAS) if impulse + vwap alignment + early window
open_drive_long = is_open_drive_window and bias == "Bullish" and impulse and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN)
open_drive_short = is_open_drive_window and bias == "Bearish" and impulse and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN)

levels_long = compute_levels(df, "Bullish")
levels_short = compute_levels(df, "Bearish")

def exit_triggered(active_dir: str) -> tuple[bool, str]:
    if active_dir == "Bullish":
        if vwap_side == "Below":
            return True, "VWAP flip"
        if current <= levels_long["exit_if"]:
            return True, "ATR stop"
    if active_dir == "Bearish":
        if vwap_side == "Above":
            return True, "VWAP flip"
        if current >= levels_short["exit_if"]:
            return True, "ATR stop"
    return False, ""

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
            # Stay active but mark caution if extended or weakening
            if extended:
                set_state(asset_key, status="CAUTION", active_direction=active)
            else:
                set_state(asset_key, status="ENTRY ACTIVE", active_direction=active)
    else:
        # Entry checks
        if confirmed_long and not extended:
            set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bullish")
        elif confirmed_short and not extended:
            set_state(asset_key, status="ENTRY ACTIVE", active_direction="Bearish")
        else:
            # OPEN DRIVE bias (early)
            if open_drive_long:
                set_state(asset_key, status="OPEN DRIVE", active_direction="Bullish")
            elif open_drive_short:
                set_state(asset_key, status="OPEN DRIVE", active_direction="Bearish")
            else:
                # If we have bias + trend-like but extended, show caution
                if trend_like and extended and bias in ("Bullish", "Bearish"):
                    set_state(asset_key, status="CAUTION", active_direction=bias)
                elif trend_like and bias in ("Bullish", "Bearish"):
                    set_state(asset_key, status="WAITING", active_direction=bias)
                else:
                    set_state(asset_key, status="STAND DOWN", active_direction=None)

s = get_state(asset_key)
status = s["status"]
active_dir = s.get("active_direction")
exit_reason = s.get("last_exit_reason")

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
            f'Quote source: {quote["source"]} • freshness: {freshness_sec:.1f}s<br>'
            f'Candle Close: {fmt(candle_close)} • Last candle time: {last_candle_time}'
            f'</div>',
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)
        m1.metric("Move vs last candle", fmt_delta(current - candle_close), fmt_pct(pct_change(current, candle_close)))
        m2.metric("Move vs session open", fmt_delta(current - session_open), fmt_pct(pct_change(current, session_open)))

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

with tab_engine:
    st.subheader("Engine Readout")
    st.write(f"**Directional Bias:** `{bias_text}`")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Bias (vs VWAP)", bias)
    e2.metric("Regime", mkt_state)
    e3.metric("VWAP Confirm", vwap_side)
    e4.metric("Momentum", mom)

    e5, e6, e7, e8 = st.columns(4)
    e5.metric("VWAP", fmt(vwap))
    e6.metric("ATR", fmt(atr))
    e7.metric("RVOL", "—" if np.isnan(rvol) else f"{rvol:.2f}")
    e8.metric("BB Width", "—" if np.isnan(bw_now) else f"{bw_now:.4f}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.write("**Notes**")
    st.write("- **OPEN DRIVE** = early impulse bias (not a full confirmation, but you should align with it)")
    st.write("- **ENTRY ACTIVE** = tradable edge")
    st.write("- **CAUTION** = extended/late entry risk")
    st.write("- **EXIT/RESET** = done, wait")

with tab_raw:
    st.subheader("Raw Session Table (last 80 candles)")
    raw = df.tail(80).copy()
    keep = ["Open", "High", "Low", "Close", "Volume", "VWAP", "EMA9", "ATR", "RVOL", "BB_bw"]
    keep = [c for c in keep if c in raw.columns]
    st.dataframe(raw[keep], use_container_width=True)

st.caption("Not financial advice. Decision-support engine only.")