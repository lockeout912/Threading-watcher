import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta, time

# ============================================================
# LOCKOUT SIGNALS • 8.8 AGGRESSIVE TREND HUNTER
# SPY + BTC • One-glance decision cockpit
# ============================================================

# ----------------------------
# PAGE / STYLE
# ----------------------------
st.set_page_config(page_title="Lockout Signals • SPY + BTC", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: .8rem; max-width: 1120px; }
.card { padding: 16px 18px; border-radius: 16px; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10); }
.h1 { font-size: 2.0rem; font-weight: 900; margin: 0 0 .25rem 0; }
.price { font-size: 3.1rem; font-weight: 900; margin: 0; line-height: 1; }
.k { opacity: .80; font-size: .92rem; }
.small { font-size: .92rem; opacity: .80; }
.hr { height: 1px; background: rgba(255,255,255,0.12); margin: 12px 0; }
.pill { display:inline-block; padding:6px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.14);
        background: rgba(255,255,255,.06);
        font-weight: 800; font-size: .88rem; margin-right: 6px; margin-bottom: 6px; }
.action { font-size: 1.45rem; font-weight: 900; margin: 0; }
.subaction { font-size: 1.05rem; font-weight: 750; margin: 0; opacity:.92; }
.bigline { font-size: 1.05rem; font-weight: 800; }
.bul { margin: .25rem 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# ASSETS
# ----------------------------
ASSETS = {
    "SPY": {"ticker": "SPY", "type": "equity"},
    "BTC": {"ticker": "BTC-USD", "type": "crypto"},
}

# ============================================================
# TUNING (Aggressive Mode)
# ============================================================
# Data
INTERVAL_PRIMARY = "1m"
INTERVAL_FALLBACK = "5m"
PERIOD = "5d"

# Indicators
EMA_LEN = 9
ATR_LEN = 14
BB_LEN = 20
BB_STD = 2
RVOL_LEN = 20

# Aggressive signal ladder gates
OPEN_DRIVE_MIN = 60                 # first 60 min for SPY
ORB_MINUTES = 5                      # 5-min opening range
FLIP_LOOKBACK = 20                   # last 20 candles flip count
MAX_FLIPS_TREND = 4                  # above this -> chop risk
RVOL_HEADSUP = 0.75                  # early heads-up (loose)
RVOL_ENTRY = 0.95                    # entry active (still loose)
EXT_ATR_CAUTION = 1.35               # caution if extended from VWAP
RESET_MIN = 10                       # cooloff after exit

# Score weights (0–100)
W_VWAP_SIDE = 25
W_VWAP_SLOPE = 10
W_EMA_SLOPE = 15
W_BBW_EXPAND = 15
W_ORB_CONFIRM = 20
W_LOW_FLIPS = 15

# Score thresholds → states
TH_ENTRY = 80
TH_SETUP = 60
TH_HEADSUP = 45

# ============================================================
# HELPERS
# ============================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def fmt(x) -> str:
    if x is None or np.isnan(x): return "—"
    return f"{x:,.2f}"

def fmt_delta(x) -> str:
    if x is None or np.isnan(x): return "—"
    s = "+" if x >= 0 else ""
    return f"{s}{x:,.2f}"

def fmt_pct(p) -> str:
    if p is None or np.isnan(p): return "—"
    return f"{p:+.2f}%"

def pct(a, b) -> float:
    if np.isnan(a) or np.isnan(b) or b == 0: return np.nan
    return (a - b) / b * 100.0

def pill(text: str) -> str:
    return f'<span class="pill">{text}</span>'

def to_tz(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

# ============================================================
# QUOTES (fast-ish)
# ============================================================
@st.cache_data(ttl=8)
def yf_quote(ticker: str) -> dict:
    t0 = now_utc()
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            p = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            p = safe_float(p)
            if not np.isnan(p):
                return {"price": p, "fetched_at": t0, "source": "yfinance.fast_info"}
        info = getattr(t, "info", {}) or {}
        p = safe_float(info.get("regularMarketPrice"))
        if not np.isnan(p):
            return {"price": p, "fetched_at": t0, "source": "yfinance.info"}
    except Exception:
        pass
    return {"price": np.nan, "fetched_at": t0, "source": "yfinance.unavailable"}

@st.cache_data(ttl=2)
def coinbase_btc_quote() -> dict:
    t0 = now_utc()
    # Coinbase Exchange is fast when it responds
    urls = [
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=2.5, headers={"User-Agent": "LockoutSignals/1.0"})
            if r.status_code == 200:
                j = r.json()
                if "exchange.coinbase.com" in url:
                    p = safe_float(j.get("price"))
                    src = "coinbase.exchange"
                else:
                    p = safe_float((((j or {}).get("data") or {}).get("amount")))
                    src = "coinbase.spot"
                if not np.isnan(p):
                    return {"price": p, "fetched_at": t0, "source": src}
        except Exception:
            pass
    return {"price": np.nan, "fetched_at": t0, "source": "coinbase.unavailable"}

# ============================================================
# CANDLES
# ============================================================
@st.cache_data(ttl=18)
def fetch_candles(ticker: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=PERIOD, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception:
        return pd.DataFrame()

def session_filter(df: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df, ""

    df = df.copy()

    if asset_type == "equity":
        df.index = to_tz(df.index, "America/New_York")
        d = df.index[-1].date()
        sid = d.isoformat()
        start = datetime.combine(d, time(9, 30)).replace(tzinfo=df.index.tz)
        end = datetime.combine(d, time(16, 0)).replace(tzinfo=df.index.tz)
        return df[(df.index >= start) & (df.index <= end)].copy(), sid

    # crypto = UTC day
    df.index = to_tz(df.index, "UTC")
    d = df.index[-1].date()
    sid = d.isoformat()
    start = datetime.combine(d, time(0, 0)).replace(tzinfo=df.index.tz)
    end = datetime.combine(d, time(23, 59)).replace(tzinfo=df.index.tz)
    return df[(df.index >= start) & (df.index <= end)].copy(), sid

# ============================================================
# INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    # EMA + slope
    df["EMA9"] = df["Close"].ewm(span=EMA_LEN, adjust=False).mean()
    df["EMA9_slope"] = df["EMA9"] - df["EMA9"].shift(3)

    # ATR
    prev = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev).abs()
    tr3 = (df["Low"] - prev).abs()
    df["TR"] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    df["ATR"] = df["TR"].rolling(ATR_LEN).mean()

    # VWAP (intraday cumulative)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    df["VWAP"] = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    df["VWAP_slope"] = df["VWAP"] - df["VWAP"].shift(3)

    # RVOL
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(RVOL_LEN).mean()

    # Bollinger Band Width (BBW) + slope
    mid = df["Close"].rolling(BB_LEN).mean()
    sd = df["Close"].rolling(BB_LEN).std(ddof=0)
    up = mid + BB_STD * sd
    lo = mid - BB_STD * sd
    df["BBW"] = (up - lo) / mid.replace(0, np.nan)
    df["BBW_slope"] = df["BBW"] - df["BBW"].shift(3)

    return df

def bias_from_vwap(price: float, vwap: float) -> str:
    if np.isnan(price) or np.isnan(vwap):
        return "Neutral"
    return "Bullish" if price >= vwap else "Bearish"

def momentum_from_ema_slope(ema_slope: float) -> str:
    if np.isnan(ema_slope):
        return "Mixed"
    if ema_slope > 0:
        return "Up"
    if ema_slope < 0:
        return "Down"
    return "Flat"

def flip_count(close: pd.Series, vwap: pd.Series, lookback: int = 20) -> int:
    if close.shape[0] < lookback + 2:
        return 999  # treat as unknown -> conservative in chop filter
    c = close.tail(lookback).values
    v = vwap.tail(lookback).values
    side = np.sign(c - v)
    # ignore zeros by forward fill
    for i in range(1, len(side)):
        if side[i] == 0:
            side[i] = side[i-1]
    flips = np.sum(side[1:] != side[:-1])
    return int(flips)

def is_open_drive_window(df: pd.DataFrame, asset_type: str) -> bool:
    if asset_type != "equity" or df.empty:
        return False
    try:
        ts = df.index[-1]  # NY tz
        mo = ts.replace(hour=9, minute=30, second=0, microsecond=0)
        return ts <= (mo + timedelta(minutes=OPEN_DRIVE_MIN))
    except Exception:
        return False

def compute_orb(df_1m: pd.DataFrame, asset_type: str) -> dict:
    """
    5-min ORB for equities; for crypto we skip ORB as 'N/A' (can still be used later).
    Returns dict with or_high/or_low and a confirmation flag based on current price vs OR.
    """
    if asset_type != "equity" or df_1m.empty:
        return {"or_high": np.nan, "or_low": np.nan, "orb": "N/A", "orb_confirm": False}

    # df_1m should be session-filtered and in NY tz
    # Use first ORB_MINUTES candles from session open
    try:
        session_start = df_1m.index[0]
        end = session_start + timedelta(minutes=ORB_MINUTES)
        first = df_1m[df_1m.index < end]
        if first.empty:
            return {"or_high": np.nan, "or_low": np.nan, "orb": "N/A", "orb_confirm": False}
        or_high = safe_float(first["High"].max())
        or_low = safe_float(first["Low"].min())
        return {"or_high": or_high, "or_low": or_low, "orb": f"{ORB_MINUTES}m ORB", "orb_confirm": False}
    except Exception:
        return {"or_high": np.nan, "or_low": np.nan, "orb": "N/A", "orb_confirm": False}

def expected_ranges(price: float, vwap: float, atr: float, direction: str, stretch_ok: bool) -> dict:
    """
    Range bands from VWAP using ATR multipliers.
    Base = 1x ATR, Trend = 2x ATR, Stretch = 3x ATR (only if stretch_ok).
    Invalidation = VWAP ± 1x ATR opposite direction.
    """
    if np.isnan(atr) or atr <= 0:
        atr = max(price * 0.003, 0.50)

    if direction == "Bullish":
        base = (vwap + 1*atr, vwap + 2*atr)
        trend = (vwap + 2*atr, vwap + 3*atr)
        stretch = (vwap + 3*atr, vwap + 4*atr) if stretch_ok else (np.nan, np.nan)
        inval = vwap - 1*atr
    elif direction == "Bearish":
        base = (vwap - 2*atr, vwap - 1*atr)
        trend = (vwap - 3*atr, vwap - 2*atr)
        stretch = (vwap - 4*atr, vwap - 3*atr) if stretch_ok else (np.nan, np.nan)
        inval = vwap + 1*atr
    else:
        base = trend = stretch = (np.nan, np.nan)
        inval = np.nan

    def move_to_range(lo, hi):
        if np.isnan(lo) or np.isnan(hi) or np.isnan(price) or price == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        dlo, dhi = lo - price, hi - price
        plo, phi = (dlo/price*100), (dhi/price*100)
        return (plo, phi, dlo, dhi)

    return {
        "base": (min(base), max(base)),
        "trend": (min(trend), max(trend)),
        "stretch": (min(stretch), max(stretch)) if stretch_ok else (np.nan, np.nan),
        "inval": inval,
        "move_base": move_to_range(min(base), max(base))
    }

# ============================================================
# STATE MACHINE (persistent per asset)
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = {}

def get_state(asset_key: str) -> dict:
    if asset_key not in st.session_state.state:
        st.session_state.state[asset_key] = {
            "status": "STAND DOWN",
            "active_dir": None,
            "reset_until": None,
            "session_id": None,
            "exit_reason": None
        }
    return st.session_state.state[asset_key]

def set_state(asset_key: str, **kwargs):
    s = get_state(asset_key)
    s.update(kwargs)
    st.session_state.state[asset_key] = s

def in_reset(s: dict) -> bool:
    return s["reset_until"] is not None and now_utc() < s["reset_until"]

# ============================================================
# UI CONTROLS
# ============================================================
st.markdown('<div class="h1">Lockout Signals • SPY + BTC</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1.15, 1.0, 1.0, 1.0])
with c1:
    asset_key = st.selectbox("Asset", list(ASSETS.keys()), index=0)
with c2:
    st.button("🔄 Refresh now")
with c3:
    auto = st.toggle("Auto-refresh", value=True)
with c4:
    refresh_s = st.selectbox("Refresh rate", [5, 10, 20, 60], index=0, format_func=lambda x: f"{x}s")

if auto:
    try:
        st.autorefresh(interval=int(refresh_s * 1000), key="auto")
    except Exception:
        st.caption("Auto-refresh not supported here; use Refresh now.")

asset = ASSETS[asset_key]
ticker = asset["ticker"]
atype = asset["type"]

# ============================================================
# LOAD CANDLES (1m preferred, fallback 5m)
# ============================================================
df1_raw = fetch_candles(ticker, INTERVAL_PRIMARY)
df5_raw = fetch_candles(ticker, INTERVAL_FALLBACK)

df_raw = df1_raw if not df1_raw.empty else df5_raw
interval_used = INTERVAL_PRIMARY if not df1_raw.empty else INTERVAL_FALLBACK

df_sess, sid = session_filter(df_raw, atype)

# Need enough bars to compute BB/ATR reasonably
if df_sess.empty or df_sess.shape[0] < max(ATR_LEN + 3, 12):
    st.warning("Not enough candles yet — let a few print, then refresh.")
    st.stop()

df = add_indicators(df_sess)

# For ORB we want 1m session if available
df1_sess = pd.DataFrame()
if not df1_raw.empty:
    df1_sess, _ = session_filter(df1_raw, atype)

# ============================================================
# QUOTE (fast-ish)
# ============================================================
if asset_key == "BTC":
    q = coinbase_btc_quote()
else:
    q = yf_quote("SPY")

price = safe_float(q["price"])
if np.isnan(price):
    price = safe_float(df["Close"].iloc[-1])

fresh_sec = (now_utc() - q["fetched_at"]).total_seconds()

# ============================================================
# ENGINE READS
# ============================================================
last = df.iloc[-1]
close_last = safe_float(last["Close"])
open_px = safe_float(df["Open"].iloc[0])
last_time = df.index[-1]

vwap = safe_float(last.get("VWAP", np.nan))
atr = safe_float(last.get("ATR", np.nan))
ema_slope = safe_float(last.get("EMA9_slope", np.nan))
vwap_slope = safe_float(last.get("VWAP_slope", np.nan))
bbw = safe_float(last.get("BBW", np.nan))
bbw_slope = safe_float(last.get("BBW_slope", np.nan))
rvol = safe_float(last.get("RVOL", np.nan))

intraday_bias = bias_from_vwap(price, vwap)
mom = momentum_from_ema_slope(ema_slope)

flips = flip_count(df["Close"], df["VWAP"], lookback=FLIP_LOOKBACK)
chop_risk = flips > MAX_FLIPS_TREND

open_drive = is_open_drive_window(df, atype)

# Impulse / extension (for CAUTION logic)
atr_use = atr if not np.isnan(atr) and atr > 0 else max(price * 0.003, 0.50)
extended = (not np.isnan(vwap)) and (abs(price - vwap) >= EXT_ATR_CAUTION * atr_use)

# ORB
orb = compute_orb(df1_sess, atype)
or_high, or_low = orb["or_high"], orb["or_low"]

orb_confirm = False
if atype == "equity" and not np.isnan(or_high) and not np.isnan(or_low):
    # "break + hold" proxy: current price beyond OR in direction + VWAP aligned
    if intraday_bias == "Bullish" and price > or_high and price >= vwap:
        orb_confirm = True
    if intraday_bias == "Bearish" and price < or_low and price <= vwap:
        orb_confirm = True

# Volatility ignition
bbw_expand = (not np.isnan(bbw_slope)) and (bbw_slope > 0)
vwap_slope_ok = (vwap_slope > 0) if intraday_bias == "Bullish" else (vwap_slope < 0) if intraday_bias == "Bearish" else False
ema_slope_ok = (ema_slope > 0) if intraday_bias == "Bullish" else (ema_slope < 0) if intraday_bias == "Bearish" else False

# RVOL flags (loose)
rvol_heads = (np.isnan(rvol) or rvol >= RVOL_HEADSUP)
rvol_entry = (np.isnan(rvol) or rvol >= RVOL_ENTRY)

# ============================================================
# CONFIDENCE SCORE (0–100)
# ============================================================
score = 0
why = []

# VWAP side
if intraday_bias in ("Bullish", "Bearish"):
    score += W_VWAP_SIDE
    why.append("Price aligned to VWAP (direction)")
else:
    why.append("VWAP unclear")

# VWAP slope
if vwap_slope_ok:
    score += W_VWAP_SLOPE
    why.append("VWAP slope supports trend")

# EMA slope (momentum)
if ema_slope_ok:
    score += W_EMA_SLOPE
    why.append("Momentum slope supports trend")

# BBW expansion (breakout fuel)
if bbw_expand:
    score += W_BBW_EXPAND
    why.append("Volatility expanding (BBW rising)")

# ORB confirmation (SPY only)
if orb_confirm:
    score += W_ORB_CONFIRM
    why.append("Opening range breakout confirmed")

# Low flips (trend integrity)
if not chop_risk and flips != 999:
    score += W_LOW_FLIPS
    why.append("Low VWAP flip-count (trend integrity)")
else:
    why.append("VWAP flips high (chop risk)")

# Cap score
score = int(max(0, min(100, score)))

# Trend strength label
if score >= 85:
    strength = "HIGH"
elif score >= 70:
    strength = "MED"
elif score >= 55:
    strength = "LOW"
else:
    strength = "WEAK"

# Stretch condition
stretch_ok = (score >= 85) or (orb_confirm and bbw_expand and not chop_risk)

# Expected ranges
ranges = expected_ranges(price, vwap, atr_use, intraday_bias if intraday_bias in ("Bullish", "Bearish") else "Neutral", stretch_ok)

# ============================================================
# STATE LOGIC (Aggressive Ladder)
# ============================================================
s = get_state(asset_key)

# Session reset
if sid and s.get("session_id") != sid:
    set_state(asset_key, status="STAND DOWN", active_dir=None, reset_until=None, session_id=sid, exit_reason=None)
    s = get_state(asset_key)

# Determine desired direction
direction = intraday_bias if intraday_bias in ("Bullish", "Bearish") else None

def exit_trigger(active_dir: str) -> tuple[bool, str]:
    # Exit if VWAP flips against us OR invalidation breaks
    if active_dir == "Bullish":
        if bias_from_vwap(price, vwap) == "Bearish":
            return True, "VWAP flip"
        if not np.isnan(ranges["inval"]) and price < ranges["inval"]:
            return True, "Invalidation break"
    if active_dir == "Bearish":
        if bias_from_vwap(price, vwap) == "Bullish":
            return True, "VWAP flip"
        if not np.isnan(ranges["inval"]) and price > ranges["inval"]:
            return True, "Invalidation break"
    return False, ""

# Aggressive bias call is ALWAYS ON, but tradability depends on state
if in_reset(s):
    set_state(asset_key, status="RESET")
else:
    active = s.get("active_dir")

    if active in ("Bullish", "Bearish"):
        do_exit, reason = exit_trigger(active)
        if do_exit:
            set_state(asset_key, status="EXIT / RESET", active_dir=None, exit_reason=reason, reset_until=now_utc() + timedelta(minutes=RESET_MIN))
        else:
            # CAUTION if extended or chop risk rising or slopes decaying
            caution = extended or chop_risk or (not ema_slope_ok) or (not bbw_expand and score < 75)
            set_state(asset_key, status="CAUTION" if caution else "ENTRY ACTIVE", active_dir=active)
    else:
        # If chop risk is high, stand down (even aggressive mode)
        if chop_risk and score < 80:
            set_state(asset_key, status="STAND DOWN", active_dir=None)
        else:
            # Ladder:
            # ENTRY ACTIVE: score high + direction clear + rvol ok + not extended (avoid late chase)
            if direction and score >= TH_ENTRY and rvol_entry and not extended:
                set_state(asset_key, status="ENTRY ACTIVE", active_dir=direction)
            # OPEN DRIVE: early session + strong impulse structure (we proxy via score + open_drive)
            elif direction and open_drive and score >= 65 and rvol_heads:
                set_state(asset_key, status="OPEN DRIVE", active_dir=direction)
            # SETUP: decent score and direction
            elif direction and score >= TH_SETUP and rvol_heads:
                set_state(asset_key, status="SETUP", active_dir=direction)
            # HEADS UP: direction exists but confidence not yet high
            elif direction and score >= TH_HEADSUP:
                set_state(asset_key, status="HEADS UP", active_dir=direction)
            else:
                set_state(asset_key, status="STAND DOWN", active_dir=None)

# Final state read
s = get_state(asset_key)
status = s["status"]
active_dir = s.get("active_dir") or direction
exit_reason = s.get("exit_reason")

# ============================================================
# DECISION TEXT (UNMISTAKABLE)
# ============================================================
def calls_or_puts(d: str) -> str:
    return "CALLS" if d == "Bullish" else "PUTS" if d == "Bearish" else "STAND DOWN"

action_side = calls_or_puts(active_dir if active_dir else "")
trend_call = "CALLS FAVORED" if intraday_bias == "Bullish" else "PUTS FAVORED" if intraday_bias == "Bearish" else "NEUTRAL"

# Human readable “what to do”
if status == "ENTRY ACTIVE":
    headline = f"{action_side} — ENTRY ACTIVE"
    subline = "Trend confirmed. Trade with direction while VWAP holds."
elif status == "OPEN DRIVE":
    headline = f"{action_side} — OPEN DRIVE"
    subline = "Early momentum. Prefer this direction. Wait for pullback to size up."
elif status == "SETUP":
    headline = f"{action_side} — SETUP"
    subline = "Trend forming. Confirmation building."
elif status == "HEADS UP":
    headline = f"{action_side} — HEADS UP"
    subline = "Direction present, but not fully confirmed."
elif status == "CAUTION":
    headline = "CAUTION — PROTECT PROFITS"
    subline = "Trend slowing / extended / chop risk rising. Tighten or take profits."
elif status == "EXIT / RESET":
    why_exit = f" ({exit_reason})" if exit_reason else ""
    headline = f"EXIT / RESET{why_exit}"
    subline = "Edge gone. Stand down and wait for next cycle."
elif status == "RESET":
    headline = "RESET (cooldown)"
    subline = "Avoid chop whipsaws. Waiting briefly before re-engaging."
else:
    headline = "STAND DOWN"
    subline = "No clean edge. Preserve capital and wait."

# ============================================================
# UI — COCKPIT (one glance)
# ============================================================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns([1.15, 1.25, 1.10])

with top1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="k">{asset_key} • {ticker} • {interval_used} • Session {sid}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="price">{fmt(price)}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="small">Quote: {q["source"]} • freshness: {fresh_sec:.1f}s • last candle: {last_time}</div>',
        unsafe_allow_html=True
    )
    m1, m2 = st.columns(2)
    m1.metric("vs last candle", fmt_delta(price - close_last), fmt_pct(pct(price, close_last)))
    m2.metric("vs open", fmt_delta(price - open_px), fmt_pct(pct(price, open_px)))
    st.markdown('</div>', unsafe_allow_html=True)

with top2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(pill(f"TREND CALL: {trend_call}"), unsafe_allow_html=True)
    st.markdown(pill(f"STATUS: {status}"), unsafe_allow_html=True)
    st.markdown(pill(f"STRENGTH: {strength}"), unsafe_allow_html=True)
    st.markdown(pill(f"SCORE: {score}/100"), unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="action">{headline}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subaction">{subline}</div>', unsafe_allow_html=True)

    # ORB and chop hints
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if atype == "equity" and not np.isnan(or_high) and not np.isnan(or_low):
        st.markdown(pill(f"{ORB_MINUTES}m OR: {fmt(or_low)} → {fmt(or_high)}"), unsafe_allow_html=True)
        st.markdown(pill(f"ORB Confirm: {'YES' if orb_confirm else 'no'}"), unsafe_allow_html=True)
    st.markdown(pill(f"VWAP Flips({FLIP_LOOKBACK}): {flips if flips != 999 else '—'}"), unsafe_allow_html=True)
    st.markdown(pill(f"Chop Risk: {'HIGH' if chop_risk else 'low'}"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="bigline">Expected Range (from VWAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if intraday_bias in ("Bullish", "Bearish") and not np.isnan(vwap):
        base_lo, base_hi = ranges["base"]
        tr_lo, tr_hi = ranges["trend"]
        st_lo, st_hi = ranges["stretch"]
        inval = ranges["inval"]
        plo, phi, dlo, dhi = ranges["move_base"]

        st.markdown(f"**Bias:** {intraday_bias.upper()}  ({action_side})")
        st.markdown(f"**VWAP:** {fmt(vwap)} • **ATR:** {fmt(atr_use)}")
        st.markdown(f"**Base Range:** {fmt(base_lo)} → {fmt(base_hi)}")
        st.markdown(f"**Trend Range:** {fmt(tr_lo)} → {fmt(tr_hi)}")
        if stretch_ok and not np.isnan(st_lo) and not np.isnan(st_hi):
            st.markdown(f"**Stretch (Trend Day):** {fmt(st_lo)} → {fmt(st_hi)}")
        st.markdown(f"**Exit if {'below' if intraday_bias=='Bullish' else 'above'}:** {fmt(inval)}")
        st.markdown(f"<div class='small'>Move to Base Range: {fmt_pct(plo)} to {fmt_pct(phi)} "
                    f"({fmt_delta(dlo)} to {fmt_delta(dhi)})</div>", unsafe_allow_html=True)
    else:
        st.markdown("No directional edge yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# WHY BULLETS (max 3, clean)
# ============================================================
st.markdown("### Why it’s saying that (3 bullets max)")
bullets = []
# Curate bullets: keep top 3 most actionable
if intraday_bias in ("Bullish", "Bearish"):
    bullets.append(f"{'Above' if intraday_bias=='Bullish' else 'Below'} VWAP (direction spine)")
if ema_slope_ok:
    bullets.append("Momentum slope supports continuation")
if bbw_expand:
    bullets.append("Volatility expanding (breakout fuel)")
if orb_confirm:
    bullets.append("Opening range breakout confirmed")
if chop_risk:
    bullets.append("Chop risk rising (VWAP flips high)")
if extended:
    bullets.append("Extended from VWAP (late entry risk)")

# pick 3 prioritized
priority = []
for b in bullets:
    if "Chop risk" in b or "Extended" in b:
        priority.append(b)  # risk alerts are high priority
for b in bullets:
    if b not in priority:
        priority.append(b)
priority = priority[:3] if len(priority) > 3 else priority

for b in priority:
    st.markdown(f"- {b}")

# ============================================================
# MICRO TREND CHART (clean)
# ============================================================
st.markdown("### Micro Trend (context)")
mini_cols = [c for c in ["Close", "VWAP", "EMA9"] if c in df.columns]
mini = df.tail(260)[mini_cols].dropna()
st.line_chart(mini, height=260)

# ============================================================
# DIAGNOSTICS (collapsible)
# ============================================================
with st.expander("Diagnostics (engine room)", expanded=False):
    a, b, c, d = st.columns(4)
    a.metric("VWAP", fmt(vwap))
    b.metric("VWAP slope", fmt(vwap_slope))
    c.metric("EMA slope", fmt(ema_slope))
    d.metric("BBW slope", "—" if np.isnan(bbw_slope) else f"{bbw_slope:.6f}")

    a2, b2, c2, d2 = st.columns(4)
    a2.metric("RVOL", "—" if np.isnan(rvol) else f"{rvol:.2f}")
    b2.metric("ATR", fmt(atr_use))
    c2.metric("Extended?", "YES" if extended else "no")
    d2.metric("Open Drive Window", "YES" if open_drive else "no")

    st.write("**Score Notes:**")
    st.write("- Aggressive system: Bias speaks early (HEADS UP/SETUP), tradable when score is high (ENTRY ACTIVE).")
    st.write("- Chop filter uses VWAP flip-count to prevent death-by-chop while staying trend-forward.")
    st.write("- Ranges are VWAP + ATR bands (Base/Trend/Stretch).")
    st.write("- For true tick-level speed, you’d eventually use a dedicated market data feed. This version is the best you can do free/near-free.")

st.caption("Decision-support only. Not financial advice.")