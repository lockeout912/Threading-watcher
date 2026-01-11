import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lockout Signals • SPY + BTC", layout="wide")

# ============================================================
# STYLE (minimal, safe)
# ============================================================
st.markdown("""
<style>
/* Keep it clean + mobile friendly */
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }
.small-muted { opacity: 0.75; font-size: 0.85rem; }
.big-price { font-size: 3.0rem; font-weight: 900; line-height: 1.05; margin-top: -6px; }
.badge {
  display:inline-block; padding: 0.35rem 0.65rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.06);
  font-weight: 900; letter-spacing: 0.2px; font-size: 0.85rem;
}
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

@st.cache_data(ttl=25)
def fetch_intraday(ticker: str) -> pd.DataFrame:
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

@st.cache_data(ttl=10)
def fetch_live_price(ticker: str) -> float:
    """
    Best-effort 'live-ish' price.
    yfinance isn't true tick-by-tick, but this helps refresh between candles.
    """
    try:
        t = yf.Ticker(ticker)
        # fast_info is usually the quickest path
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            p = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            p = safe_float(p)
            if not np.isnan(p):
                return p
        info = t.info if hasattr(t, "info") else {}
        p = safe_float(info.get("regularMarketPrice"))
        if not np.isnan(p):
            return p
        return np.nan
    except Exception:
        return np.nan

def to_tz(dt_index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    idx = dt_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

def filter_session(df: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, str]:
    """
    SPY: RTH ET 09:30-16:00
    BTC: UTC day
    """
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
    vwap = safe_float(last["VWAP"])
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
    pct_low = (usd_low / current) * 100 if current else 0.0
    pct_high = (usd_high / current) * 100 if current else 0.0

    return {
        "current": current, "vwap": vwap, "atr": atr,
        "entry": entry, "target_low": lo, "target_high": hi, "exit_if": exit_if,
        "usd_low": usd_low, "usd_high": usd_high, "pct_low": pct_low, "pct_high": pct_high
    }


# ============================================================
# SESSION STATE
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = {}

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


# ============================================================
# TOP CONTROLS
# ============================================================
st.title("Lockout Signals • SPY + BTC")

c1, c2, c3 = st.columns([1.1, 0.9, 1.2])
with c1:
    asset_key = st.selectbox("Asset", list(ASSETS.keys()), index=0)
with c2:
    st.button("🔄 Refresh")
with c3:
    auto_refresh = st.toggle("Auto-refresh (20s)", value=False)

# Optional auto refresh (no extra dependencies)
if auto_refresh:
    st.caption("Auto-refresh ON (every 20s).")
    st.experimental_set_query_params(_=str(datetime.now().timestamp()))
    st.rerun()

asset = ASSETS[asset_key]
ticker = asset["ticker"]
atype = asset["type"]

# ============================================================
# DATA
# ============================================================
df_raw = fetch_intraday(ticker)
df_sess, session_id = filter_session(df_raw, atype)

s = get_state(asset_key)
if session_id and s.get("last_session_id") != session_id:
    set_state(asset_key, status="STAND DOWN", active_direction=None, reset_until=None, last_session_id=session_id, last_exit_reason=None)

df = compute_indicators(df_sess)
if df.empty or df.shape[0] < 30:
    st.warning("Not enough session data yet — let more 5-minute candles print.")
    st.stop()

last = df.iloc[-1]
close_price = safe_float(last["Close"])
live_price = fetch_live_price(ticker)
current = live_price if not np.isnan(live_price) else close_price

vwap = safe_float(last["VWAP"])
atr = safe_float(last.get("ATR", np.nan))
rvol = safe_float(last.get("RVOL", np.nan))

mkt_state, bw_now = bandwidth_state(df)
vwap_side = vwap_confirmed_side(df)
mom = momentum_state(df)

bias = "Mixed"
if not np.isnan(current) and not np.isnan(vwap):
    bias = "Bullish" if current >= vwap else "Bearish"


# ============================================================
# HEADS UP
# ============================================================
vol_prev = safe_float(df["Volume"].iloc[-2]) if df.shape[0] >= 2 else np.nan
vol_now = safe_float(df["Volume"].iloc[-1])

near_vwap = (not np.isnan(atr) and not np.isnan(current) and not np.isnan(vwap) and abs(current - vwap) <= (HEADSUP_VWAP_ATR_BAND * atr))

momentum_improving = False
if "EMA9_slope" in df.columns and df["EMA9_slope"].shape[0] >= 5:
    s_now = safe_float(df["EMA9_slope"].iloc[-1])
    s_prev = safe_float(df["EMA9_slope"].iloc[-3])
    momentum_improving = (not np.isnan(s_now) and not np.isnan(s_prev) and s_now > s_prev)

participation_rising = (not np.isnan(vol_now) and not np.isnan(vol_prev) and vol_now > vol_prev and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN))

bw_series = df["BB_bw"].dropna()
bw_improving = (bw_series.shape[0] >= 3 and safe_float(bw_series.iloc[-1]) > safe_float(bw_series.iloc[-2]))

heads_up = (mkt_state == "Chop" and near_vwap and momentum_improving and participation_rising and bw_improving)


# ============================================================
# CONFIRMED ENTRY ACTIVE
# ============================================================
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
    # LATER: CAUTION only if 2+ weakening factors while still on the right VWAP side
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


# ============================================================
# STATE MACHINE
# ============================================================
now_utc = datetime.now(timezone.utc)
s = get_state(asset_key)

in_reset = s["reset_until"] is not None and now_utc < s["reset_until"]
if in_reset:
    set_state(asset_key, status="RESET")
else:
    active = s.get("active_direction")

    if active in ("Bullish", "Bearish"):
        do_exit, reason = exit_triggered(active)
        if do_exit:
            set_state(
                asset_key,
                status="EXIT / RESET",
                active_direction=None,
                reset_until=now_utc + timedelta(minutes=RESET_MINUTES),
                last_exit_reason=reason
            )
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


# ============================================================
# DECISION OUTPUT (SINGLE SOURCE OF TRUTH)
# ============================================================
def status_box(sts: str):
    if sts == "ENTRY ACTIVE":
        return st.success
    if sts in ("CAUTION", "HEADS UP"):
        return st.warning
    if sts in ("EXIT / RESET",):
        return st.error
    return st.info

def decision_text(sts: str, direction: str | None):
    if sts == "ENTRY ACTIVE" and direction == "Bullish":
        return "ENTER LONG (still valid)", "Buy calls / long exposure while price holds ABOVE VWAP. Use Exit-If."
    if sts == "ENTRY ACTIVE" and direction == "Bearish":
        return "ENTER SHORT (still valid)", "Buy puts / short exposure while price holds BELOW VWAP. Use Exit-If."
    if sts == "CAUTION":
        return "CAUTION (late entry risk)", "If entering now, size down. Prefer re-confirmation or a fresh setup."
    if sts == "HEADS UP":
        return "HEADS UP (do not enter yet)", "Pressure is building. Wait for ENTRY ACTIVE confirmation."
    if sts == "EXIT / RESET":
        why = f" ({exit_reason})" if exit_reason else ""
        return f"EXIT / STAND DOWN{why}", "Move is done. No chasing. Wait for the next ENTRY ACTIVE cycle."
    if sts == "RESET":
        return "RESET (cooling off)", "No trades during reset. We avoid chop whipsaws."
    if sts == "WAITING":
        return "WAITING (setup forming)", "Trend exists but confirmation isn’t clean yet. Patience."
    return "STAND DOWN (chop)", "No edge right now. Preserve capital and wait."

direction = active_dir if active_dir in ("Bullish", "Bearish") else ("Bullish" if bias == "Bullish" else "Bearish")
levels = compute_levels(df, "Bullish" if direction == "Bullish" else "Bearish")

title, instruction = decision_text(status, active_dir)

# ============================================================
# MAIN VIEW (CLEAN, MOBILE-FIRST)
# ============================================================
topL, topR = st.columns([1.35, 1])

with topL:
    st.markdown(f"**{asset_key} • {ticker} • {INTERVAL} • Session {session_id}**")
    st.markdown(f'<div class="big-price">{fmt(current)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-muted">Last candle time: {df.index[-1]}</div>', unsafe_allow_html=True)

with topR:
    st.markdown(f'<span class="badge">{status}</span>', unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    status_box(status)(instruction)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)

entry_line = f"Above {fmt(levels['entry'])}" if direction == "Bullish" else f"Below {fmt(levels['entry'])}"
exit_line  = f"Below {fmt(levels['exit_if'])}" if direction == "Bullish" else f"Above {fmt(levels['exit_if'])}"

k1.metric("Entry Rule (VWAP)", entry_line)
k2.metric("Target Zone", f"{fmt(levels['target_low'])} → {fmt(levels['target_high'])}")
k3.metric("Exit-If", exit_line)

st.caption(f"Move: {levels['pct_low']:+.2f}% to {levels['pct_high']:+.2f}%  |  {levels['usd_low']:+.2f} to {levels['usd_high']:+.2f}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ============================================================
# ENGINE READOUT (always visible, simple)
# ============================================================
st.subheader("Engine Readout (why the decision is what it is)")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Bias", bias)
r2.metric("Market State", mkt_state)
r3.metric("VWAP Confirm", vwap_side)
r4.metric("Momentum", mom)

r5, r6, r7, r8 = st.columns(4)
r5.metric("VWAP", fmt(vwap))
r6.metric("ATR", fmt(atr))
r7.metric("RVOL", "—" if np.isnan(rvol) else f"{rvol:.2f}")
r8.metric("BB Width", "—" if np.isnan(bw_now) else f"{bw_now:.4f}")

st.caption("Heads Up ≠ Entry. ENTRY ACTIVE = still tradable. CAUTION = late entry risk. EXIT/RESET = move ended—wait.")