import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="SPY + BTC Command", layout="wide")

# ============================================================
# FANCY WALL STREET STYLING
# ============================================================
st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(1200px 800px at 20% 0%, rgba(255,255,255,0.06), transparent 60%),
              radial-gradient(1200px 800px at 80% 0%, rgba(255,255,255,0.04), transparent 60%),
              #0b1220;
  color: #e8eefc;
}
* { -webkit-font-smoothing: antialiased; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.card2 {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
.label {
  opacity: 0.75;
  font-size: 12px;
  letter-spacing: 0.35px;
  text-transform: uppercase;
}
.headline {
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.4px;
  margin: 2px 0 6px 0;
}
.subline {
  font-size: 14px;
  opacity: 0.92;
  margin-top: 2px;
}
.pill {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-weight: 800;
  letter-spacing: 0.3px;
}
.hr {
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 12px 0;
}
.good { color: #6cf2a5; }
.bad  { color: #ff6b6b; }
.warn { color: #ffd166; }
.neutral { color: #b9c3d9; }
.pulse {
  height: 3px;
  border-radius: 8px;
  background: linear-gradient(90deg, rgba(108,242,165,0.0), rgba(108,242,165,0.55), rgba(108,242,165,0.0));
  opacity: 0.7;
  margin-top: 8px;
}
.small { font-size: 12px; opacity: 0.72; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS / DEFAULTS
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

# Confirmation / guardrails
VWAP_CONFIRM_BARS = 2
RESET_MINUTES = 15

# Heads up (not confirmed)
HEADSUP_VWAP_ATR_BAND = 0.50
RVOL_HEADSUP_MIN = 1.00

# Confirmed Entry Active thresholds
RVOL_ENTRY_MIN = 1.30

# BB width expansion gate
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

@st.cache_data(ttl=60)
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

def to_tz(dt_index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    idx = dt_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

def filter_session(df: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, str]:
    """
    SPY: regular trading hours ET (09:30-16:00), session_id = YYYY-MM-DD (ET)
    BTC: UTC day, session_id = YYYY-MM-DD (UTC)
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
        session_df = df[(df.index >= start) & (df.index <= end)].copy()
        return session_df, session_id

    df.index = to_tz(df.index, "UTC")
    session_date = df.index[-1].date()
    session_id = session_date.isoformat()

    start = datetime.combine(session_date, time(0, 0)).replace(tzinfo=df.index.tz)
    end = datetime.combine(session_date, time(23, 59)).replace(tzinfo=df.index.tz)
    session_df = df[(df.index >= start) & (df.index <= end)].copy()
    return session_df, session_id

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: VWAP (session anchored), EMA9, EMA9_slope, ATR, RVOL, BB width
    """
    df = df.copy()
    if df.empty or df.shape[0] < max(BB_PERIOD, ATR_PERIOD, RVOL_LOOKBACK, EMA_PERIOD) + 5:
        return df

    # EMA9 + slope
    df["EMA9"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA9_slope"] = df["EMA9"] - df["EMA9"].shift(3)

    # ATR
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["TR"] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()

    # VWAP
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    pv = tp * vol
    cum_vol = vol.cumsum().replace(0, np.nan)
    df["VWAP"] = pv.cumsum() / cum_vol

    # RVOL
    vol_ma = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / vol_ma

    # Bollinger Bandwidth (BBW)
    mid = df["Close"].rolling(BB_PERIOD).mean()
    sd = df["Close"].rolling(BB_PERIOD).std(ddof=0)
    upper = mid + BB_STD * sd
    lower = mid - BB_STD * sd
    df["BB_bw"] = (upper - lower) / mid.replace(0, np.nan)

    return df

def bandwidth_state(df: pd.DataFrame) -> tuple[str, float]:
    """
    Trend if bandwidth rising over last N bars AND lifted from recent compression.
    Else Chop.
    """
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

def vwap_confirmed_side(df: pd.DataFrame) -> tuple[str, int]:
    """
    2-bar VWAP hold confirmation.
    """
    if df.empty or "VWAP" not in df.columns:
        return "Mixed/None", 0

    closes = df["Close"].tail(VWAP_CONFIRM_BARS)
    vwaps = df["VWAP"].tail(VWAP_CONFIRM_BARS)
    if closes.shape[0] < VWAP_CONFIRM_BARS:
        return "Mixed/None", 0

    if (closes > vwaps).all():
        return "Above", VWAP_CONFIRM_BARS
    if (closes < vwaps).all():
        return "Below", VWAP_CONFIRM_BARS
    return "Mixed/None", 0

def momentum_state(df: pd.DataFrame) -> str:
    if df.empty or "EMA9_slope" not in df.columns:
        return "Mixed"
    s = safe_float(df["EMA9_slope"].iloc[-1])
    if np.isnan(s):
        return "Mixed"
    return "Up" if s > 0 else ("Down" if s < 0 else "Flat")

def pace_label(mkt_state: str, atr: float, bw: float, rvol: float) -> str:
    if mkt_state == "Chop":
        return "Stalled"
    score = 0.0
    if not np.isnan(rvol):
        score += min(max((rvol - 1.0) * 25, 0), 40)
    if not np.isnan(bw):
        score += min(max(bw * 120, 0), 35)
    if not np.isnan(atr):
        score += min(max(atr * 2, 0), 25)

    if score >= 70:
        return "Fast"
    if score >= 45:
        return "Normal"
    return "Extended"

def strength_bucket(bias: str, mkt_state: str, vwap_side: str, mom: str, rvol: float) -> tuple[str, int]:
    score = 0
    if bias in ("Bullish Bias", "Bearish Bias"):
        score += 15
    if mkt_state == "Trend":
        score += 20
    if vwap_side in ("Above", "Below"):
        score += 25
    if mom in ("Up", "Down"):
        score += 20
    if not np.isnan(rvol) and rvol >= RVOL_ENTRY_MIN:
        score += 20

    if mkt_state == "Chop":
        score -= 25
    if vwap_side == "Mixed/None":
        score -= 20
    if mom in ("Flat", "Mixed"):
        score -= 10
    if not np.isnan(rvol) and rvol < 1.0:
        score -= 10

    score = int(max(0, min(100, score)))
    if score >= 80:
        return "Extreme", score
    if score >= 60:
        return "High", score
    if score >= 40:
        return "Medium", score
    return "Low", score

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
    else:  # Bearish
        entry = vwap
        target_low = entry - 3.0 * atr
        target_high = entry - 1.5 * atr
        exit_if = entry + 1.25 * atr

    lo = float(min(target_low, target_high))
    hi = float(max(target_low, target_high))

    pct_lo = (lo - current) / current * 100 if current else 0.0
    pct_hi = (hi - current) / current * 100 if current else 0.0
    usd_lo = lo - current
    usd_hi = hi - current

    return {
        "current": float(current),
        "vwap": float(vwap),
        "atr": float(atr),
        "entry": float(entry),
        "target_low": lo,
        "target_high": hi,
        "exit_if": float(exit_if),
        "pct_low": float(pct_lo),
        "pct_high": float(pct_hi),
        "usd_low": float(usd_lo),
        "usd_high": float(usd_hi),
    }

def format_price(x: float) -> str:
    if np.isnan(x):
        return "—"
    return f"{x:,.2f}"

# ============================================================
# SESSION STATE
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = {}

def get_asset_state(key: str) -> dict:
    if key not in st.session_state.state:
        st.session_state.state[key] = {
            "status": "STAND DOWN",
            "last_session_id": None,
            "reset_until": None,
            "active_direction": None,  # "Bullish"/"Bearish" when an entry cycle is active
            "last_exit_reason": None,
        }
    return st.session_state.state[key]

def set_state(key: str, status: str, direction: str | None = None, reset_until: datetime | None = None, exit_reason: str | None = None):
    s = get_asset_state(key)
    s["status"] = status
    if direction is not None:
        s["active_direction"] = direction
    if reset_until is not None:
        s["reset_until"] = reset_until
    if exit_reason is not None:
        s["last_exit_reason"] = exit_reason
    st.session_state.state[key] = s

# ============================================================
# UI HEADER
# ============================================================
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="headline">SPY + BTC Command</div>', unsafe_allow_html=True)
    st.markdown('<div class="pulse"></div>', unsafe_allow_html=True)
with right:
    st.markdown(f'<div class="small">Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1])
with c1:
    asset_key = st.selectbox("Asset", options=list(ASSETS.keys()), index=0)
with c2:
    st.button("🔄 Refresh Now")

asset = ASSETS[asset_key]
ticker = asset["ticker"]
asset_type = asset["type"]

# ============================================================
# DATA
# ============================================================
df_raw = fetch_intraday(ticker)
df_sess, session_id = filter_session(df_raw, asset_type)

s = get_asset_state(asset_key)

# hard reset at new session boundary
if session_id and s.get("last_session_id") != session_id:
    s["last_session_id"] = session_id
    s["status"] = "STAND DOWN"
    s["reset_until"] = None
    s["active_direction"] = None
    s["last_exit_reason"] = None
    st.session_state.state[asset_key] = s

df = compute_indicators(df_sess)
if df.empty or df.shape[0] < 30:
    st.warning("Not enough intraday session data yet. Let a few 5-minute candles print.")
    st.stop()

# ============================================================
# CORE READS
# ============================================================
mkt_state, bw_now = bandwidth_state(df)
vwap_side, _ = vwap_confirmed_side(df)
mom = momentum_state(df)

last = df.iloc[-1]
current = safe_float(last["Close"])
vwap = safe_float(last["VWAP"])
atr = safe_float(last.get("ATR", np.nan))
rvol = safe_float(last.get("RVOL", np.nan))

# Bias is intraday and session-scoped
if np.isnan(current) or np.isnan(vwap):
    bias = "Mixed Bias"
else:
    bias = "Bullish Bias" if current >= vwap else "Bearish Bias"

strength, strength_score = strength_bucket(bias, mkt_state, vwap_side, mom, rvol)
pace = pace_label(mkt_state, atr, bw_now, rvol)

# ============================================================
# HEADS UP (NOT CONFIRMED)
# ============================================================
vol_prev = safe_float(df["Volume"].iloc[-2]) if df.shape[0] >= 2 else np.nan
vol_now = safe_float(df["Volume"].iloc[-1])

near_vwap = False
if not np.isnan(atr) and not np.isnan(current) and not np.isnan(vwap):
    near_vwap = abs(current - vwap) <= (HEADSUP_VWAP_ATR_BAND * atr)

momentum_improving = False
if "EMA9_slope" in df.columns and df["EMA9_slope"].shape[0] >= 5:
    s_now = safe_float(df["EMA9_slope"].iloc[-1])
    s_prev = safe_float(df["EMA9_slope"].iloc[-3])
    momentum_improving = (not np.isnan(s_now) and not np.isnan(s_prev) and (s_now > s_prev))

participation_rising = (
    not np.isnan(vol_now) and not np.isnan(vol_prev) and vol_now > vol_prev
    and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN)
)

bw_series = df["BB_bw"].dropna()
bw_improving = False
if bw_series.shape[0] >= 3:
    bw_improving = safe_float(bw_series.iloc[-1]) > safe_float(bw_series.iloc[-2])

# Heads Up only when not already in an active move AND market not fully Trend yet
heads_up = (mkt_state == "Chop" and near_vwap and momentum_improving and participation_rising and bw_improving)

# ============================================================
# ENTRY ACTIVE (CONFIRMED) CONDITIONS
# ============================================================
confirmed_long = (
    bias == "Bullish Bias"
    and mkt_state == "Trend"
    and vwap_side == "Above"
    and mom == "Up"
    and (not np.isnan(rvol) and rvol >= RVOL_ENTRY_MIN)
)

confirmed_short = (
    bias == "Bearish Bias"
    and mkt_state == "Trend"
    and vwap_side == "Below"
    and mom == "Down"
    and (not np.isnan(rvol) and rvol >= RVOL_ENTRY_MIN)
)

# ============================================================
# LEVELS FOR EXIT CHECKS
# ============================================================
levels_long = compute_levels(df, "Bullish")
levels_short = compute_levels(df, "Bearish")

# ============================================================
# CAUTION (LATER): ONLY IF 2+ WEAKENING CONDITIONS
# ============================================================
def caution_for(direction: str) -> bool:
    """
    LATER (more aggressive):
    CAUTION only triggers if still on the correct VWAP side
    AND at least TWO weakening conditions are present.
    """
    if direction == "Bullish":
        still_side = (current >= vwap)

        weak_count = 0
        if mkt_state != "Trend":
            weak_count += 1
        if mom != "Up":
            weak_count += 1
        if np.isnan(rvol) or rvol < RVOL_ENTRY_MIN:
            weak_count += 1
        if vwap_side != "Above":
            weak_count += 1

        return bool(still_side and weak_count >= 2)

    if direction == "Bearish":
        still_side = (current <= vwap)

        weak_count = 0
        if mkt_state != "Trend":
            weak_count += 1
        if mom != "Down":
            weak_count += 1
        if np.isnan(rvol) or rvol < RVOL_ENTRY_MIN:
            weak_count += 1
        if vwap_side != "Below":
            weak_count += 1

        return bool(still_side and weak_count >= 2)

    return False

# ============================================================
# EXIT / RESET RULES
# ============================================================
def exit_triggered(active_dir: str) -> tuple[bool, str]:
    """
    Exit triggers when the move is done (confirmed flip) or emergency ATR stop zone is breached.
    """
    if active_dir == "Bullish":
        if vwap_side == "Below":
            return True, "VWAP flip confirmed"
        if current <= levels_long["exit_if"]:
            return True, "Emergency ATR stop"
    elif active_dir == "Bearish":
        if vwap_side == "Above":
            return True, "VWAP flip confirmed"
        if current >= levels_short["exit_if"]:
            return True, "Emergency ATR stop"
    return False, ""

# ============================================================
# STATE MACHINE (NO IN PLAY, MARKET-STATE ONLY)
# ============================================================
now_utc = datetime.now(timezone.utc)

s = get_asset_state(asset_key)
status = s.get("status", "STAND DOWN")
active_dir = s.get("active_direction")
reset_until = s.get("reset_until")

# If inside reset window
in_reset = reset_until is not None and isinstance(reset_until, datetime) and now_utc < reset_until
if in_reset:
    set_state(asset_key, "RESET")
    status = "RESET"
else:
    # If we have an active direction, manage it
    if active_dir in ("Bullish", "Bearish"):
        do_exit, reason = exit_triggered(active_dir)
        if do_exit:
            # Exit -> immediate cooloff
            set_state(
                asset_key,
                "EXIT / RESET",
                reset_until=now_utc + timedelta(minutes=RESET_MINUTES),
                exit_reason=reason
            )
            # clear active direction explicitly
            s = get_asset_state(asset_key)
            s["active_direction"] = None
            st.session_state.state[asset_key] = s
            status = "EXIT / RESET"
        else:
            # no exit, decide if still Entry Active or caution or stand down
            if active_dir == "Bullish":
                if confirmed_long:
                    set_state(asset_key, "ENTRY ACTIVE", direction="Bullish")
                    status = "ENTRY ACTIVE"
                elif caution_for("Bullish"):
                    set_state(asset_key, "CAUTION", direction="Bullish")
                    status = "CAUTION"
                else:
                    # edge gone (but no confirmed flip) -> wait or stand down
                    if mkt_state == "Chop":
                        set_state(asset_key, "STAND DOWN")
                        status = "STAND DOWN"
                    else:
                        set_state(asset_key, "WAITING")
                        status = "WAITING"

            elif active_dir == "Bearish":
                if confirmed_short:
                    set_state(asset_key, "ENTRY ACTIVE", direction="Bearish")
                    status = "ENTRY ACTIVE"
                elif caution_for("Bearish"):
                    set_state(asset_key, "CAUTION", direction="Bearish")
                    status = "CAUTION"
                else:
                    if mkt_state == "Chop":
                        set_state(asset_key, "STAND DOWN")
                        status = "STAND DOWN"
                    else:
                        set_state(asset_key, "WAITING")
                        status = "WAITING"

    else:
        # No active move: can we start one?
        if confirmed_long:
            set_state(asset_key, "ENTRY ACTIVE", direction="Bullish")
            status = "ENTRY ACTIVE"
        elif confirmed_short:
            set_state(asset_key, "ENTRY ACTIVE", direction="Bearish")
            status = "ENTRY ACTIVE"
        else:
            if heads_up:
                set_state(asset_key, "HEADS UP")
                status = "HEADS UP"
            elif mkt_state == "Chop":
                set_state(asset_key, "STAND DOWN")
                status = "STAND DOWN"
            else:
                set_state(asset_key, "WAITING")
                status = "WAITING"

# Reload final state
s = get_asset_state(asset_key)
status = s["status"]
active_dir = s.get("active_direction")
last_exit_reason = s.get("last_exit_reason")

# ============================================================
# UI: COMMAND STRIP
# ============================================================
def color_class_for_bias(b: str) -> str:
    if b.startswith("Bullish"):
        return "good"
    if b.startswith("Bearish"):
        return "bad"
    return "neutral"

def color_class_for_state(stt: str) -> str:
    return "good" if stt == "Trend" else "neutral"

def color_class_for_status(sts: str) -> str:
    if sts == "ENTRY ACTIVE":
        return "good" if bias.startswith("Bullish") else ("bad" if bias.startswith("Bearish") else "neutral")
    if sts in ("CAUTION", "HEADS UP"):
        return "warn"
    if sts == "EXIT / RESET":
        return "bad"
    return "neutral"

cA, cB, cC = st.columns(3)

with cA:
    st.markdown(
        f"""
        <div class="card2">
          <div class="label">Bias</div>
          <div class="headline {color_class_for_bias(bias)}">{bias.replace(" Bias","")}</div>
          <span class="pill">Intraday</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cB:
    st.markdown(
        f"""
        <div class="card2">
          <div class="label">Market State</div>
          <div class="headline {color_class_for_state(mkt_state)}">{mkt_state}</div>
          <span class="pill">{'Tradable' if mkt_state=='Trend' else 'Chop Gate'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cC:
    st.markdown(
        f"""
        <div class="card2">
          <div class="label">Status</div>
          <div class="headline {color_class_for_status(status)}">{status}</div>
          <span class="pill">{strength} ({strength_score}/100)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# PRIMARY ACTION CARD
# ============================================================
direction = active_dir if active_dir in ("Bullish", "Bearish") else ("Bullish" if bias.startswith("Bullish") else "Bearish")
levels = compute_levels(df, direction)

def action_body(levels: dict, direction: str, status: str) -> str:
    entry = format_price(levels["entry"])
    tlo = format_price(levels["target_low"])
    thi = format_price(levels["target_high"])
    exi = format_price(levels["exit_if"])
    cur = format_price(levels["current"])

    pct_low = levels["pct_low"]
    pct_high = levels["pct_high"]
    usd_low = levels["usd_low"]
    usd_high = levels["usd_high"]

    pct_txt = f"{pct_low:+.2f}% to {pct_high:+.2f}%"
    usd_txt = f"{usd_low:+.2f} to {usd_high:+.2f}"

    entry_txt = f"Above {entry}" if direction == "Bullish" else f"Below {entry}"
    exit_txt = f"Below {exi}" if direction == "Bullish" else f"Above {exi}"

    caution_note = ""
    if status == "CAUTION":
        caution_note = '<div class="small warn">CAUTION: Late entries are higher risk (2+ weakening signals detected).</div>'

    return f"""
    {caution_note}
    <div class="subline"><span class="label">Current</span><br><span class="headline neutral">{cur}</span></div>
    <div class="hr"></div>

    <div class="subline"><span class="label">Entry (Session VWAP)</span><br><span class="headline good">{entry_txt}</span></div>

    <div class="subline" style="margin-top:10px;">
      <span class="label">Target Zone</span><br>
      <span class="headline neutral">{tlo} – {thi}</span>
      <div class="small">{pct_txt} / {usd_txt}</div>
    </div>

    <div class="subline" style="margin-top:10px;">
      <span class="label">Exit If</span><br>
      <span class="headline bad">{exit_txt}</span>
      <div class="small">Primary exit: VWAP flip confirmed (2 bars). Emergency: ATR stop.</div>
    </div>

    <div class="hr"></div>
    <span class="pill">PACE: {pace}</span>
    """

def render_card(title: str, subtitle: str, body_html: str):
    st.markdown(
        f"""
        <div class="card">
          <div class="label">{title}</div>
          <div class="headline neutral">{subtitle}</div>
          <div class="hr"></div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True
    )

if status in ("ENTRY ACTIVE", "CAUTION"):
    render_card("CONFIRMED MOVE", f"{direction} — {status}", action_body(levels, direction, status))

elif status == "EXIT / RESET":
    reason_line = f"Reason: {last_exit_reason}" if last_exit_reason else "Reason: Move ended"
    body = f"""
    <div class="subline"><span class="headline bad">EXIT / RESET</span></div>
    <div class="subline">{reason_line}</div>
    <div class="subline">Cooling off to avoid chop re-entries. Then we hunt again.</div>
    <div class="hr"></div>
    <div class="small">Signals are session-based and reset daily.</div>
    """
    render_card("CONFIRMED MOVE", "Exit executed", body)

elif status == "RESET":
    body = f"""
    <div class="subline"><span class="headline neutral">RESET</span></div>
    <div class="subline">Cooling off to avoid chop re-entries.</div>
    <div class="hr"></div>
    <div class="small">Next: Stand Down / Heads Up / Entry Active depending on conditions.</div>
    """
    render_card("CONFIRMED MOVE", "Cooling off", body)

else:
    body = f"""
    <div class="subline"><span class="headline neutral">No active entry right now.</span></div>
    <div class="subline">Waiting for confirmed conditions (VWAP hold + Trend + participation).</div>
    <div class="hr"></div>
    <span class="pill">Status: {status}</span>
    """
    render_card("CONFIRMED MOVE", f"{bias} • {mkt_state}", body)

# ============================================================
# SECONDARY HEADS UP CARD (SEPARATE)
# ============================================================
st.write("")
if status == "HEADS UP":
    bullets = []
    if near_vwap:
        bullets.append("Pressure building near VWAP")
    if momentum_improving:
        bullets.append("Momentum improving")
    if participation_rising:
        bullets.append("Participation rising")
    if bw_improving:
        bullets.append("Volatility waking up")

    bullets_html = "".join([f"<li>{b}</li>" for b in bullets]) if bullets else "<li>Monitoring conditions</li>"

    st.markdown(
        f"""
        <div class="card">
          <div class="label">⚠️ HEADS UP — Not Confirmed</div>
          <div class="headline warn">Pay attention — don’t act yet.</div>
          <div class="hr"></div>
          <ul style="margin:0; padding-left:18px;">{bullets_html}</ul>
          <div class="hr"></div>
          <div class="small">Awareness only. Confirmed moves show as ENTRY ACTIVE above.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DETAILS (COLLAPSED)
# ============================================================
st.write("")
with st.expander("Details (advanced)", expanded=False):
    st.write("**Session-scoped signals:** reset daily (SPY session ET, BTC day UTC).")
    st.write(f"**Session ID:** {session_id}")
    st.write("")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Current", format_price(current))
        st.metric("VWAP", format_price(vwap))
        st.metric("VWAP Confirm Side", vwap_side)
    with cols[1]:
        st.metric("EMA9", format_price(safe_float(last.get("EMA9", np.nan))))
        st.metric("Momentum", mom)
        st.metric("ATR", format_price(atr))
    with cols[2]:
        st.metric("BB Width", f"{bw_now:.4f}" if not np.isnan(bw_now) else "—")
        st.metric("RVOL", f"{rvol:.2f}" if not np.isnan(rvol) else "—")
        st.metric("Market State", mkt_state)

    if status not in ("ENTRY ACTIVE", "CAUTION"):
        reasons = []
        if mkt_state != "Trend":
            reasons.append("Market State not Trend (bandwidth not expanding cleanly).")
        if vwap_side == "Mixed/None":
            reasons.append("VWAP not confirmed (2-bar hold not satisfied).")
        if bias == "Bullish Bias" and mom != "Up":
            reasons.append("Momentum not aligned (EMA9 slope not up).")
        if bias == "Bearish Bias" and mom != "Down":
            reasons.append("Momentum not aligned (EMA9 slope not down).")
        if np.isnan(rvol) or rvol < RVOL_ENTRY_MIN:
            reasons.append("Participation not confirmed (RVOL below threshold).")

        if reasons:
            st.write("### Why Not (blocked conditions)")
            for r in reasons:
                st.write(f"- {r}")

st.caption("Signals are session-based and reset daily. Heads Up ≠ Entry. ENTRY ACTIVE means the move is still valid; CAUTION means late entries are higher risk.")