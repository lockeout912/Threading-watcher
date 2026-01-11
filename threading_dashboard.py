import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="SPY + BTC Signal Engine", layout="wide")

# ============================================================
# CLEAN WALL STREET UI
# ============================================================
st.markdown(
    """
<style>
.stApp{
  background: #0b1220;
  color:#e8eefc;
}
*{-webkit-font-smoothing:antialiased}
#MainMenu, footer, header {visibility:hidden;}

.wrap{max-width:1100px; margin:0 auto;}

.decision{
  border:1px solid rgba(255,255,255,0.12);
  border-radius:20px;
  padding:18px 18px;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 14px 44px rgba(0,0,0,0.40);
}

.toprow{
  display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
}
.asset{
  font-size:12px; opacity:0.8; letter-spacing:0.35px; text-transform:uppercase;
}
.price{
  font-size:44px; font-weight:900; letter-spacing:0.5px; line-height:1.05;
}
.meta{
  font-size:12px; opacity:0.75;
}

.badge{
  display:inline-block; padding:7px 12px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.06);
  font-weight:900; letter-spacing:0.35px;
  text-transform:uppercase; font-size:12px;
}

.good{color:#6cf2a5}
.bad{color:#ff6b6b}
.warn{color:#ffd166}
.neutral{color:#b9c3d9}

.hr{height:1px; background:rgba(255,255,255,0.12); margin:12px 0;}
.row{display:flex; gap:12px; flex-wrap:wrap;}

.kpi{
  flex:1; min-width:220px;
  border:1px solid rgba(255,255,255,0.10);
  border-radius:16px;
  padding:12px 12px;
  background: rgba(255,255,255,0.03);
}
.klabel{font-size:12px; opacity:0.75; text-transform:uppercase; letter-spacing:0.35px;}
.kvalue{font-size:18px; font-weight:900; margin-top:4px;}
.khint{font-size:12px; opacity:0.70; margin-top:4px;}

.flow{
  display:flex; gap:10px; flex-wrap:wrap;
}
.step{
  padding:7px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  font-size:12px; font-weight:900;
}
.stepOn{
  background: rgba(255,255,255,0.10);
  border-color: rgba(255,255,255,0.22);
}
.note{
  font-size:14px; opacity:0.92; line-height:1.35;
}
.small{font-size:12px; opacity:0.72}
</style>
""",
    unsafe_allow_html=True,
)

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

@st.cache_data(ttl=20)
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
# SESSION STATE (MARKET-ONLY)
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
# HEADER + CONTROLS (CLEAN)
# ============================================================
st.markdown('<div class="wrap">', unsafe_allow_html=True)

colA, colB, colC = st.columns([1.1, 1, 1])
with colA:
    asset_key = st.selectbox("Asset", list(ASSETS.keys()), index=0)
with colB:
    st.button("🔄 Refresh")
with colC:
    auto = st.toggle("Auto-refresh (every 20s)", value=False)

if auto:
    st.caption("Auto-refresh is ON.")
    st.experimental_set_query_params(_=str(datetime.now().timestamp()))
    st.autorefresh(interval=20_000, key="refresh")

asset = ASSETS[asset_key]
ticker = asset["ticker"]
atype = asset["type"]

# ============================================================
# DATA LOAD
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
current = safe_float(last["Close"])
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
# HEADS UP (SIMPLE)
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
    momentum_improving = (not np.isnan(s_now) and not np.isnan(s_prev) and s_now > s_prev)

participation_rising = (
    not np.isnan(vol_now) and not np.isnan(vol_prev) and vol_now > vol_prev
    and (np.isnan(rvol) or rvol >= RVOL_HEADSUP_MIN)
)

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
# STATE MACHINE (CLEAR FLOW)
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
                last_exit_reason=reason,
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
last_exit_reason = s.get("last_exit_reason")

# ============================================================
# DECISION TEXT (USER DOESN’T THINK)
# ============================================================
def status_color(sts: str) -> str:
    if sts == "ENTRY ACTIVE":
        return "good"
    if sts in ("CAUTION", "HEADS UP"):
        return "warn"
    if sts in ("EXIT / RESET",):
        return "bad"
    return "neutral"

def action_script(sts: str, direction: str | None) -> tuple[str, str]:
    # title, what to do
    if sts == "ENTRY ACTIVE" and direction:
        if direction == "Bullish":
            return "ENTER LONG (still valid)", "Buy calls / long shares while price holds above VWAP. Use Exit-If below."
        return "ENTER SHORT (still valid)", "Buy puts / short exposure while price holds below VWAP. Use Exit-If below."
    if sts == "CAUTION" and direction:
        return "CAUTION (late entry risk)", "If you enter now, size down. Prefer waiting for re-confirmation or a fresh setup."
    if sts == "HEADS UP":
        return "HEADS UP (do not enter yet)", "Conditions are building. Wait for ENTRY ACTIVE confirmation."
    if sts == "EXIT / RESET":
        return "EXIT / STAND DOWN", "Move is done. Don’t chase. Wait for the next ENTRY ACTIVE cycle."
    if sts == "RESET":
        return "RESET (cooling off)", "No trades during reset. We’re avoiding chop whipsaws."
    if sts == "WAITING":
        return "WAITING (setup forming)", "Trend exists but confirmation isn’t clean yet. Patience pays."
    return "STAND DOWN (chop)", "No edge right now. Protect capital. Wait."

# ============================================================
# SINGLE DECISION CARD (CLEAN)
# ============================================================
direction = active_dir if active_dir in ("Bullish", "Bearish") else ("Bullish" if bias == "Bullish" else "Bearish")
levels = compute_levels(df, "Bullish" if direction == "Bullish" else "Bearish")

title, instruction = action_script(status, active_dir)
badge_cls = status_color(status)

entry_line = f"Above {fmt(levels['entry'])}" if direction == "Bullish" else f"Below {fmt(levels['entry'])}"
exit_line  = f"Below {fmt(levels['exit_if'])}" if direction == "Bullish" else f"Above {fmt(levels['exit_if'])}"

tlo, thi = fmt(levels["target_low"]), fmt(levels["target_high"])
pct_txt = f"{levels['pct_low']:+.2f}% to {levels['pct_high']:+.2f}%"
usd_txt = f"{levels['usd_low']:+.2f} to {levels['usd_high']:+.2f}"

flow_steps = ["HEADS UP", "ENTRY ACTIVE", "CAUTION", "EXIT / RESET", "RESET"]
def step_on(step: str) -> str:
    return "step stepOn" if status == step else "step"

flow_html = "".join([f'<span class="{step_on(x)}">{x}</span>' for x in flow_steps])

why_one_liner = f"Trend={mkt_state} • VWAP={vwap_side} • Mom={mom} • RVOL={('—' if np.isnan(rvol) else f'{rvol:.2f}')}"
if status == "EXIT / RESET" and last_exit_reason:
    why_one_liner = f"Exit reason: {last_exit_reason}"

st.markdown(
    f"""
<div class="wrap">
  <div class="decision">
    <div class="toprow">
      <div>
        <div class="asset">{asset_key} • {ticker} • {INTERVAL} • Session {session_id}</div>
        <div class="price">{fmt(current)}</div>
        <div class="meta">Last candle time: {df.index[-1]}</div>
      </div>
      <div style="text-align:right;">
        <div class="badge {badge_cls}">{status}</div><br>
        <div style="margin-top:10px;" class="badge neutral">{title}</div>
      </div>
    </div>

    <div class="hr"></div>

    <div class="note"><b>What to do:</b> {instruction}</div>
    <div class="small" style="margin-top:6px;">{why_one_liner}</div>

    <div class="hr"></div>

    <div class="row">
      <div class="kpi">
        <div class="klabel">Entry Rule</div>
        <div class="kvalue good">{entry_line}</div>
        <div class="khint">Confirmation uses 2-bar VWAP hold + Trend + Momentum + Participation.</div>
      </div>

      <div class="kpi">
        <div class="klabel">Target Zone</div>
        <div class="kvalue neutral">{tlo} – {thi}</div>
        <div class="khint">{pct_txt} / {usd_txt}</div>
      </div>

      <div class="kpi">
        <div class="klabel">Exit-If</div>
        <div class="kvalue bad">{exit_line}</div>
        <div class="khint">Primary: VWAP flips. Backup: ATR emergency stop.</div>
      </div>
    </div>

    <div class="hr"></div>

    <div class="flow">{flow_html}</div>
    <div class="small" style="margin-top:10px;">Signals are session-based. No confusion, no long-term mixing.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# OPTIONAL: ADVANCED DETAILS (HIDDEN)
# ============================================================
with st.expander("Advanced (optional)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VWAP", fmt(vwap))
    c2.metric("ATR", fmt(atr))
    c3.metric("RVOL", "—" if np.isnan(rvol) else f"{rvol:.2f}")
    c4.metric("BB Width", "—" if np.isnan(bw_now) else f"{bw_now:.4f}")

st.caption("Heads Up ≠ Entry. ENTRY ACTIVE = still tradable. CAUTION = late entry risk. EXIT/RESET = move ended, wait.")
st.markdown("</div>", unsafe_allow_html=True)