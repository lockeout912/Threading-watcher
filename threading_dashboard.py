# app.py
import time
from datetime import datetime, time as dtime

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lockout Signals • SPY + BTC", layout="wide")


# ============================================================
# UTIL: MARKET STATUS
# ============================================================
def market_status_for(asset: str) -> str:
    """SPY uses US equities hours; BTC is 24/7."""
    if asset.upper() in ["BTC", "BTC-USD", "BITCOIN", "ETH", "ETH-USD", "ETHEREUM"]:
        return "24/7"

    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern).time()

    if dtime(9, 30) <= now <= dtime(16, 0):
        return "MARKET OPEN"
    elif dtime(4, 0) <= now < dtime(9, 30):
        return "PRE-MARKET"
    else:
        return "AFTER HOURS"


# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=12)
def fetch_intraday(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch intraday candles with best-effort prepost support."""
    try:
        df = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            prepost=True,   # IMPORTANT: helps for BTC, sometimes helps for SPY
            actions=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = df.reset_index()
        # Normalize column name (sometimes it's 'Datetime' or 'Date')
        if "Datetime" in df.columns:
            ts_col = "Datetime"
        elif "Date" in df.columns:
            ts_col = "Date"
        else:
            ts_col = df.columns[0]
        df = df.rename(columns={ts_col: "ts"})
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def fetch_daily(ticker: str, period: str = "10d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", actions=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return pd.DataFrame()


def get_yahoo_regular_market_price(ticker: str):
    """Option B: Try Yahoo's regularMarketPrice (can lag; best-effort)."""
    try:
        info = yf.Ticker(ticker).info
        p = info.get("regularMarketPrice")
        if p is None:
            return None
        p = float(p)
        if p <= 0:
            return None
        return p
    except Exception:
        return None


# ============================================================
# INDICATORS
# ============================================================
def add_ema(df: pd.DataFrame, span: int, price_col: str = "Close") -> pd.Series:
    return df[price_col].ewm(span=span, adjust=False).mean()


def add_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP (cumulative)."""
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    v = d["Volume"].fillna(0)
    pv = (tp * v).cumsum()
    vv = v.cumsum().replace(0, pd.NA)
    return (pv / vv).fillna(method="ffill")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR using True Range on provided timeframe."""
    d = df.copy()
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            (d["High"] - d["Low"]).abs(),
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def slope(series: pd.Series, lookback: int = 8) -> float:
    """Simple slope proxy: last - value N bars ago."""
    if series is None or len(series) < lookback + 1:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-lookback - 1])


# ============================================================
# LEVELS (THE "MAP")
# ============================================================
def compute_levels(symbol: str, df_5m: pd.DataFrame, df_1m: pd.DataFrame):
    levels = {}

    # Prior day levels (SPY mostly; for BTC it'll still compute from daily candles)
    daily = fetch_daily(symbol, "12d")
    if daily is not None and not daily.empty and len(daily) >= 2:
        prev = daily.iloc[-2]
        levels["PDH"] = float(prev["High"])
        levels["PDL"] = float(prev["Low"])
        levels["PDC"] = float(prev["Close"])

    # Opening Range (first 5 minutes from today's session in df_5m)
    if df_5m is not None and not df_5m.empty:
        # approximate "session day" by last candle's date
        last_day = df_5m["ts"].iloc[-1].date()
        today = df_5m[df_5m["ts"].dt.date == last_day].copy()
        if not today.empty:
            first_bar = today.iloc[0]
            levels["OPEN"] = float(first_bar["Open"])

            # OR = first 1 bar of 5m (true opening range is often first 5m)
            or_bar = today.iloc[0:1]
            levels["ORH"] = float(or_bar["High"].max())
            levels["ORL"] = float(or_bar["Low"].min())

    # VWAP (from 1m if available else 5m)
    vwap_now = None
    if df_1m is not None and not df_1m.empty:
        vwap_now = float(df_1m["VWAP"].iloc[-1])
    elif df_5m is not None and not df_5m.empty:
        vwap_now = float(df_5m["VWAP"].iloc[-1])

    if vwap_now is not None:
        levels["VWAP"] = vwap_now

    return levels


def nearest_level(levels: dict, price: float):
    """Find the closest major level and return (name, value, distance)."""
    if not levels:
        return None, None, None
    items = [(k, float(v)) for k, v in levels.items() if v is not None]
    if not items:
        return None, None, None
    best = min(items, key=lambda kv: abs(kv[1] - price))
    return best[0], best[1], abs(best[1] - price)


# ============================================================
# BIAS + REGIME (5m + 15m)
# ============================================================
def compute_bias_regime(df_5m: pd.DataFrame, df_15m: pd.DataFrame):
    # Defaults
    bias = "NEUTRAL"
    regime = "RANGE"

    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return bias, regime, 0

    px = float(df_5m["Close"].iloc[-1])
    vwap = float(df_5m["VWAP"].iloc[-1])
    ema9 = float(df_5m["EMA9"].iloc[-1])
    ema21 = float(df_5m["EMA21"].iloc[-1])
    vwap_slope = slope(df_5m["VWAP"], lookback=8)
    atr5 = float(df_5m["ATR"].iloc[-1]) if pd.notna(df_5m["ATR"].iloc[-1]) else 0.0

    # 15m context filter
    ctx_ok_bull = True
    ctx_ok_bear = True
    if df_15m is not None and not df_15m.empty and len(df_15m) > 20:
        ema9_15 = float(df_15m["EMA9"].iloc[-1])
        ema21_15 = float(df_15m["EMA21"].iloc[-1])
        ctx_ok_bull = ema9_15 >= ema21_15
        ctx_ok_bear = ema9_15 <= ema21_15

    # Regime: trend if VWAP slope meaningful and price separated from VWAP
    if atr5 and atr5 > 0:
        sep = abs(px - vwap)
        if abs(vwap_slope) > 0.12 * atr5 and sep > 0.18 * atr5:
            regime = "TREND"
        else:
            regime = "RANGE"

    # Bias
    bull = (px > vwap) and (ema9 >= ema21) and (vwap_slope >= 0)
    bear = (px < vwap) and (ema9 <= ema21) and (vwap_slope <= 0)

    if bull and ctx_ok_bull:
        bias = "BULLISH"
    elif bear and ctx_ok_bear:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Score (0-100) simple & fast (aggressive leaning)
    score = 0
    if bias == "BULLISH":
        score += 55
    elif bias == "BEARISH":
        score += 55

    if regime == "TREND":
        score += 15
    else:
        score += 5

    # Bonus for momentum separation from VWAP
    if atr5 and atr5 > 0:
        score += int(min(30, (abs(px - vwap) / atr5) * 20))

    score = max(0, min(100, score))
    return bias, regime, score


# ============================================================
# TRIGGERS (1m) — reclaim/reject engine
# ============================================================
def classify_action(df_1m: pd.DataFrame, df_5m: pd.DataFrame, levels: dict, bias: str):
    """
    Returns:
      action: ENTER / WAIT / CAUTION / EXIT
      side: CALLS / PUTS / WAIT
      why: one-liner
      invalid: price level
      key_level: (name, value)
    """
    if df_1m is None or df_1m.empty or len(df_1m) < 40 or df_5m is None or df_5m.empty:
        return "WAIT", "WAIT", "Not enough intraday candles yet.", None, (None, None)

    px = float(df_1m["Close"].iloc[-1])
    vwap = float(df_1m["VWAP"].iloc[-1])

    atr5 = float(df_5m["ATR"].iloc[-1]) if pd.notna(df_5m["ATR"].iloc[-1]) else 0.0
    threshold = (0.18 * atr5) if atr5 and atr5 > 0 else (0.25)

    # Determine the key level “in play” (closest)
    lvl_name, lvl_val, lvl_dist = nearest_level(levels, px)
    if lvl_name is None:
        lvl_name, lvl_val, lvl_dist = "VWAP", vwap, abs(px - vwap)

    # If not near any level, default to VWAP as the “level in play”
    if lvl_dist is None or lvl_dist > threshold:
        lvl_name, lvl_val = "VWAP", vwap

    # 1m micro momentum
    ema9_1 = df_1m["EMA9"]
    ema21_1 = df_1m["EMA21"]
    slope_ema9 = slope(ema9_1, lookback=6)

    # Recent candle behavior around the level
    recent = df_1m.tail(6).copy()
    closes = list(map(float, recent["Close"].values))
    highs = list(map(float, recent["High"].values))
    lows = list(map(float, recent["Low"].values))

    # Helpers: “hold above/below”
    def held_above(level: float, n: int = 3):
        return all(c > level for c in closes[-n:])

    def held_below(level: float, n: int = 3):
        return all(c < level for c in closes[-n:])

    # Reclaim / Reject
    # reclaim = traded below then closes above and holds
    reclaim = (min(lows[-4:]) < lvl_val) and (closes[-1] > lvl_val)
    reject = (max(highs[-4:]) > lvl_val) and (closes[-1] < lvl_val)

    # Aggressive Heads Up
    heads_up_long = (bias in ["BULLISH", "NEUTRAL"]) and reclaim and (closes[-1] > vwap)
    heads_up_short = (bias in ["BEARISH", "NEUTRAL"]) and reject and (closes[-1] < vwap)

    # Confirmed entry
    enter_long = (bias == "BULLISH") and reclaim and held_above(lvl_val, 2) and (ema9_1.iloc[-1] >= ema21_1.iloc[-1])
    enter_short = (bias == "BEARISH") and reject and held_below(lvl_val, 2) and (ema9_1.iloc[-1] <= ema21_1.iloc[-1])

    # Exit logic (simple & decisive)
    exit_long = (closes[-1] < vwap) and (ema9_1.iloc[-1] < ema21_1.iloc[-1])
    exit_short = (closes[-1] > vwap) and (ema9_1.iloc[-1] > ema21_1.iloc[-1])

    # Caution: momentum fades after move starts
    caution_long = (bias == "BULLISH") and (slope_ema9 < 0) and (closes[-1] < float(ema9_1.iloc[-1]))
    caution_short = (bias == "BEARISH") and (slope_ema9 > 0) and (closes[-1] > float(ema9_1.iloc[-1]))

    # Decide
    if enter_long:
        invalid = min(vwap, lvl_val) if lvl_name != "PDL" else levels.get("PDL", vwap)
        return "ENTRY ACTIVE", "CALLS", f"Reclaimed {lvl_name} and held. VWAP aligned.", float(invalid), (lvl_name, float(lvl_val))

    if enter_short:
        invalid = max(vwap, lvl_val) if lvl_name != "PDH" else levels.get("PDH", vwap)
        return "ENTRY ACTIVE", "PUTS", f"Rejected {lvl_name}. VWAP aligned down.", float(invalid), (lvl_name, float(lvl_val))

    if heads_up_long:
        invalid = vwap
        return "HEADS UP", "CALLS", f"Testing reclaim at {lvl_name}. Watch hold.", float(invalid), (lvl_name, float(lvl_val))

    if heads_up_short:
        invalid = vwap
        return "HEADS UP", "PUTS", f"Testing rejection at {lvl_name}. Watch breakdown.", float(invalid), (lvl_name, float(lvl_val))

    if bias == "BULLISH" and caution_long:
        return "CAUTION", "CALLS", "Momentum fading — protect gains.", levels.get("VWAP", vwap), (lvl_name, float(lvl_val))

    if bias == "BEARISH" and caution_short:
        return "CAUTION", "PUTS", "Momentum fading — protect gains.", levels.get("VWAP", vwap), (lvl_name, float(lvl_val))

    if bias == "BULLISH" and exit_long:
        return "EXIT / RESET", "CALLS", "Lost VWAP + momentum flipped.", levels.get("VWAP", vwap), (lvl_name, float(lvl_val))

    if bias == "BEARISH" and exit_short:
        return "EXIT / RESET", "PUTS", "Reclaimed VWAP + momentum flipped.", levels.get("VWAP", vwap), (lvl_name, float(lvl_val))

    # Neutral / range
    return "WAIT — NO EDGE", ("CALLS" if bias == "BULLISH" else "PUTS" if bias == "BEARISH" else "WAIT"), "No clean reclaim/reject edge yet.", levels.get("VWAP", vwap), (lvl_name, float(lvl_val))


# ============================================================
# TARGETS (ATR(5m) + next level alignment)
# ============================================================
def compute_targets(price: float, bias: str, levels: dict, atr5: float):
    if not atr5 or atr5 <= 0:
        atr5 = max(0.5, price * 0.001)  # fallback

    likely = 0.25 * atr5
    poss = 0.50 * atr5
    stretch = 0.75 * atr5

    if bias == "BEARISH":
        t1 = price - likely
        t2 = price - poss
        t3 = price - stretch
    else:
        t1 = price + likely
        t2 = price + poss
        t3 = price + stretch

    # Align “possible” and “stretch” to next meaningful level if it’s nearby
    level_vals = sorted([float(v) for v in levels.values() if v is not None])
    if level_vals:
        if bias == "BEARISH":
            below = [v for v in level_vals if v < price]
            if below:
                near = max(below)
                if abs(near - t2) < (0.35 * atr5):
                    t2 = near
        else:
            above = [v for v in level_vals if v > price]
            if above:
                near = min(above)
                if abs(near - t2) < (0.35 * atr5):
                    t2 = near

    return float(t1), float(t2), float(t3)


# ============================================================
# UI: CSS + COMMAND RIBBON
# ============================================================
CSS = """
<style>
:root{
  --bg:#0b0f14;
  --panel:#0f1520;
  --text:#e7edf5;
  --muted:#9db0c6;
  --good:#00ff95;
  --bad:#ff4d4d;
  --warn:#ffc44d;
  --cyan:#4de3ff;
}

html, body, [class*="css"] { background-color: var(--bg) !important; }
.block-container { padding-top: 0.6rem; }

.ribbon {
  position: sticky;
  top: 0;
  z-index: 9999;
  background: rgba(10,14,20,0.92);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 10px 14px;
  margin-bottom: 14px;
  overflow: hidden;
}

.marquee {
  white-space: nowrap;
  overflow: hidden;
  position: relative;
}
.marquee span {
  display: inline-block;
  padding-left: 100%;
  animation: marquee 14s linear infinite;
  font-weight: 800;
  letter-spacing: 0.6px;
  color: var(--text);
}
@keyframes marquee {
  0% { transform: translateX(0%); }
  100% { transform: translateX(-100%); }
}

.pill {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  font-weight: 800;
  color: var(--text);
}

.good { color: var(--good); border-color: rgba(0,255,149,0.35); }
.bad  { color: var(--bad);  border-color: rgba(255,77,77,0.35); }
.warn { color: var(--warn); border-color: rgba(255,196,77,0.35); }
.cyan { color: var(--cyan); border-color: rgba(77,227,255,0.35); }

.hero {
  text-align:center;
  padding: 18px 10px 12px 10px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.06);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
}

.symbol { font-size: 34px; font-weight: 900; color: rgba(231,237,245,0.72); letter-spacing:2px; }
.price  { font-size: 86px; line-height: 1.0; font-weight: 1000; margin: 8px 0 10px; }
.action { font-size: 44px; font-weight: 1000; margin: 6px 0 10px; letter-spacing:1px; }

.subline { font-size: 18px; color: var(--muted); font-weight: 650; margin-top: 6px; }
.whyline { font-size: 16px; color: rgba(231,237,245,0.75); margin-top: 6px; }

.krow {
  display:flex;
  gap:10px;
  justify-content:center;
  flex-wrap: wrap;
  margin-top: 10px;
}

.hr { height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }

.targets{
  text-align:center;
  margin-top: 10px;
  font-weight: 900;
  letter-spacing: 0.8px;
}
.targets .label{ color: rgba(231,237,245,0.75); font-weight: 900; }
.targets .val { font-size: 28px; font-weight: 1000; margin-top: 4px; }

.smallnote { text-align:center; color: rgba(231,237,245,0.55); font-size: 13px; margin-top: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.title("Lockout Signals • SPY + BTC")

# ✅ ONLY CHANGE IN THIS ENTIRE FILE: more tickers added here
assets = {
    "SPY": "SPY",
    "BTC": "BTC-USD",

    "TSLA": "TSLA",
    "GME": "GME",
    "NVDA": "NVDA",
    "PLTR": "PLTR",
    "AMC": "AMC",
    "OPEN": "OPEN",
    "AMD": "AMD",
    "ASTS": "ASTS",
    "Unity": "U",
    "HYMC": "HYMC",
    "BITO": "BITO",
    "RIOT": "RIOT",
    "MARA": "MARA",
    "Ethereum": "ETH-USD",
    "MSTR": "MSTR",
    "MSTU": "MSTU",
    "QQQ": "QQQ",
    "IREN": "IREN",
    "NOK": "NOK",
    "CLSK": "CLSK",
    "Exon": "XOM",
    "OXY": "OXY",
    "SOFI": "SOFI",
}

colA, colB, colC = st.columns([1.2, 0.9, 1.0])
with colA:
    asset_choice = st.selectbox("Asset", list(assets.keys()), index=0)
with colB:
    st.write("")
    refresh_now = st.button("🔄 Refresh now", use_container_width=True)
with colC:
    auto = st.toggle("Auto-refresh", value=True)
    refresh_s = st.selectbox("Refresh rate", [5, 10, 15, 20, 30], index=1)

symbol = assets[asset_choice]
mkt_status = market_status_for(asset_choice)


# ============================================================
# BUILD DATA
# ============================================================
# Always try to keep intraday windows short and fast
df_1m = fetch_intraday(symbol, interval="1m", period="2d")
df_5m = fetch_intraday(symbol, interval="5m", period="5d")
df_15m = fetch_intraday(symbol, interval="15m", period="10d")

# If 1m is missing (common for SPY off-hours), we still build 5m / 15m and use Yahoo price best-effort
# Add indicators
def prep(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in d.columns:
            d[c] = pd.NA
    d["EMA9"] = add_ema(d, 9)
    d["EMA21"] = add_ema(d, 21)
    d["VWAP"] = add_vwap(d) if "Volume" in d.columns else pd.NA
    return d

df_1m = prep(df_1m)
df_5m = prep(df_5m)
df_15m = prep(df_15m)

# ATR on 5m
if df_5m is not None and not df_5m.empty:
    df_5m["ATR"] = atr(df_5m, 14)
else:
    df_5m = pd.DataFrame()


# ============================================================
# PRICE SOURCE (OPEN vs PRE/AFTER)
# ============================================================
price = None
price_source = "Intraday"

# If market open and 1m exists, use latest close
if mkt_status == "MARKET OPEN" and df_1m is not None and not df_1m.empty:
    price = float(df_1m["Close"].iloc[-1])
    price_source = "Intraday (1m)"
elif df_1m is not None and not df_1m.empty:
    price = float(df_1m["Close"].iloc[-1])
    price_source = "Last Candle (1m)"
else:
    # Try Yahoo regularMarketPrice (best-effort)
    p = get_yahoo_regular_market_price(symbol)
    if p is not None:
        price = float(p)
        price_source = "Yahoo Live"
    elif df_5m is not None and not df_5m.empty:
        price = float(df_5m["Close"].iloc[-1])
        price_source = "Last Candle (5m)"
    else:
        price = 0.0
        price_source = "No Data"


# ============================================================
# COMPUTE MAP / BIAS / ACTION / TARGETS
# ============================================================
levels = compute_levels(symbol, df_5m, df_1m)

bias, regime, score = compute_bias_regime(df_5m, df_15m)

action, side, why, invalid, (lvl_name, lvl_val) = classify_action(df_1m, df_5m, levels, bias)

atr5 = float(df_5m["ATR"].iloc[-1]) if (df_5m is not None and not df_5m.empty and pd.notna(df_5m["ATR"].iloc[-1])) else 0.0
t1, t2, t3 = compute_targets(price, bias if bias != "NEUTRAL" else ("BULLISH" if side == "CALLS" else "BEARISH" if side == "PUTS" else "BULLISH"), levels, atr5)


# ============================================================
# COMMAND RIBBON (scrolling)
# ============================================================
def cls_for_bias(b):
    return "good" if b == "BULLISH" else "bad" if b == "BEARISH" else "warn"

def cls_for_action(a):
    if "ENTRY ACTIVE" in a:
        return "good"
    if "HEADS UP" in a:
        return "cyan"
    if "CAUTION" in a:
        return "warn"
    if "EXIT" in a:
        return "bad"
    return "warn"

bias_txt = f"{bias} — {'CALLS' if bias=='BULLISH' else 'PUTS' if bias=='BEARISH' else 'WAIT'}"
lvl_txt = f"{lvl_name}:{lvl_val:.2f}" if (lvl_name and lvl_val) else "LEVEL:—"
inv_txt = f"INVALID:{float(invalid):.2f}" if invalid is not None else "INVALID:—"
rng_txt = f"TARGETS: {t1:.2f} | {t2:.2f} | {t3:.2f}"

ribbon_msg = f"NOW: {action} • BIAS: {bias_txt} • {lvl_txt} • {rng_txt} • {inv_txt} • SCORE:{score}/100 • {mkt_status} • PRICE:{price:.2f} ({price_source}) • "

st.markdown(
    f"""
    <div class="ribbon">
      <div class="marquee">
        <span>{ribbon_msg}{ribbon_msg}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO COMMAND CENTER
# ============================================================
def cls_for_action(a):
    if "ENTRY ACTIVE" in a:
        return "good"
    if "HEADS UP" in a:
        return "cyan"
    if "CAUTION" in a:
        return "warn"
    if "EXIT" in a:
        return "bad"
    return "warn"

price_color = "good" if bias == "BULLISH" else "bad" if bias == "BEARISH" else "warn"
action_color = cls_for_action(action)

status_color = "good" if mkt_status == "MARKET OPEN" else "warn" if mkt_status == "PRE-MARKET" else "cyan" if mkt_status == "24/7" else "warn"

side_line = "CALLS" if side == "CALLS" else "PUTS" if side == "PUTS" else "WAIT"

st.markdown(
    f"""
    <div class="hero">
      <div class="symbol">{asset_choice} • {price_source}</div>
      <div class="price {price_color}">{price:,.2f}</div>
      <div class="action {action_color}">{action}</div>

      <div class="krow">
        <span class="pill {cls_for_bias(bias)}">{bias_txt}</span>
        <span class="pill {status_color}">{mkt_status}</span>
        <span class="pill cyan">REGIME: {regime}</span>
        <span class="pill">SCORE: {score}/100</span>
      </div>

      <div class="hr"></div>

      <div class="targets">
        <div class="label">EXPECTED MOVE (FROM HERE)</div>
        <div class="val {cls_for_bias(bias if bias!='NEUTRAL' else 'BULLISH')}">
          LIKELY {t1:,.2f} &nbsp; | &nbsp; POSS {t2:,.2f} &nbsp; | &nbsp; STRETCH {t3:,.2f}
        </div>
      </div>

      <div class="subline">{'Trade ' + side_line + '.' if side_line!='WAIT' else 'Stand down.'} {inv_txt}</div>
      <div class="whyline">{why}</div>

      <div class="smallnote">Decision-support only. Not financial advice.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FAST READOUT (no clutter)
# ============================================================
st.write("")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Key Level In Play", f"{lvl_name if lvl_name else '—'}", f"{lvl_val:.2f}" if lvl_val else "—")
with c2:
    vwap_now = None
    if df_1m is not None and not df_1m.empty:
        vwap_now = float(df_1m["VWAP"].iloc[-1])
    elif df_5m is not None and not df_5m.empty:
        vwap_now = float(df_5m["VWAP"].iloc[-1])
    st.metric("VWAP", f"{vwap_now:.2f}" if vwap_now else "—", "")
with c3:
    st.metric("ATR(5m)", f"{atr5:.2f}" if atr5 else "—", "")
with c4:
    st.metric("Map", "PDH/PDL/PDC + OR + VWAP", "")

# ============================================================
# MINI CHART (last ~120 mins)
# ============================================================
if df_1m is not None and not df_1m.empty:
    chart_df = df_1m.tail(140).copy()
    chart_df = chart_df.set_index("ts")[["Close", "EMA9", "VWAP"]].dropna()
    st.line_chart(chart_df, height=260, use_container_width=True)
elif df_5m is not None and not df_5m.empty:
    chart_df = df_5m.tail(140).copy()
    chart_df = chart_df.set_index("ts")[["Close", "EMA9", "VWAP"]].dropna()
    st.line_chart(chart_df, height=260, use_container_width=True)
else:
    st.warning("No chart data available right now.")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


# ============================================================
# AUTO REFRESH (simple + reliable)
# ============================================================
# Trigger refresh if button clicked
if refresh_now:
    st.cache_data.clear()
    st.rerun()

# Auto-refresh loop (best-effort)
if auto:
    time.sleep(int(refresh_s))
    st.rerun()