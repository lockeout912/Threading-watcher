# app.py
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Lockout Signals • Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPONSOR_LINK = "https://join.robinhood.com/alisonp311"
SPONSOR_IMAGE_PATH = "robinhood.webp"  # put in same folder as app.py

ASSETS = [
    "SPY", "QQQ",
    "BTC", "Ethereum", "XRP", "XLM", "Solana", "Cardano",
    "TSLA", "GME", "NVDA", "PLTR", "AMC", "OPEN", "AMD", "ASTS", "U", "HYMC",
    "BITO", "RIOT", "MARA", "MSTR", "MSTU", "IREN", "NOK", "CLSK", "XOM", "OXY", "SOFI",
]

TOP_MOVERS_UNIVERSE = sorted(list(set([
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLK", "XLE",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "PLTR", "SOFI", "MARA", "RIOT", "MSTR", "GME", "AMC", "U", "OPEN",
    "COIN", "ROKU", "NIO", "LCID", "RIVN", "SMCI", "ARM",
    "XOM", "CVX", "OXY",
    "BITO",
])))


# =========================
# SYMBOL MAPS
# =========================
def norm_symbol(asset: str) -> str:
    asset = asset.strip()
    crypto_map = {
        "BTC": "BTC-USD",
        "Ethereum": "ETH-USD",
        "XRP": "XRP-USD",
        "XLM": "XLM-USD",
        "Solana": "SOL-USD",
        "Cardano": "ADA-USD",
    }
    return crypto_map.get(asset, asset)


def now_local_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# =========================
# DATA NORMALIZATION (THE FIX)
# =========================
def normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns (e.g., ('High','SPY')).
    This function guarantees a simple single-index OHLCV frame:
    Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # If MultiIndex columns, try to select the symbol level.
    if isinstance(df.columns, pd.MultiIndex):
        # Common shapes:
        # 1) columns level0 = OHLCV, level1 = ticker
        # 2) columns level0 = ticker, level1 = OHLCV
        cols0 = df.columns.get_level_values(0).astype(str)
        cols1 = df.columns.get_level_values(1).astype(str)

        target = str(symbol) if symbol is not None else None

        # Case A: level1 contains ticker
        if target and (target in set(cols1)):
            df = df.xs(target, axis=1, level=1)
        # Case B: level0 contains ticker
        elif target and (target in set(cols0)):
            df = df.xs(target, axis=1, level=0)
        else:
            # Fallback: try to keep OHLCV by dropping one level
            # Prefer level0 names if they look like OHLCV
            maybe_ohlc = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if set(cols0).intersection(maybe_ohlc):
                df.columns = cols0
            else:
                df.columns = cols1

    # Standardize column names
    rename_map = {
        "Adj Close": "AdjClose",
        "adjclose": "AdjClose",
    }
    df.rename(columns=rename_map, inplace=True)

    # Keep only the columns we need (and ensure they exist)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            # Some crypto feeds might miss Volume; synthesize Volume as 1s
            if c == "Volume":
                df["Volume"] = 1.0
            else:
                return pd.DataFrame()

    df = df[needed].dropna()
    return df


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan)
    pv = (tp * vol).cumsum()
    vv = vol.cumsum()
    return (pv / vv).ffill()


def chop_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # CHOP = 100 * log10(sum(TR,n)/(maxH-minL)) / log10(n)
    high = df["High"].rolling(n).max()
    low = df["Low"].rolling(n).min()
    tr = atr(df, 1)
    sum_tr = tr.rolling(n).sum()
    denom = (high - low).replace(0, np.nan)
    chop = 100 * (np.log10(sum_tr / denom) / np.log10(n))
    return chop.clip(0, 100).ffill()


# =========================
# FETCH
# =========================
@st.cache_data(ttl=8, show_spinner=False)
def fetch_intraday(symbol: str, interval: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    period = "7d" if interval == "1m" else "5d"
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=True,
            threads=True,
        )
        df = normalize_ohlcv(df, symbol)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_daily(symbols: list[str]) -> pd.DataFrame:
    if yf is None or not symbols:
        return pd.DataFrame()

    try:
        df = yf.download(
            symbols,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True
        )
        return df
    except Exception:
        return pd.DataFrame()


# =========================
# ENGINE
# =========================
def decide_engine(df5: pd.DataFrame, df1: pd.DataFrame, mode: str):
    out = {
        "price": np.nan,
        "last_time": "",
        "bias": "NEUTRAL",
        "direction": "WAIT",
        "regime": "RANGE",
        "action": "HEADS UP",
        "score": 0,
        "vwap_5m": np.nan,
        "atr_5m": np.nan,
        "chop": np.nan,
        "likely": np.nan,
        "poss": np.nan,
        "stretch": np.nan,
        "invalid": np.nan,
        "why": "No clear alignment.",
        "one_liner": "Stand down. No edge.",
        "status": "AFTER HOURS",
    }

    if df5 is None or df5.empty:
        out["why"] = "No 5m data."
        out["action"] = "WAIT — NO DATA"
        return out

    # Use 1m close if available for “current-ish” price
    if df1 is not None and not df1.empty:
        price = float(df1["Close"].iloc[-1])
        last_time = df1.index[-1]
    else:
        price = float(df5["Close"].iloc[-1])
        last_time = df5.index[-1]

    out["price"] = price
    out["last_time"] = str(last_time)

    # Status from candle freshness (simple + robust)
    try:
        now = datetime.now(timezone.utc)
        lt = last_time.to_pydatetime() if hasattr(last_time, "to_pydatetime") else datetime.now(timezone.utc)
        if lt.tzinfo is None:
            lt = lt.replace(tzinfo=timezone.utc)
        age_sec = max(0, (now - lt.astimezone(timezone.utc)).total_seconds())
        out["status"] = "MARKET OPEN" if age_sec <= 360 else "AFTER HOURS"
    except Exception:
        out["status"] = "AFTER HOURS"

    df5 = df5.copy()
    df5["EMA9"] = ema(df5["Close"], 9)
    df5["EMA21"] = ema(df5["Close"], 21)
    df5["VWAP"] = session_vwap(df5)
    df5["ATR"] = atr(df5, 14)
    df5["CHOP"] = chop_index(df5, 14)

    c5 = float(df5["Close"].iloc[-1])
    ema9_5 = float(df5["EMA9"].iloc[-1])
    ema21_5 = float(df5["EMA21"].iloc[-1])
    vwap5 = float(df5["VWAP"].iloc[-1])
    atr5 = float(df5["ATR"].iloc[-1]) if not np.isnan(df5["ATR"].iloc[-1]) else float(df5["Close"].rolling(14).std().iloc[-1] or 0.0)
    chop = float(df5["CHOP"].iloc[-1]) if not np.isnan(df5["CHOP"].iloc[-1]) else 50.0

    out["vwap_5m"] = vwap5
    out["atr_5m"] = atr5
    out["chop"] = chop

    bullish_stack = (c5 > vwap5) and (ema9_5 > ema21_5)
    bearish_stack = (c5 < vwap5) and (ema9_5 < ema21_5)

    if bullish_stack:
        out["bias"] = "BULLISH"
        out["direction"] = "CALLS"
    elif bearish_stack:
        out["bias"] = "BEARISH"
        out["direction"] = "PUTS"
    else:
        out["bias"] = "NEUTRAL"
        out["direction"] = "WAIT"

    slope = float((df5["EMA9"].iloc[-1] - df5["EMA9"].iloc[-4]) / max(1e-9, atr5))
    if chop >= 55:
        out["regime"] = "RANGE"
    else:
        out["regime"] = "TREND" if abs(slope) > 0.15 else "RANGE"

    trigger_ok = False
    momentum_ok = False

    if df1 is not None and not df1.empty and len(df1) >= 20:
        df1 = df1.copy()
        df1["EMA9"] = ema(df1["Close"], 9)
        df1["EMA21"] = ema(df1["Close"], 21)
        c1 = float(df1["Close"].iloc[-1])
        e9 = float(df1["EMA9"].iloc[-1])
        e21 = float(df1["EMA21"].iloc[-1])

        if out["bias"] == "BULLISH":
            trigger_ok = (c1 > e9) or (e9 > e21)
            momentum_ok = (e9 > e21)
        elif out["bias"] == "BEARISH":
            trigger_ok = (c1 < e9) or (e9 < e21)
            momentum_ok = (e9 < e21)

    score = 0
    score += 30 if out["bias"] != "NEUTRAL" else 10
    score += 25 if out["regime"] == "TREND" else 10
    score += 20 if trigger_ok else 0
    score += 15 if momentum_ok else 0
    score += 10 if chop < 55 else 0
    score = int(max(0, min(100, score)))
    out["score"] = score

    if out["bias"] == "NEUTRAL":
        out["action"] = "HEADS UP"
        out["one_liner"] = "Stand down. No edge."
        out["why"] = "No clear alignment."
    else:
        if trigger_ok and score >= (70 if mode == "FULL SEND" else 78):
            out["action"] = f"ENTRY ACTIVE — {out['direction']}"
            out["one_liner"] = "Alignment + trigger confirmed."
            out["why"] = "Trend + VWAP alignment, trigger confirmed."
        elif score >= (55 if mode == "FULL SEND" else 62):
            out["action"] = f"CAUTION — {out['direction']}"
            out["one_liner"] = "Bias is set, but trigger is shaky."
            out["why"] = "Bias is set; momentum/trigger not fully confirmed."
        else:
            out["action"] = "WAIT — NO EDGE"
            out["one_liner"] = "Stand down. No edge."
            out["why"] = "Bias exists but conditions are weak or choppy."

    # Expected move anchored to current price
    if atr5 <= 0 or np.isnan(atr5):
        atr5 = max(0.15, price * 0.001)

    if out["bias"] == "BULLISH":
        sign = +1
        invalid = price - (0.9 * atr5)
    elif out["bias"] == "BEARISH":
        sign = -1
        invalid = price + (0.9 * atr5)
    else:
        sign = 0
        invalid = price - (0.9 * atr5)

    base = 0.75 if out["regime"] == "RANGE" else 1.05
    if mode == "FULL SEND":
        base *= 1.15

    out["likely"] = float(price + sign * (base * 0.9 * atr5))
    out["poss"] = float(price + sign * (base * 1.35 * atr5))
    out["stretch"] = float(price + sign * (base * 2.0 * atr5))
    out["invalid"] = float(invalid)

    return out


def fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def color_for_action(action: str) -> str:
    a = (action or "").upper()
    if "ENTRY ACTIVE" in a:
        return "#19ff8a"
    if "CAUTION" in a:
        return "#ffb020"
    if "HEADS UP" in a:
        return "#4dd8ff"
    if "WAIT" in a:
        return "#ffdf6e"
    return "#ffffff"


def badge(label: str, color: str, border: str = None) -> str:
    border = border or color
    return f"""
    <span class="pill" style="border:1px solid {border}; color:{color};">
        {label}
    </span>
    """


def inject_theme_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            background: #0b0f14 !important;
            color: #e7edf5 !important;
        }
        section[data-testid="stSidebar"] {
            background: #0a0e13 !important;
        }
        .pill {
            display:inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
            letter-spacing: .4px;
            margin: 4px 6px 0 0;
            background: rgba(255,255,255,0.02);
            backdrop-filter: blur(8px);
        }
        .command-card {
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 22px 18px;
            background: radial-gradient(1200px 500px at 50% 0%, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            box-shadow: 0 12px 40px rgba(0,0,0,0.55);
        }
        .asset-title {
            font-size: clamp(18px, 2.2vw, 28px);
            opacity: .9;
            font-weight: 800;
            letter-spacing: .6px;
            text-align: center;
            margin-bottom: 10px;
        }
        .price {
            font-size: clamp(60px, 8vw, 92px);
            font-weight: 900;
            letter-spacing: 1px;
            text-align: center;
            margin: 4px 0 2px 0;
        }
        .action {
            font-size: clamp(28px, 4.6vw, 56px);
            font-weight: 900;
            letter-spacing: 1px;
            text-align: center;
            margin: 2px 0 10px 0;
        }
        .expected-title {
            text-align:center;
            margin-top: 14px;
            opacity: .75;
            font-weight: 800;
            letter-spacing: .8px;
        }
        .expected-line {
            text-align:center;
            font-size: clamp(16px, 2.2vw, 22px);
            font-weight: 900;
            letter-spacing: .8px;
            margin-top: 6px;
        }
        .subline {
            text-align:center;
            margin-top: 10px;
            font-size: 16px;
            opacity: .85;
        }
        .micro {
            text-align:center;
            margin-top: 4px;
            font-size: 14px;
            opacity: .65;
        }
        .marquee-wrap {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 8px 0;
            background: rgba(0,0,0,0.25);
            margin: 6px 0 14px 0;
        }
        .marquee {
            display: inline-block;
            white-space: nowrap;
            will-change: transform;
            animation: marquee 16s linear infinite;
            font-weight: 900;
            letter-spacing: .8px;
            font-size: 14px;
            padding-left: 100%;
        }
        @keyframes marquee {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        button[data-baseweb="tab"] {
            font-weight: 900 !important;
            letter-spacing: .4px !important;
            color: #e7edf5 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(25,255,138,0.10) !important;
            border-bottom: 2px solid #19ff8a !important;
        }
        .footer {
            opacity: .55;
            font-size: 12px;
            margin-top: 10px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def inject_autorefresh(seconds: int):
    st.components.v1.html(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {int(seconds)*1000});
        </script>
        """,
        height=0,
        scrolling=False,
    )


def build_top_movers():
    df = fetch_daily([s for s in TOP_MOVERS_UNIVERSE if s])
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # yfinance multi-symbol daily: MultiIndex columns (field, ticker)
    try:
        close = df["Close"]
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame()

    if len(close) < 2:
        return pd.DataFrame(), pd.DataFrame()

    prev = close.iloc[-2]
    last = close.iloc[-1]
    pct = ((last - prev) / prev.replace(0, np.nan)) * 100.0

    movers = pd.DataFrame({
        "Ticker": pct.index.astype(str),
        "%": pct.values,
        "Last": last.values
    }).dropna()

    movers["%"] = movers["%"].astype(float)
    movers["Last"] = movers["Last"].astype(float)

    top_up = movers.sort_values("%", ascending=False).head(10).reset_index(drop=True)
    top_dn = movers.sort_values("%", ascending=True).head(10).reset_index(drop=True)
    return top_up, top_dn


# =========================
# UI
# =========================
inject_theme_css()
st.title("Lockout Signals • Command Center")

with st.sidebar:
    st.markdown("### Controls")
    asset = st.selectbox("Asset", ASSETS, index=0)
    mode = st.radio("Mode", ["AGGRESSIVE", "FULL SEND"], index=0)

    auto_on = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.selectbox("Refresh seconds", [5, 10, 15, 20, 30, 60], index=1)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 Refresh now", use_container_width=True):
            st.rerun()
    with c2:
        st.caption("")

    st.markdown("---")
    st.markdown("### Sponsor")
    try:
        st.image(SPONSOR_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.caption("(Add `robinhood.webp` to your repo to show the image.)")
    st.markdown(f"**Sign up for Robinhood** 🎁  \n{SPONSOR_LINK}")

    st.markdown("---")
    st.markdown("### Top Movers (Universe)")
    top_up, top_dn = build_top_movers()
    if top_up.empty:
        st.caption("Top movers unavailable (data feed may be rate-limited).")
    else:
        st.markdown("**Top 10 Up**")
        st.dataframe(
            top_up.style.format({"%": "{:+.2f}", "Last": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("**Top 10 Down**")
        st.dataframe(
            top_dn.style.format({"%": "{:+.2f}", "Last": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

if auto_on:
    inject_autorefresh(refresh_seconds)

symbol = norm_symbol(asset)
df5 = fetch_intraday(symbol, "5m")
df1 = fetch_intraday(symbol, "1m")

engine = decide_engine(df5, df1, mode)
action_color = color_for_action(engine["action"])

feed = (
    f"PRICE: {fmt(engine['price'])} • ACTION: {engine['action']} • "
    f"BIAS: {engine['bias']} — {engine['direction']} • "
    f"STATUS: {engine['status']} • REGIME: {engine['regime']} • "
    f"SCORE: {engine['score']}/100 • INVALID: {fmt(engine['invalid'])} • "
    f"LIKELY: {fmt(engine['likely'])} • POSS: {fmt(engine['poss'])} • STRETCH: {fmt(engine['stretch'])} • "
    f"LAST: {engine['last_time']} • "
)

st.markdown(
    f"""
    <div class="marquee-wrap">
        <div class="marquee" style="color:{action_color};">
            {feed} {feed} {feed}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

bias_chip_color = "#19ff8a" if engine["bias"] == "BULLISH" else ("#ff5a6e" if engine["bias"] == "BEARISH" else "#ffdf6e")
status_chip_color = "#19ff8a" if engine["status"] == "MARKET OPEN" else "#ffb020"
regime_chip_color = "#4dd8ff" if engine["regime"] == "RANGE" else "#b993ff"

st.markdown(
    f"""
    <div class="command-card">
        <div class="asset-title">{asset} • 5m Brain / 1m Trigger</div>
        <div class="price" style="color:{action_color};">{fmt(engine['price'])}</div>
        <div class="action" style="color:{action_color};">{engine['action']}</div>

        <div style="text-align:center;">
            {badge(f"{engine['bias']} — {engine['direction']}", bias_chip_color)}
            {badge(engine["status"], status_chip_color)}
            {badge(f"REGIME: {engine['regime']}", regime_chip_color)}
            {badge(f"SCORE: {engine['score']}/100", "#ffffff", "rgba(255,255,255,0.25)")}
            {badge(f"MODE: {mode}", "#ffffff", "rgba(255,255,255,0.25)")}
        </div>

        <div class="expected-title">EXPECTED MOVE (FROM HERE)</div>
        <div class="expected-line" style="color:{action_color};">
            LIKELY {fmt(engine['likely'])} &nbsp; | &nbsp; POSS {fmt(engine['poss'])} &nbsp; | &nbsp; STRETCH {fmt(engine['stretch'])}
        </div>

        <div class="subline">
            {engine["one_liner"]} <span style="opacity:.7;">Invalid:</span> <b>{fmt(engine["invalid"])}</b>
        </div>
        <div class="micro">{engine["why"]}</div>

        <div class="footer">
            Decision-support only. Not financial advice. Data timestamp: {engine["last_time"]} • {now_local_str()}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Bias", engine["bias"])
with k2:
    st.metric("VWAP", fmt(engine["vwap_5m"]))
with k3:
    st.metric("ATR(5m)", fmt(engine["atr_5m"]))
with k4:
    st.metric("Chop", f"{int(round(engine['chop']))}/100")