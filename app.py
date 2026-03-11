import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals Command Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0,255,180,0.07), transparent 22%),
            radial-gradient(circle at 88% 10%, rgba(0,160,255,0.07), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(255,0,120,0.04), transparent 28%),
            linear-gradient(180deg, #020306 0%, #050913 46%, #07101a 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09111d 0%, #0c1525 100%);
    }

    .title-main {
        font-size: 2.75rem;
        font-weight: 1000;
        line-height: 1.03;
        color: #ffffff;
        margin-bottom: 0.7rem;
        text-shadow: 0 0 24px rgba(0,180,255,0.08);
    }

    .pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 900;
        margin: 0 8px 8px 0;
        background: linear-gradient(180deg, rgba(20,29,49,0.96), rgba(11,18,31,0.96));
        border: 1px solid rgba(114,149,220,0.18);
        box-shadow:
            0 8px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pill-blue { color: #dfefff; }
    .pill-green { color: #dfffee; }
    .pill-red { color: #ffe4e9; }
    .pill-gold { color: #fff2c7; }

    .ticker-shell {
        border-radius: 18px;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(11,19,33,0.98), rgba(8,15,27,0.98));
        border: 1px solid rgba(96,126,194,0.18);
        box-shadow:
            0 14px 28px rgba(0,0,0,0.26),
            inset 0 1px 0 rgba(255,255,255,0.02);
        margin-bottom: 18px;
    }

    .ticker-track {
        white-space: nowrap;
        overflow: hidden;
        padding: 12px 0;
    }

    .ticker-text {
        display: inline-block;
        padding-left: 100%;
        animation: ticker-scroll 22s linear infinite;
        font-size: 0.96rem;
        font-weight: 900;
        letter-spacing: 0.3px;
    }

    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .deck {
        border-radius: 26px;
        padding: 18px;
        margin-bottom: 24px;
        background:
            linear-gradient(180deg, rgba(11,18,31,0.98), rgba(7,13,23,0.98));
        border: 1px solid rgba(96,126,194,0.16);
        box-shadow:
            0 18px 38px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .hero-wrap {
        border-radius: 24px;
        padding: 18px;
        margin-bottom: 16px;
        background:
            radial-gradient(circle at top center, rgba(0,255,170,0.05), transparent 34%),
            linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(98,129,194,0.18);
        box-shadow:
            0 18px 42px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .hero-symbol {
        font-size: 1.45rem;
        font-weight: 1000;
        letter-spacing: 1.4px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .symbol-bull { color: #9effcd; }
    .symbol-bear { color: #ff9db0; }
    .symbol-neutral { color: #ffe89a; }

    .hero-price-box {
        border-radius: 24px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow:
            0 16px 30px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .price-bull {
        background:
            radial-gradient(circle at center, rgba(49,240,149,0.10), transparent 60%),
            linear-gradient(180deg, rgba(8,35,24,0.98), rgba(8,22,18,0.98));
        border: 1px solid rgba(49,240,149,0.18);
    }

    .price-bear {
        background:
            radial-gradient(circle at center, rgba(255,98,125,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,10,18,0.98), rgba(24,8,12,0.98));
        border: 1px solid rgba(255,98,125,0.18);
    }

    .price-neutral {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,34,8,0.98), rgba(22,18,8,0.98));
        border: 1px solid rgba(255,216,106,0.16);
    }

    .hero-price {
        font-size: 3.8rem;
        font-weight: 1000;
        line-height: 0.95;
        margin: 0;
    }

    .hero-price-bull {
        color: #31f095;
        text-shadow: 0 0 24px rgba(49,240,149,0.12);
    }

    .hero-price-bear {
        color: #ff627d;
        text-shadow: 0 0 24px rgba(255,98,125,0.12);
    }

    .hero-price-neutral {
        color: #ffd86a;
        text-shadow: 0 0 24px rgba(255,216,106,0.10);
    }

    .daily-pill {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 1000;
    }

    .daily-green {
        color: #dffff0;
        background: linear-gradient(180deg, rgba(0,72,45,0.95), rgba(0,110,70,0.88));
        border: 1px solid rgba(49,240,149,0.22);
    }

    .daily-red {
        color: #ffe5ea;
        background: linear-gradient(180deg, rgba(72,12,20,0.95), rgba(120,16,32,0.88));
        border: 1px solid rgba(255,98,125,0.22);
    }

    .daily-neutral {
        color: #fff4d5;
        background: linear-gradient(180deg, rgba(70,56,10,0.95), rgba(110,88,12,0.88));
        border: 1px solid rgba(255,216,106,0.20);
    }

    .hero-signal {
        font-size: 1.75rem;
        font-weight: 1000;
        line-height: 1.12;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .signal-bull { color: #31f095; }
    .signal-bear { color: #ff627d; }
    .signal-neutral { color: #ffd86a; }

    .section-pill {
        display: inline-block;
        margin-bottom: 10px;
        padding: 7px 14px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(17,27,46,0.95), rgba(10,18,31,0.95));
        border: 1px solid rgba(105,140,212,0.18);
        color: #9fc0ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 1000;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .panel-sunken {
        border-radius: 22px;
        padding: 12px;
        background: linear-gradient(180deg, rgba(5,9,16,0.98), rgba(8,14,24,0.98));
        border: 1px solid rgba(64, 89, 142, 0.16);
        box-shadow:
            inset 0 12px 20px rgba(0,0,0,0.34),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .metric-card {
        border-radius: 16px;
        padding: 11px 12px;
        margin-bottom: 8px;
        background: linear-gradient(180deg, rgba(18,27,45,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(110,145,220,0.12);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .metric-label {
        color: #8da6d3;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        font-weight: 900;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.02rem;
        font-weight: 1000;
    }

    .tone-green .metric-value { color: #31f095; }
    .tone-red .metric-value { color: #ff627d; }
    .tone-gold .metric-value { color: #ffd86a; }
    .tone-blue .metric-value { color: #8dc8ff; }

    .feed-wrap {
        border-radius: 18px;
        padding: 12px;
        background: linear-gradient(180deg, rgba(8,14,24,0.98), rgba(6,10,17,0.98));
        border: 1px solid rgba(75,101,160,0.16);
        box-shadow:
            inset 0 10px 18px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .feed-item {
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        font-weight: 900;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .feed-green {
        background: linear-gradient(180deg, rgba(0,64,40,0.94), rgba(0,93,60,0.88));
        border: 1px solid rgba(55,255,166,0.20);
        color: #e1fff0;
    }

    .feed-red {
        background: linear-gradient(180deg, rgba(70,10,18,0.94), rgba(110,16,30,0.88));
        border: 1px solid rgba(255,91,121,0.20);
        color: #ffe7ec;
    }

    .feed-gold {
        background: linear-gradient(180deg, rgba(76,60,10,0.94), rgba(112,88,12,0.88));
        border: 1px solid rgba(255,219,78,0.18);
        color: #fff4ce;
    }

    .feed-blue {
        background: linear-gradient(180deg, rgba(16,30,56,0.94), rgba(11,22,40,0.88));
        border: 1px solid rgba(89,173,255,0.18);
        color: #e6f2ff;
    }

    .commentary-box {
        background: linear-gradient(180deg, rgba(14,24,40,0.98), rgba(9,15,26,0.98));
        border-left: 5px solid #43c2ff;
        border-radius: 16px;
        padding: 14px;
        color: #e1ecff;
        line-height: 1.55;
        box-shadow:
            0 12px 24px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .small-note {
        color: #748aad;
        font-size: 0.84rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 16px;
        border: 1px solid rgba(118,157,232,0.20);
        background: linear-gradient(180deg, #15213b 0%, #0d1728 100%);
        color: #f7fbff;
        font-weight: 1000;
        padding: 0.60rem 1rem;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(148,191,255,0.48);
        color: white;
    }

    div[data-baseweb="select"] > div {
        background: rgba(14,22,38,0.96);
        border-radius: 14px;
        border: 1px solid rgba(101,139,215,0.18);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def fmt(x):
    return f"{x:.2f}" if x is not None else "N/A"

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def pill(text: str, tone: str = "blue") -> str:
    return f'<span class="pill pill-{tone}">{text}</span>'

def metric_card(label: str, value: str, tone: str = "blue"):
    st.markdown(
        f"""
        <div class="metric-card tone-{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def feed_tone(text: str):
    t = text.lower()
    if any(x in t for x in ["bull", "calls", "above", "surge"]):
        return "green"
    if any(x in t for x in ["bear", "puts", "below", "breakdown"]):
        return "red"
    if any(x in t for x in ["chop", "warning", "battle zone"]):
        return "gold"
    return "blue"

def symbol_class(bias):
    if bias == "BULLISH":
        return "hero-symbol symbol-bull"
    if bias == "BEARISH":
        return "hero-symbol symbol-bear"
    return "hero-symbol symbol-neutral"

def price_class(bias):
    if bias == "BULLISH":
        return "hero-price hero-price-bull"
    if bias == "BEARISH":
        return "hero-price hero-price-bear"
    return "hero-price hero-price-neutral"

def signal_class(bias):
    if bias == "BULLISH":
        return "hero-signal signal-bull"
    if bias == "BEARISH":
        return "hero-signal signal-bear"
    return "hero-signal signal-neutral"

def bias_badge_html(bias: str):
    if bias == "BULLISH":
        cls = "hero-bias bias-bull"
    elif bias == "BEARISH":
        cls = "hero-bias bias-bear"
    else:
        cls = "hero-bias bias-neutral"
    return f'<div class="{cls}">{bias}</div>'

def price_box_class(bias):
    if bias == "BULLISH":
        return "hero-price-box price-bull"
    if bias == "BEARISH":
        return "hero-price-box price-bear"
    return "hero-price-box price-neutral"

def daily_class(pct):
    if pct is None:
        return "daily-pill daily-neutral"
    if pct > 0:
        return "daily-pill daily-green"
    if pct < 0:
        return "daily-pill daily-red"
    return "daily-pill daily-neutral"

def calc_day_change(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None, None

    closes = df["Close"].dropna()
    if len(closes) < 2:
        return None, None

    last_price = safe_float(closes.iloc[-1])
    first_price = safe_float(closes.iloc[0])

    if first_price == 0:
        return None, None

    change = last_price - first_price
    pct = (change / first_price) * 100
    return change, pct

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Command Controls")
    selected_ticker = st.selectbox("Primary Ticker", ["SPY", "QQQ", "IWM", "DIA", "AAPL", "NVDA", "TSLA"], index=0)
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m"], index=1)
    show_secondary = st.toggle("Show Secondary Ticker", value=True)
    secondary_ticker = st.selectbox("Secondary Ticker", ["QQQ", "SPY", "IWM", "DIA", "AAPL", "NVDA", "TSLA"], index=0)
    auto_refresh = st.toggle("Auto Refresh 30s", value=False)

if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

tickers = [selected_ticker]
if show_secondary and secondary_ticker != selected_ticker:
    tickers.append(secondary_ticker)

# =========================================================
# PAGE HEADER
# =========================================================
primary_sig = generate_signal(selected_ticker, interval=interval)

st.markdown('<div class="title-main">Lockout Signals • Command Center</div>', unsafe_allow_html=True)

if "error" not in primary_sig:
    rail = "".join([
        pill(f"Ticker: {selected_ticker}", "blue"),
        pill(f"Timeframe: {interval}", "blue"),
        pill(f"Session: {primary_sig['session_status']}", "blue"),
        pill(f"State: {primary_sig['market_state']}", "blue"),
        pill(f"Conviction: {primary_sig['conviction']}", "gold"),
        pill(f"Live Price: {primary_sig['price']:.2f}", "green" if primary_sig["bias"] == "BULLISH" else "red" if primary_sig["bias"] == "BEARISH" else "gold"),
    ])
    st.markdown(f'<div>{rail}</div>', unsafe_allow_html=True)

    tape_parts = [
        f'<span style="color:#31f095;">▲ {selected_ticker} {primary_sig["signal"]}</span>',
        f'<span style="color:#8dc8ff;">STATE {primary_sig["market_state"]}</span>',
        f'<span style="color:#31f095;">CALLS&gt;{primary_sig["calls_favored_above"]:.2f}</span>',
        f'<span style="color:#ff627d;">PUTS&lt;{primary_sig["puts_favored_below"]:.2f}</span>',
        f'<span style="color:#ffd86a;">WARNING {primary_sig["warning_line"]:.2f}</span>',
        f'<span style="color:#ff627d;">INVALID {primary_sig["invalidation"]:.2f}</span>',
    ]
    tape_text = " &nbsp;&nbsp;•&nbsp;&nbsp; ".join(tape_parts)

    st.markdown(
        f"""
        <div class="ticker-shell">
            <div class="ticker-track">
                <div class="ticker-text">{tape_text} &nbsp;&nbsp;•&nbsp;&nbsp; {tape_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("Refresh Now"):
        st.rerun()
with c2:
    st.markdown('<div class="small-note">Opening Range logic becomes strongest after 9:35 AM ET.</div>', unsafe_allow_html=True)

# =========================================================
# DECKS
# =========================================================
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="deck">', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    df_day = get_data(ticker, interval=interval)
    _, day_change_pct = calc_day_change(df_day)

    if day_change_pct is None:
        day_text = "Day N/A"
    else:
        sign = "+" if day_change_pct >= 0 else ""
        day_text = f"{sign}{day_change_pct:.2f}%"

    # HERO
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(
            f'<div class="{symbol_class(sig["bias"])}">{ticker} • {interval} Tactical Brain</div>',
            unsafe_allow_html=True
        )
    with hc2:
        st.markdown(bias_badge_html(sig["bias"]), unsafe_allow_html=True)

    st.markdown(f'<div class="{price_box_class(sig["bias"])}">', unsafe_allow_html=True)
    pc1, pc2 = st.columns([3, 1])
    with pc1:
        st.markdown(
            f'<div class="{price_class(sig["bias"])}">{sig["price"]:.2f}</div>',
            unsafe_allow_html=True
        )
    with pc2:
        st.markdown(
            f'<div class="{daily_class(day_change_pct)}">{day_text}</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="{signal_class(sig["bias"])}">{sig["signal"]}</div>',
        unsafe_allow_html=True
    )

    hero_pills = "".join([
        pill(f"🎯 {sig['bias']}", "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold"),
        pill(f"🛰 {sig['market_state']}", "blue"),
        pill(f"📊 SCORE {sig['pressure']}/100", "gold"),
        pill(f"🔥 {sig['heat']}", "gold"),
    ])
    st.markdown(f'<div>{hero_pills}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # MAIN TWO-COLUMN LAYOUT
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-pill">Bias + Structure</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sunken">', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            metric_card("Bias", sig["bias"], "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold")
        with r1c2:
            metric_card("Signal", sig["signal"], "blue")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            metric_card("Market State", sig["market_state"], "blue")
        with r2c2:
            metric_card("RSI", f"{sig['rsi']:.2f}", "blue")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            metric_card("EMA9", fmt(sig["ema9"]), "blue")
        with r3c2:
            metric_card("EMA20", fmt(sig["ema20"]) if "ema20" in sig else "N/A", "blue")

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            metric_card("VWAP", fmt(sig["vwap"]), "gold")
        with r4c2:
            metric_card("Conviction", sig["conviction"], "gold")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-pill">Rolling Feed</div>', unsafe_allow_html=True)
        st.markdown('<div class="feed-wrap">', unsafe_allow_html=True)
        for item in sig.get("feed", []):
            tone = feed_tone(item)
            st.markdown(f'<div class="feed-item feed-{tone}">{item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-pill">Battle Zones + Targets</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sunken">', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            metric_card("Calls Favored Above", fmt(sig["calls_favored_above"]), "green")
        with r1c2:
            metric_card("Puts Favored Below", fmt(sig["puts_favored_below"]), "red")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            metric_card("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}", "gold")
        with r2c2:
            metric_card("Warning Line", fmt(sig["warning_line"]), "gold")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            metric_card("Invalidation", fmt(sig["invalidation"]), "red")
        with r3c2:
            metric_card("Session", sig["session_status"], "blue")

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            metric_card("Likely Up", fmt(sig["likely_up"]), "green")
        with r4c2:
            metric_card("Stretch Up", fmt(sig["stretch_up"]), "green")

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            metric_card("Likely Down", fmt(sig["likely_down"]), "red")
        with r5c2:
            metric_card("Stretch Down", fmt(sig["stretch_down"]), "red")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-pill">Session Levels</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sunken">', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("PM High", fmt(sig["pm_high"]), "blue")
        with s2:
            metric_card("PM Low", fmt(sig["pm_low"]), "blue")
        with s3:
            metric_card("OR High", fmt(sig["or_high"]), "blue")
        with s4:
            metric_card("OR Low", fmt(sig["or_low"]), "blue")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-pill">Commentary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="commentary-box">{sig["commentary"]}</div>', unsafe_allow_html=True)

    with st.expander(f"Open {ticker} Chart Deck"):
        df = get_data(ticker, interval=interval)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df.columns]
            st.line_chart(df[cols], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)