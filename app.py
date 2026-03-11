import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals Command Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PAGE STYLE
# =========================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0,255,180,0.08), transparent 22%),
            radial-gradient(circle at 88% 10%, rgba(0,160,255,0.08), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(255,0,120,0.05), transparent 28%),
            linear-gradient(180deg, #020306 0%, #050913 46%, #07101a 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09111d 0%, #0c1525 100%);
    }

    .cc-title {
        font-size: 3rem;
        font-weight: 1000;
        line-height: 1.02;
        letter-spacing: 0.5px;
        color: #ffffff;
        margin: 0 0 14px 0;
        text-shadow:
            0 0 12px rgba(255,255,255,0.05),
            0 0 24px rgba(0,180,255,0.10);
    }

    .top-rail {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }

    .top-pill {
        padding: 9px 14px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(21,30,50,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.18);
        color: #eef4ff;
        font-size: 0.82rem;
        font-weight: 900;
        box-shadow:
            0 10px 20px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

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
        animation: ticker-scroll 20s linear infinite;
        font-size: 0.96rem;
        font-weight: 900;
        letter-spacing: 0.3px;
    }

    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .toolbar-note {
        color: #788caf;
        font-size: 0.84rem;
        margin-top: 6px;
        margin-bottom: 16px;
    }

    .deck {
        border-radius: 28px;
        padding: 18px;
        margin-bottom: 24px;
        background:
            linear-gradient(180deg, rgba(11,18,31,0.98), rgba(7,13,23,0.98));
        border: 1px solid rgba(96,126,194,0.16);
        box-shadow:
            0 18px 38px rgba(0,0,0,0.32),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .hero {
        border-radius: 24px;
        padding: 18px;
        margin-bottom: 16px;
        background:
            radial-gradient(circle at top center, rgba(0,255,170,0.05), transparent 34%),
            linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(98,129,194,0.18);
        box-shadow:
            0 18px 42px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .hero-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }

    .hero-symbol {
        font-size: 1.45rem;
        font-weight: 1000;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .symbol-bull { color: #9effcd; }
    .symbol-bear { color: #ff9db0; }
    .symbol-neutral { color: #ffe89a; }

    .hero-bias {
        padding: 12px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 1000;
        letter-spacing: 1px;
        white-space: nowrap;
        box-shadow:
            0 12px 24px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .bias-bull {
        color: #ddffed;
        background: linear-gradient(135deg, rgba(0,77,46,0.98), rgba(7,116,67,0.88));
        border: 1px solid rgba(55,255,166,0.34);
    }

    .bias-bear {
        color: #ffe4e9;
        background: linear-gradient(135deg, rgba(84,8,22,0.98), rgba(153,15,42,0.88));
        border: 1px solid rgba(255,91,121,0.34);
    }

    .bias-neutral {
        color: #fff3cc;
        background: linear-gradient(135deg, rgba(88,71,7,0.98), rgba(156,122,8,0.88));
        border: 1px solid rgba(255,219,78,0.30);
    }

    .price-box {
        border-radius: 24px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow:
            0 16px 30px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .price-box-bull {
        background:
            radial-gradient(circle at center, rgba(49,240,149,0.12), transparent 60%),
            linear-gradient(180deg, rgba(8,35,24,0.98), rgba(8,22,18,0.98));
        border: 1px solid rgba(49,240,149,0.20);
    }

    .price-box-bear {
        background:
            radial-gradient(circle at center, rgba(255,98,125,0.12), transparent 60%),
            linear-gradient(180deg, rgba(44,10,18,0.98), rgba(24,8,12,0.98));
        border: 1px solid rgba(255,98,125,0.20);
    }

    .price-box-neutral {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,34,8,0.98), rgba(22,18,8,0.98));
        border: 1px solid rgba(255,216,106,0.18);
    }

    .price-row {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }

    .hero-price {
        font-size: 4rem;
        font-weight: 1000;
        line-height: 0.95;
        margin: 0;
    }

    .hero-price-bull {
        color: #31f095;
        text-shadow:
            0 0 14px rgba(49,240,149,0.18),
            0 0 28px rgba(49,240,149,0.08);
    }

    .hero-price-bear {
        color: #ff627d;
        text-shadow:
            0 0 14px rgba(255,98,125,0.18),
            0 0 28px rgba(255,98,125,0.08);
    }

    .hero-price-neutral {
        color: #ffd86a;
        text-shadow:
            0 0 14px rgba(255,216,106,0.16),
            0 0 28px rgba(255,216,106,0.07);
    }

    .daily-change {
        padding: 10px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 1000;
        white-space: nowrap;
        box-shadow:
            0 10px 20px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .change-green {
        color: #dffff0;
        background: linear-gradient(180deg, rgba(0,72,45,0.95), rgba(0,110,70,0.88));
        border: 1px solid rgba(49,240,149,0.24);
    }

    .change-red {
        color: #ffe5ea;
        background: linear-gradient(180deg, rgba(72,12,20,0.95), rgba(120,16,32,0.88));
        border: 1px solid rgba(255,98,125,0.24);
    }

    .change-neutral {
        color: #fff4d5;
        background: linear-gradient(180deg, rgba(70,56,10,0.95), rgba(110,88,12,0.88));
        border: 1px solid rgba(255,216,106,0.22);
    }

    .momentum-meter {
        margin-top: 12px;
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.25);
    }

    .momentum-fill {
        height: 100%;
        border-radius: 999px;
    }

    .momentum-green {
        background: linear-gradient(90deg, #17c964, #31f095);
        box-shadow: 0 0 10px rgba(49,240,149,0.20);
    }

    .momentum-red {
        background: linear-gradient(90deg, #ff627d, #ff8da1);
        box-shadow: 0 0 10px rgba(255,98,125,0.18);
    }

    .momentum-gold {
        background: linear-gradient(90deg, #ffca3a, #ffd86a);
        box-shadow: 0 0 10px rgba(255,216,106,0.14);
    }

    .hero-signal {
        font-size: 1.9rem;
        font-weight: 1000;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .signal-bull { color: #31f095; }
    .signal-bear { color: #ff627d; }
    .signal-neutral { color: #ffd86a; }

    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .pill {
        padding: 9px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 900;
        background: linear-gradient(180deg, rgba(19,29,50,0.96), rgba(11,19,35,0.96));
        border: 1px solid rgba(114,149,220,0.18);
        box-shadow:
            0 10px 20px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pill-green {
        color: #e0ffef;
        border-color: rgba(55,255,166,0.26);
    }

    .pill-red {
        color: #ffe4e9;
        border-color: rgba(255,91,121,0.26);
    }

    .pill-gold {
        color: #fff2c7;
        border-color: rgba(255,219,78,0.24);
    }

    .pill-blue {
        color: #dfefff;
        border-color: rgba(89,173,255,0.24);
    }

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

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .metric-card {
        border-radius: 16px;
        padding: 11px 12px;
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

def tone_for_value(label: str):
    label = label.lower()
    if "calls" in label or "up" in label or "bull" in label:
        return "green"
    if "puts" in label or "down" in label or "invalid" in label or "bear" in label:
        return "red"
    if "warning" in label or "chop" in label or "vwap" in label:
        return "gold"
    return "blue"

def metric_card_html(label: str, value: str, tone: str = "blue"):
    return f"""
    <div class="metric-card tone-{tone}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def hero_price_class(bias):
    if bias == "BULLISH":
        return "hero-price hero-price-bull"
    if bias == "BEARISH":
        return "hero-price hero-price-bear"
    return "hero-price hero-price-neutral"

def hero_signal_class(bias):
    if bias == "BULLISH":
        return "hero-signal signal-bull"
    if bias == "BEARISH":
        return "hero-signal signal-bear"
    return "hero-signal signal-neutral"

def bias_badge_class(bias):
    if bias == "BULLISH":
        return "hero-bias bias-bull"
    if bias == "BEARISH":
        return "hero-bias bias-bear"
    return "hero-bias bias-neutral"

def symbol_class(bias):
    if bias == "BULLISH":
        return "hero-symbol symbol-bull"
    if bias == "BEARISH":
        return "hero-symbol symbol-bear"
    return "hero-symbol symbol-neutral"

def pill_html(text: str, tone: str = "blue"):
    return f'<div class="pill pill-{tone}">{text}</div>'

def feed_tone(text: str):
    t = text.lower()
    if any(x in t for x in ["bull", "calls", "breakout up", "above", "surge"]):
        return "green"
    if any(x in t for x in ["bear", "puts", "below", "breakdown"]):
        return "red"
    if any(x in t for x in ["chop", "warning", "inside active battle zone"]):
        return "gold"
    return "blue"

def get_price_box_class(bias):
    if bias == "BULLISH":
        return "price-box price-box-bull"
    if bias == "BEARISH":
        return "price-box price-box-bear"
    return "price-box price-box-neutral"

def get_change_class(change_pct: float):
    if change_pct > 0:
        return "daily-change change-green"
    if change_pct < 0:
        return "daily-change change-red"
    return "daily-change change-neutral"

def get_momentum_class(bias: str):
    if bias == "BULLISH":
        return "momentum-fill momentum-green"
    if bias == "BEARISH":
        return "momentum-fill momentum-red"
    return "momentum-fill momentum-gold"

def calc_day_change(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None, None

    closes = df["Close"].dropna()
    if len(closes) < 2:
        return None, None

    last_price = safe_float(closes.iloc[-1])
    prev_close = safe_float(closes.iloc[0])

    if prev_close == 0:
        return None, None

    change = last_price - prev_close
    pct = (change / prev_close) * 100
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

st.markdown('<div class="cc-title">Lockout Signals • Command Center</div>', unsafe_allow_html=True)

if "error" not in primary_sig:
    top_pills = [
        pill_html(f"Ticker: {selected_ticker}", "blue"),
        pill_html(f"Timeframe: {interval}", "blue"),
        pill_html(f"Session: {primary_sig['session_status']}", "blue"),
        pill_html(f"State: {primary_sig['market_state']}", "blue"),
        pill_html(f"Conviction: {primary_sig['conviction']}", "gold"),
        pill_html(
            f"Live Price: {primary_sig['price']:.2f}",
            "green" if primary_sig["bias"] == "BULLISH" else "red" if primary_sig["bias"] == "BEARISH" else "gold"
        ),
    ]
    st.markdown(f'<div class="top-rail">{"".join(top_pills)}</div>', unsafe_allow_html=True)

    tape_parts = [
        f'<span style="color:#31f095;">▲ {selected_ticker} {primary_sig["signal"]}</span>',
        f'<span style="color:#8dc8ff;">STATE {primary_sig["market_state"]}</span>',
        f'<span style="color:#31f095;">CALLS&gt;{primary_sig["calls_favored_above"]:.2f}</span>',
        f'<span style="color:#ff627d;">PUTS&lt;{primary_sig["puts_favored_below"]:.2f}</span>',
        f'<span style="color:#ffd86a;">WARNING {primary_sig["warning_line"]:.2f}</span>',
        f'<span style="color:#ff627d;">INVALID {primary_sig["invalidation"]:.2f}</span>',
        f'<span style="color:#31f095;">LIKELY UP {primary_sig["likely_up"]:.2f}</span>',
        f'<span style="color:#ff627d;">LIKELY DOWN {primary_sig["likely_down"]:.2f}</span>',
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

refresh_col, note_col = st.columns([1, 4])
with refresh_col:
    if st.button("Refresh Now"):
        st.rerun()
with note_col:
    st.markdown('<div class="small-note">Opening Range logic becomes strongest after 9:35 AM ET.</div>', unsafe_allow_html=True)

# =========================================================
# SYMBOL DECKS
# =========================================================
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="deck">', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    # get daily % from chart data for hero
    df_day = get_data(ticker, interval=interval)
    day_change, day_change_pct = calc_day_change(df_day)

    if day_change_pct is None:
        day_change_text = "Day N/A"
    else:
        sign = "+" if day_change_pct >= 0 else ""
        day_change_text = f"{sign}{day_change_pct:.2f}%"

    momentum_width = max(8, min(int(sig["pressure"]), 100))

    hero_pills = [
        pill_html(
            f"🎯 {sig['bias']}",
            "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold"
        ),
        pill_html(f"🛰 {sig['market_state']}", "blue"),
        pill_html(f"📊 SCORE {sig['pressure']}/100", "gold"),
        pill_html(f"🔥 {sig['heat']}", "gold"),
    ]

    hero_html = f"""
    <div class="hero">
        <div class="hero-top">
            <div class="{symbol_class(sig['bias'])}">{ticker} • {interval} Tactical Brain</div>
            <div class="{bias_badge_class(sig['bias'])}">{sig['bias']}</div>
        </div>

        <div class="{get_price_box_class(sig['bias'])}">
            <div class="price-row">
                <div class="{hero_price_class(sig['bias'])}">{sig['price']:.2f}</div>
                <div class="{get_change_class(day_change_pct if day_change_pct is not None else 0)}">{day_change_text}</div>
            </div>
            <div class="momentum-meter">
                <div class="{get_momentum_class(sig['bias'])}" style="width:{momentum_width}%;"></div>
            </div>
        </div>

        <div class="{hero_signal_class(sig['bias'])}">{sig['signal']}</div>

        <div class="pill-row">
            {''.join(hero_pills)}
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # TWO-COLUMN MAIN LAYOUT
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-pill">Bias + Structure</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sunken">', unsafe_allow_html=True)

        structure_cards = "".join([
            metric_card_html("Bias", sig["bias"], "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold"),
            metric_card_html("Signal", sig["signal"], "blue"),
            metric_card_html("Market State", sig["market_state"], "blue"),
            metric_card_html("RSI", f"{sig['rsi']:.2f}", "blue"),
            metric_card_html("EMA9", fmt(sig["ema9"]), "blue"),
            metric_card_html("EMA20", fmt(sig["ema20"]) if "ema20" in sig else "N/A", "blue"),
            metric_card_html("VWAP", fmt(sig["vwap"]), "gold"),
            metric_card_html("Session", sig["session_status"], "blue"),
            metric_card_html("Conviction", sig["conviction"], "gold"),
        ])

        st.markdown(f'<div class="metric-grid">{structure_cards}</div>', unsafe_allow_html=True)
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

        battle_cards = "".join([
            metric_card_html("Calls Favored Above", fmt(sig["calls_favored_above"]), "green"),
            metric_card_html("Puts Favored Below", fmt(sig["puts_favored_below"]), "red"),
            metric_card_html("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}", "gold"),
            metric_card_html("Warning Line", fmt(sig["warning_line"]), "gold"),
            metric_card_html("Invalidation", fmt(sig["invalidation"]), "red"),
            metric_card_html("Likely Up", fmt(sig["likely_up"]), "green"),
            metric_card_html("Stretch Up", fmt(sig["stretch_up"]), "green"),
            metric_card_html("Likely Down", fmt(sig["likely_down"]), "red"),
            metric_card_html("Stretch Down", fmt(sig["stretch_down"]), "red"),
        ])

        st.markdown(f'<div class="metric-grid">{battle_cards}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-pill">Session Levels</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sunken">', unsafe_allow_html=True)

        session_cards = "".join([
            metric_card_html("Premarket High", fmt(sig["pm_high"]), "blue"),
            metric_card_html("Premarket Low", fmt(sig["pm_low"]), "blue"),
            metric_card_html("OR High", fmt(sig["or_high"]), "blue"),
            metric_card_html("OR Low", fmt(sig["or_low"]), "blue"),
        ])

        st.markdown(f'<div class="metric-grid">{session_cards}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # COMMENTARY
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