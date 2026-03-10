import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from agent import generate_signal, train_simple_ml, get_data, add_indicators

st.set_page_config(page_title="Lockout Signals • SPY/QQQ", layout="wide")

print("app.py started")

# ---------- Auto refresh ----------
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

if st.session_state.auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='60'>",
        unsafe_allow_html=True
    )

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .card { background: #1e1e2e; border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    .bull { color: #00ff7f; font-weight: bold; }
    .bear { color: #ff4500; font-weight: bold; }
    .neutral { color: #a9a9a9; }
    .header { font-size: 2.2rem; color: #ffd700; text-align: center; margin-bottom: 20px; }
    .ticker-scroll { white-space: nowrap; overflow: hidden; background: #111; padding: 10px; border-radius: 8px; margin-bottom: 20px; font-size: 1.1rem; color: #00ff7f; }
    .signal-box { font-size: 1.6rem; padding: 15px; border-radius: 10px; text-align: center; margin: 15px 0; }
    .glow { animation: glow 2s infinite alternate; }
    @keyframes glow { from {box-shadow: 0 0 10px #00ff7f;} to {box-shadow: 0 0 25px #00ff7f;} }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header">Lockout Signals • SPY / QQQ Command Center</div>',
    unsafe_allow_html=True
)

# ---------- Controls ----------
top1, top2, top3 = st.columns([1, 1, 1])

with top1:
    if st.button("Train ML Model (one-time)"):
        with st.spinner("Training ML..."):
            ok = train_simple_ml()
        if ok:
            st.success("ML trained successfully.")
        else:
            st.warning("ML training did not complete. Check logs and available Yahoo data.")

with top2:
    if st.button("Force Refresh", type="primary"):
        st.rerun()

with top3:
    auto_refresh = st.toggle("Auto refresh every 60s", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

# ---------- Ticker strip ----------
prices = []
for t in ['SPY', 'QQQ', 'ES=F', 'NQ=F', '^VIX']:
    try:
        hist = yf.download(
            t,
            interval='1m',
            period='1d',
            progress=False,
            auto_adjust=False,
            threads=False
        )
        if not hist.empty and 'Close' in hist.columns:
            p = float(hist['Close'].iloc[-1])
            prices.append(f"{t}: ${p:.2f}")
        else:
            prices.append(f"{t}: N/A")
    except Exception as e:
        print(f"Ticker strip error for {t}: {e}")
        prices.append(f"{t}: N/A")

st.markdown(
    f'<div class="ticker-scroll">{" • ".join(prices)}</div>',
    unsafe_allow_html=True
)

# ---------- Main panels ----------
col1, col2 = st.columns(2)

for ticker, col in zip(['SPY', 'QQQ'], [col1, col2]):
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(ticker)

        sig = generate_signal(ticker)

        if 'error' in sig:
            st.error(sig['error'])
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        price_str = f"${sig['price']:.2f}"
        bias_class = (
            "bull" if sig['bias'] == "BULLISH"
            else "bear" if sig['bias'] == "BEARISH"
            else "neutral"
        )

        st.markdown(
            f"<h2>{price_str} <span class='{bias_class}'>{sig['bias']}</span></h2>",
            unsafe_allow_html=True
        )

        st.metric("Momentum Pressure", f"{sig['pressure']}/100", delta=sig['heat'])

        if "ENTRY" in sig['signal']:
            box_color = "#1a3c1a" if sig['bias'] == "BULLISH" else "#3c1a1a"
            st.markdown(
                f'<div class="signal-box glow" style="background:{box_color};">{sig["signal"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(sig['signal'])

        if sig.get('why'):
            st.write("**Why this setup?**")
            for item in sig['why']:
                st.write(f"• {item}")

        st.metric("ML Up Move Prob (next ~30min)", f"{sig['ml_prob_up']}%")

        gauge_min = min(sig['invalid'], sig['stretch'], sig['likely'], sig['possible'], sig['price'])
        gauge_max = max(sig['invalid'], sig['stretch'], sig['likely'], sig['possible'], sig['price'])

        if gauge_min == gauge_max:
            gauge_min -= 1
            gauge_max += 1

        steps = []
        if sig['bias'] == "BULLISH":
            steps = [
                {'range': [gauge_min, sig['likely']], 'color': "red"},
                {'range': [sig['likely'], sig['possible']], 'color': "orange"},
                {'range': [sig['possible'], gauge_max], 'color': "green"}
            ]
        elif sig['bias'] == "BEARISH":
            steps = [
                {'range': [gauge_min, sig['possible']], 'color': "green"},
                {'range': [sig['possible'], sig['likely']], 'color': "orange"},
                {'range': [sig['likely'], gauge_max], 'color': "red"}
            ]
        else:
            steps = [{'range': [gauge_min, gauge_max], 'color': "gray"}]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sig['price'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [gauge_min, gauge_max]},
                'bar': {'color': "gold"},
                'steps': steps
            },
            title={'text': "Invalid → Likely → Possible → Stretch"}
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**Invalid / Stop**: {sig['invalid']:.2f}")
        st.write(f"**Likely**: {sig['likely']:.2f}")
        st.write(f"**Possible**: {sig['possible']:.2f}")
        st.write(f"**Stretch**: {sig['stretch']:.2f}")

        es_val = sig['es']
        nq_val = sig['nq']
        es_text = f"{es_val:.2f}" if isinstance(es_val, (int, float)) else es_val
        nq_text = f"{nq_val:.2f}" if isinstance(nq_val, (int, float)) else nq_val

        st.write(f"**VIX**: {sig['vix']:.2f} | **ES**: {es_text} | **NQ**: {nq_text}")

        if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
            df_chart = get_data(ticker, '5m', '5d')
            if df_chart.empty:
                st.warning("No chart data available.")
            else:
                df_chart = add_indicators(df_chart)
                cols_to_plot = [c for c in ['Close', 'EMA9', 'VWAP'] if c in df_chart.columns]
                if cols_to_plot:
                    st.line_chart(df_chart[cols_to_plot])
                else:
                    st.warning("Expected chart columns were not found.")

        st.markdown('</div>', unsafe_allow_html=True)

st.caption("Optional 60-second auto refresh • Not financial advice • Trade your plan")