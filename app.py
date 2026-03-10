import streamlit as st
import time
import plotly.graph_objects as go
from agent import generate_signal, train_simple_ml, get_data

st.set_page_config(page_title="Lockout Signals • SPY/QQQ", layout="wide")

print("app.py started")

# ML button (not auto-run)
if st.button("Train ML Model (one-time)"):
    with st.spinner("Training ML..."):
        train_simple_ml()
    st.success("ML trained (check logs if needed)")

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

st.markdown('<div class="header">Lockout Signals • SPY / QQQ Command Center</div>', unsafe_allow_html=True)

prices = []
for t in ['SPY', 'QQQ', 'ES=F', 'NQ=F', '^VIX']:
    try:
        p = yf.Ticker(t).info.get('regularMarketPrice', 'N/A')
        prices.append(f"{t}: {p if p=='N/A' else f'${p:.2f}'}")
    except:
        pass
st.markdown(f'<div class="ticker-scroll">{" • ".join(prices)}</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

for ticker, col in zip(['SPY', 'QQQ'], [col1, col2]):
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(ticker)

        sig = generate_signal(ticker)

        if 'error' in sig:
            st.error(sig['error'])
            continue

        price_str = f"${sig['price']:.2f}"
        bias_class = "bull" if sig['bias'] == "BULLISH" else "bear" if sig['bias'] == "BEARISH" else "neutral"
        st.markdown(f"<h2>{price_str} <span class='{bias_class}'>{sig['bias']}</span></h2>", unsafe_allow_html=True)

        st.metric("Momentum Pressure", f"{sig['pressure']}/100", delta=sig['heat'])

        if "ENTRY" in sig['signal']:
            st.markdown(f'<div class="signal-box glow" style="background:#1a3c1a;">{sig["signal"]}</div>', unsafe_allow_html=True)
        else:
            st.info(sig['signal'])

        if sig['why']:
            st.write("**Why this setup?**")
            for item in sig['why']:
                st.write(f"• {item}")

        st.metric("ML Up Move Prob (next ~30min)", f"{sig['ml_prob_up']}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sig['price'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [sig['invalid'], sig['stretch']]},
                'bar': {'color': "gold"},
                'steps': [
                    {'range': [sig['invalid'], sig['likely']], 'color': "red"},
                    {'range': [sig['likely'], sig['possible']], 'color': "orange"},
                    {'range': [sig['possible'], sig['stretch']], 'color': "green"}
                ]
            },
            title={'text': "Invalid → Likely → Possible → Stretch"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**Invalid / Stop**: {sig['invalid']:.2f}")
        st.write(f"**VIX**: {sig['vix']:.2f} | ES: {sig['es']} | NQ: {sig['nq']}")

        if st.button(f"Show Chart {ticker}"):
            df_chart = get_data(ticker)
            st.line_chart(df_chart[['Close', 'EMA9', 'VWAP']])

        st.markdown('</div>', unsafe_allow_html=True)

if st.button("Force Refresh", type="primary"):
    st.rerun()

st.caption("Auto-refreshing every 60 seconds • Not financial advice • Trade your plan")

time.sleep(60)
st.rerun()