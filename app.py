import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(page_title="Lockout Signals", layout="wide")

st.title("Lockout Signals • SPY / QQQ")
st.caption("Safe-mode build to get the app running first.")

tickers = ["SPY", "QQQ"]

for ticker in tickers:
    st.subheader(ticker)

    sig = generate_signal(ticker)

    if "error" in sig:
        st.error(sig["error"])
        continue

    st.write(f"**Price:** ${sig['price']:.2f}")
    st.write(f"**Bias:** {sig['bias']}")
    st.write(f"**Signal:** {sig['signal']}")
    st.write(f"**Pressure:** {sig['pressure']}/100")
    st.write(f"**Heat:** {sig['heat']}")
    st.write(f"**RSI:** {sig['rsi']:.2f}")
    st.write(f"**EMA9:** {sig['ema9']:.2f}")
    st.write(f"**VWAP:** {sig['vwap']:.2f}")

    if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
        df = get_data(ticker)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "VWAP"] if c in df.columns]
            st.line_chart(df[cols])

st.divider()
st.write("Framework is live. Next step: rebuild the battle-map version cleanly.")