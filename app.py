import streamlit as st
import time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Safe global ML model
ml_model = None
ml_features = ['RSI', 'MACD', 'ADX', 'ATR', 'Close', 'Volume']

# -------------------------------
# Data & Helpers
# -------------------------------

@st.cache_data(ttl=60)
def get_data(ticker, interval='5m', period='5d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, prepost=True, progress=False)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_chop(df, period=14):
    try:
        if len(df) < period:
            return np.nan
        tr = np.maximum(df['High'] - df['Low'],
                        np.maximum(abs(df['High'] - df['Close'].shift()),
                                   abs(df['Low'] - df['Close'].shift())))
        atr_sum = tr.rolling(period).sum()
        range_hl = (df['High'].rolling(period).max() - df['Low'].rolling(period).min())
        chop = 100 * np.log10(atr_sum / period / range_hl) / np.log10(period)
        return chop.iloc[-1]
    except:
        return np.nan

def calculate_bb_squeeze(df, period=20):
    try:
        if len(df) < period:
            return False, np.nan
        rolling_mean = df['Close'].rolling(period).mean()
        rolling_std = df['Close'].rolling(period).std()
        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std
        bb_width = (upper - lower) / rolling_mean
        squeeze = bb_width < bb_width.rolling(50).min() * 1.05
        return squeeze.iloc[-1], bb_width.iloc[-1]
    except:
        return False, np.nan

def detect_divergence(df, col, lookback=20):
    try:
        if len(df) < lookback:
            return False
        prices = df['Close'].iloc[-lookback:]
        ind = df[col].iloc[-lookback:]
        price_ll_idx = prices.argmin()
        ind_ll_idx = ind.argmin()
        if price_ll_idx > ind_ll_idx and ind.iloc[-1] > ind.iloc[ind_ll_idx]:
            return True
        return False
    except:
        return False

def get_vix_futures():
    vix = 20.0
    try:
        vix_df = get_data('^VIX', '1m', '1d')
        if not vix_df.empty:
            vix = vix_df['Close'].iloc[-1]
    except:
        pass
    es = yf.Ticker('ES=F').info.get('regularMarketPrice', 'N/A')
    nq = yf.Ticker('NQ=F').info.get('regularMarketPrice', 'N/A')
    return vix, es, nq

# -------------------------------
# Manual Indicators
# -------------------------------

def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()

    # EMA9
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    # RSI14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR14
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = tr.rolling(14).mean()

    # ADX14 (approx)
    plus_dm = (df['High'] - df['High'].shift()).clip(lower=0)
    minus_dm = (df['Low'].shift() - df['Low']).clip(lower=0)
    tr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Stoch
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # CHOP
    df['CHOP'] = pd.Series([calculate_chop(df.iloc[:i+1]) for i in range(len(df))])

    return df

def pressure_score(df):
    if df.empty:
        return 0, "COLD"
    latest = df.iloc[-1]
    score = 0
    if pd.notna(latest.get('RSI')) and latest['RSI'] > 60: score += 20
    if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_sig')) and latest['MACD'] > latest['MACD_sig']: score += 20
    if pd.notna(latest.get('EMA9')) and pd.notna(latest.get('VWAP')) and latest['Close'] > latest['EMA9'] > latest['VWAP']: score += 25
    if pd.notna(latest.get('ADX')) and latest['ADX'] > 25: score += 15
    if pd.notna(latest.get('CHOP')) and latest['CHOP'] < 40: score += 10
    if len(df) > 1 and df['OBV'].diff().iloc[-1] > 0: score += 10
    score = min(score, 100)
    heat = "HOT" if score >= 75 else "WARM" if score >= 45 else "COLD"
    return score, heat

# ML training (delayed & safe)
def train_simple_ml():
    global ml_model
    try:
        df = get_data('SPY', '5m', '730d')
        if df.empty:
            return
        df = add_indicators(df)
        df['Future_Return'] = df['Close'].shift(-6) / df['Close'] - 1
        df['Target'] = np.where(df['Future_Return'] > 0.002, 1, 0)
        df = df.dropna()
        if len(df) < 10:
            return
        X = df[ml_features].fillna(0)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        ml_model = model
    except Exception as e:
        st.warning(f"ML training skipped: {e}")

def predict_ml(df):
    if ml_model is None or df.empty:
        return 50.0
    try:
        latest = df[ml_features].iloc[-1:].fillna(0)
        prob = ml_model.predict_proba(latest)[0][1] * 100
        return round(prob, 1)
    except:
        return 50.0

# -------------------------------
# Dashboard
# -------------------------------

st.set_page_config(page_title="Lockout Signals • SPY/QQQ", layout="wide")

# Train ML safely
if 'ml_trained' not in st.session_state:
    train_simple_ml()
    st.session_state.ml_trained = True

# CSS (fancy dark mode)
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

# Ticker scroll
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