import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Helper Functions
# -------------------------------

def get_data(ticker, interval='5m', period='5d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, prepost=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def calculate_chop(df, period=14):
    if len(df) < period:
        return np.nan
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    range_high_low = (df['High'].rolling(period).max() - df['Low'].rolling(period).min())
    chop = 100 * np.log10(atr_sum / period / range_high_low) / np.log10(period)
    return chop.iloc[-1] if not chop.empty else np.nan

def calculate_bb_squeeze(df, period=20):
    if len(df) < period:
        return False, np.nan
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=period, nbdevup=2, nbdevdn=2)
    bb_width = (upper - lower) / middle
    squeeze = bb_width < bb_width.rolling(50).min() * 1.05
    return squeeze.iloc[-1], bb_width.iloc[-1]

def detect_divergence(df, col='RSI', lookback=20):
    if len(df) < lookback:
        return False
    prices = df['Close'].iloc[-lookback:]
    ind = df[col].iloc[-lookback:]
    price_ll_idx = prices.argmin()
    ind_ll_idx = ind.argmin()
    if price_ll_idx > ind_ll_idx and ind.iloc[-1] > ind.iloc[ind_ll_idx]:
        return True
    return False

def get_vix_futures():
    try:
        vix_df = get_data('^VIX', '1m', '1d')
        vix = vix_df['Close'].iloc[-1] if not vix_df.empty else 20.0
    except:
        vix = 20.0
    es = yf.Ticker('ES=F').info.get('regularMarketPrice', 'N/A')
    nq = yf.Ticker('NQ=F').info.get('regularMarketPrice', 'N/A')
    return vix, es, nq

# -------------------------------
# Indicators & Scoring
# -------------------------------

def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df['EMA9'] = talib.EMA(df['Close'], timeperiod=9)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_sig'], _ = talib.MACD(df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    df['CHOP'] = pd.Series([calculate_chop(df.iloc[:i+1]) for i in range(len(df))])
    return df

def pressure_score(df):
    if df.empty:
        return 0, "COLD"
    latest = df.iloc[-1]
    score = 0
    if latest['RSI'] > 60: score += 20
    if latest['MACD'] > latest['MACD_sig']: score += 20
    if latest['Close'] > latest['EMA9'] > latest['VWAP']: score += 25
    if latest['ADX'] > 25: score += 15
    if latest.get('CHOP', 100) < 40: score += 10
    if df['OBV'].diff().iloc[-1] > 0: score += 10
    score = min(score, 100)
    heat = "HOT" if score >= 75 else "WARM" if score >= 45 else "COLD"
    return score, heat

# -------------------------------
# Simple ML Predictor (trained once on load)
# -------------------------------

ml_model = None
ml_features = ['RSI', 'MACD', 'ADX', 'ATR', 'Close', 'Volume']

def train_simple_ml():
    global ml_model
    try:
        df = get_data('SPY', '5m', '730d')  # ~2 years
        if df.empty:
            return
        df = add_indicators(df)
        df['Future_Return'] = df['Close'].shift(-6) / df['Close'] - 1  # next ~30 min
        df['Target'] = np.where(df['Future_Return'] > 0.002, 1, 0)  # up >0.2%
        df = df.dropna()

        X = df[ml_features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"ML Backtest Accuracy: {acc:.2%}")
        ml_model = model
    except Exception as e:
        print(f"ML training failed: {e}")
        ml_model = None

def predict_ml(df):
    if ml_model is None or df.empty:
        return 50.0
    latest = df[ml_features].iloc[-1:].fillna(0)
    prob = ml_model.predict_proba(latest)[0][1] * 100  # prob of up move
    return round(prob, 1)

# -------------------------------
# Main Signal Generator
# -------------------------------

def generate_signal(ticker='SPY'):
    df_5m = get_data(ticker, '5m', '5d')
    if df_5m.empty:
        return {"error": "No data"}

    df_5m = add_indicators(df_5m)
    latest = df_5m.iloc[-1]

    vix, es, nq = get_vix_futures()
    if vix > 22:
        return {"signal": "High VIX – Choppy / Avoid", "bias": "NEUTRAL"}

    pressure, heat = pressure_score(df_5m)
    squeeze, bbw = calculate_bb_squeeze(df_5m)
    div_rsi = detect_divergence(df_5m, 'RSI')
    div_macd = detect_divergence(df_5m, 'MACD')
    vol_surge = df_5m['OBV'].diff().iloc[-1] > 0 if len(df_5m) > 1 else False

    bias = "BULLISH" if latest['Close'] > latest['VWAP'] and latest['EMA9'] > latest['VWAP'] else \
           "BEARISH" if latest['Close'] < latest['VWAP'] and latest['EMA9'] < latest['VWAP'] else "NEUTRAL"

    regime = "TREND" if latest.get('CHOP', 100) < 45 and latest['ADX'] > 22 else "RANGE"

    current = latest['Close']
    atr = latest['ATR'] if not np.isnan(latest['ATR']) else 1.0
    dir_mult = 1 if bias == "BULLISH" else -1 if bias == "BEARISH" else 0

    likely   = current + dir_mult * atr * 1.0
    possible = current + dir_mult * atr * 1.5
    stretch  = current + dir_mult * atr * 2.5
    invalid  = current - dir_mult * atr * 0.5

    why_list = []
    if squeeze: why_list.append("Bollinger Squeeze → potential breakout")
    if div_rsi or div_macd: why_list.append("Bullish divergence detected")
    if vol_surge: why_list.append("Volume surge leading price")
    if vix < 18: why_list.append("VIX low/falling → bullish fuel")

    ml_prob = predict_ml(df_5m)

    signal_text = "No Clear Signal"
    if pressure >= 75 and latest.get('CHOP', 100) < 45 and bias != "NEUTRAL" and len(why_list) >= 2:
        direction = "CALLS" if bias == "BULLISH" else "PUTS"
        signal_text = f"ENTRY ACTIVE — {direction} @ {current:.2f}"

    return {
        'ticker': ticker,
        'price': current,
        'bias': bias,
        'regime': regime,
        'pressure': pressure,
        'heat': heat,
        'vix': vix,
        'es': es,
        'nq': nq,
        'why': why_list,
        'likely': likely,
        'possible': possible,
        'stretch': stretch,
        'invalid': invalid,
        'ml_prob_up': ml_prob,
        'signal': signal_text
    }