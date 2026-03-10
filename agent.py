import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("agent.py loaded successfully")

ml_model = None
ml_features = ['RSI', 'MACD', 'ADX', 'ATR', 'Close', 'Volume']

def get_data(ticker, interval='5m', period='5d'):
    print(f"Fetching data for {ticker}")
    try:
        df = yf.download(ticker, interval=interval, period=period, prepost=True, progress=False)
        print(f"Data shape for {ticker}: {df.shape}")
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"Data fetch error for {ticker}: {e}")
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
    except Exception as e:
        print(f"CHOP error: {e}")
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
    except Exception as e:
        print(f"BB squeeze error: {e}")
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
    except Exception as e:
        print(f"Divergence error: {e}")
        return False

def get_vix_futures():
    vix = 20.0
    es = 'N/A'
    nq = 'N/A'
    try:
        vix_df = get_data('^VIX', '1m', '1d')
        if not vix_df.empty:
            vix = vix_df['Close'].iloc[-1]
    except Exception as e:
        print(f"VIX fetch error: {e}")
    try:
        es = yf.Ticker('ES=F').info.get('regularMarketPrice', 'N/A')
        nq = yf.Ticker('NQ=F').info.get('regularMarketPrice', 'N/A')
    except:
        pass
    return vix, es, nq

def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = tr.rolling(14).mean()

    plus_dm = (df['High'] - df['High'].shift()).clip(lower=0)
    minus_dm = (df['Low'].shift() - df['Low']).clip(lower=0)
    tr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    df['CHOP'] = pd.Series([calculate_chop(df.iloc[:i+1]) for i in range(len(df))])

    print("Indicators added successfully")
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

def train_simple_ml():
    global ml_model
    try:
        print("Starting ML training")
        df = get_data('SPY', '5m', '730d')
        if df.empty:
            print("No data for ML")
            return
        df = add_indicators(df)
        df['Future_Return'] = df['Close'].shift(-6) / df['Close'] - 1
        df['Target'] = np.where(df['Future_Return'] > 0.002, 1, 0)
        df = df.dropna()
        if len(df) < 10:
            print("Too few data for ML")
            return
        X = df[ml_features].fillna(0)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        print(f"ML trained - accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
        ml_model = model
    except Exception as e:
        print(f"ML training failed: {e}")
        ml_model = None

def predict_ml(df):
    if ml_model is None or df.empty:
        return 50.0
    try:
        latest = df[ml_features].iloc[-1:].fillna(0)
        prob = ml_model.predict_proba(latest)[0][1] * 100
        return round(prob, 1)
    except Exception as e:
        print(f"ML predict error: {e}")
        return 50.0

def generate_signal(ticker='SPY'):
    try:
        print(f"Generating signal for {ticker}")
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

        regime = "TREND" if latest.get('CHOP', 100) < 45 and latest.get('ADX', 0) > 22 else "RANGE"

        current = latest['Close']
        atr = latest.get('ATR', 1.0) if not np.isnan(latest.get('ATR')) else 1.0
        dir_mult = 1 if bias == "BULLISH" else -1 if bias == "BEARISH" else 0

        likely = current + dir_mult * atr * 1.0
        possible = current + dir_mult * atr * 1.5
        stretch = current + dir_mult * atr * 2.5
        invalid = current - dir_mult * atr * 0.5

        why_list = []
        if squeeze: why_list.append("Bollinger Squeeze → potential breakout")
        if div_rsi or div_macd: why_list.append("Bullish divergence detected")
        if vol_surge: why_list.append("Volume surge leading price")
        if vix < 18: why_list.append("VIX low/falling → bullish fuel")

        ml_prob = predict_ml(df_5m)

        signal_text = "No Clear Signal"
        if pressure >= 75 and latest.get('CHOP', 100) < 45 and bias != "NEUTRAL" and len(why_list) >= 1:
            direction = "CALLS" if bias == "BULLISH" else "PUTS"
            signal_text = f"ENTRY ACTIVE — {direction} @ {current:.2f}"

        print(f"Signal generated for {ticker}")
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
    except Exception as e:
        print(f"Generate signal error for {ticker}: {e}")
        return {"error": f"Signal error: {str(e)}"}