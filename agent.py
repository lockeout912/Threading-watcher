import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("agent.py loaded successfully")

ml_model = None
ml_features = ['RSI', 'MACD', 'ADX', 'ATR', 'Close', 'Volume']


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def get_data(ticker, interval='5m', period='5d'):
    print(f"Fetching data for {ticker} | interval={interval} | period={period}")
    try:
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            prepost=True,
            progress=False,
            auto_adjust=False,
            threads=False
        )
        df = flatten_columns(df)

        if df.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()

        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in needed:
            if col not in df.columns:
                print(f"Missing column {col} for {ticker}")
                return pd.DataFrame()

        df = df.dropna(subset=['Close']).copy()
        print(f"Data shape for {ticker}: {df.shape}")
        return df

    except Exception as e:
        print(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()


def calculate_chop(df, period=14):
    try:
        if len(df) < period:
            return np.nan

        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ],
            axis=1
        ).max(axis=1)

        atr_sum = tr.rolling(period).sum()
        range_hl = high.rolling(period).max() - low.rolling(period).min()
        range_hl = range_hl.replace(0, np.nan)

        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        return chop.iloc[-1]

    except Exception as e:
        print(f"CHOP error: {e}")
        return np.nan


def calculate_bb_squeeze(df, period=20):
    try:
        if len(df) < max(period, 50):
            return False, np.nan

        rolling_mean = df['Close'].rolling(period).mean()
        rolling_std = df['Close'].rolling(period).std()

        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std

        bb_width = (upper - lower) / rolling_mean.replace(0, np.nan)
        rolling_min = bb_width.rolling(50).min()
        squeeze = bb_width < (rolling_min * 1.05)

        return bool(squeeze.iloc[-1]) if pd.notna(squeeze.iloc[-1]) else False, float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else np.nan

    except Exception as e:
        print(f"BB squeeze error: {e}")
        return False, np.nan


def detect_divergence(df, col, lookback=20):
    try:
        if df.empty or col not in df.columns or len(df) < lookback:
            return False

        prices = df['Close'].iloc[-lookback:]
        ind = df[col].iloc[-lookback:]

        if prices.isna().all() or ind.isna().all():
            return False

        price_ll_idx = prices.values.argmin()
        ind_ll_idx = ind.values.argmin()

        if price_ll_idx > ind_ll_idx and ind.iloc[-1] > ind.iloc[ind_ll_idx]:
            return True

        return False

    except Exception as e:
        print(f"Divergence error ({col}): {e}")
        return False


def get_last_price_fast(ticker, fallback='N/A'):
    try:
        df = yf.download(
            ticker,
            interval='1m',
            period='1d',
            progress=False,
            auto_adjust=False,
            threads=False
        )
        df = flatten_columns(df)
        if not df.empty and 'Close' in df.columns:
            return float(df['Close'].iloc[-1])
    except Exception as e:
        print(f"Last price fetch error for {ticker}: {e}")
    return fallback


def get_vix_futures():
    vix = 20.0
    es = 'N/A'
    nq = 'N/A'

    try:
        vix_df = get_data('^VIX', '1m', '1d')
        if not vix_df.empty:
            vix = float(vix_df['Close'].iloc[-1])
    except Exception as e:
        print(f"VIX fetch error: {e}")

    es = get_last_price_fast('ES=F', 'N/A')
    nq = get_last_price_fast('NQ=F', 'N/A')

    return vix, es, nq


def add_indicators(df):
    if df.empty:
        return df

    df = df.copy()

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ],
        axis=1
    ).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.rolling(14).mean()

    df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()

    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df['Stoch_K'] = 100 * (df['Close'] - low14) / denom
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tpv = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum().replace(0, np.nan)
    df['VWAP'] = cumulative_tpv / cumulative_vol

    df['CHOP'] = pd.Series(
        [calculate_chop(df.iloc[:i + 1]) for i in range(len(df))],
        index=df.index
    )

    print("Indicators added successfully")
    return df


def pressure_score(df):
    if df.empty:
        return 0, "COLD"

    latest = df.iloc[-1]
    score = 0

    if pd.notna(latest.get('RSI')) and latest['RSI'] > 60:
        score += 20
    if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_sig')) and latest['MACD'] > latest['MACD_sig']:
        score += 20
    if pd.notna(latest.get('EMA9')) and pd.notna(latest.get('VWAP')) and latest['Close'] > latest['EMA9'] > latest['VWAP']:
        score += 25
    if pd.notna(latest.get('ADX')) and latest['ADX'] > 25:
        score += 15
    if pd.notna(latest.get('CHOP')) and latest['CHOP'] < 40:
        score += 10
    if len(df) > 1 and pd.notna(df['OBV'].diff().iloc[-1]) and df['OBV'].diff().iloc[-1] > 0:
        score += 10

    score = min(score, 100)
    heat = "HOT" if score >= 75 else "WARM" if score >= 45 else "COLD"
    return score, heat


def train_simple_ml():
    global ml_model

    try:
        print("Starting ML training")

        # Yahoo intraday history is limited, so keep this realistic
        df = get_data('SPY', '5m', '60d')
        if df.empty:
            print("No data for ML")
            return False

        df = add_indicators(df)

        df['Future_Return'] = df['Close'].shift(-6) / df['Close'] - 1
        df['Target'] = np.where(df['Future_Return'] > 0.002, 1, 0)

        df = df.dropna(subset=ml_features + ['Target'])

        if len(df) < 50:
            print("Too few rows for ML training")
            return False

        X = df[ml_features].fillna(0)
        y = df['Target']

        if y.nunique() < 2:
            print("Target has only one class; ML training skipped")
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            min_samples_split=10
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"ML trained - accuracy: {acc:.2%}")
        ml_model = model
        return True

    except Exception as e:
        print(f"ML training failed: {e}")
        ml_model = None
        return False


def predict_ml(df):
    if ml_model is None or df.empty:
        return 50.0

    try:
        latest = df[ml_features].iloc[-1:].fillna(0)
        prob = ml_model.predict_proba(latest)[0][1] * 100
        return round(float(prob), 1)

    except Exception as e:
        print(f"ML predict error: {e}")
        return 50.0


def build_neutral_response(ticker, price, vix, es, nq, message="No Clear Signal"):
    return {
        'ticker': ticker,
        'price': float(price) if pd.notna(price) else 0.0,
        'bias': 'NEUTRAL',
        'regime': 'RANGE',
        'pressure': 0,
        'heat': 'COLD',
        'vix': float(vix) if pd.notna(vix) else 20.0,
        'es': es,
        'nq': nq,
        'why': [],
        'likely': float(price) if pd.notna(price) else 0.0,
        'possible': float(price) if pd.notna(price) else 0.0,
        'stretch': float(price) if pd.notna(price) else 0.0,
        'invalid': float(price) if pd.notna(price) else 0.0,
        'ml_prob_up': 50.0,
        'signal': message
    }


def generate_signal(ticker='SPY'):
    try:
        print(f"Generating signal for {ticker}")

        df_5m = get_data(ticker, '5m', '5d')
        if df_5m.empty:
            return {"error": f"No data returned for {ticker}"}

        df_5m = add_indicators(df_5m)
        latest = df_5m.iloc[-1]

        current = float(latest['Close'])
        vix, es, nq = get_vix_futures()

        if vix > 22:
            return build_neutral_response(
                ticker=ticker,
                price=current,
                vix=vix,
                es=es,
                nq=nq,
                message="High VIX — Choppy / Avoid"
            )

        pressure, heat = pressure_score(df_5m)
        squeeze, bbw = calculate_bb_squeeze(df_5m)
        div_rsi = detect_divergence(df_5m, 'RSI')
        div_macd = detect_divergence(df_5m, 'MACD')
        vol_surge = bool(len(df_5m) > 1 and pd.notna(df_5m['OBV'].diff().iloc[-1]) and df_5m['OBV'].diff().iloc[-1] > 0)

        close_val = latest.get('Close', np.nan)
        vwap_val = latest.get('VWAP', np.nan)
        ema9_val = latest.get('EMA9', np.nan)

        if pd.notna(close_val) and pd.notna(vwap_val) and pd.notna(ema9_val):
            if close_val > vwap_val and ema9_val > vwap_val:
                bias = "BULLISH"
            elif close_val < vwap_val and ema9_val < vwap_val:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"
        else:
            bias = "NEUTRAL"

        chop_val = latest.get('CHOP', 100)
        adx_val = latest.get('ADX', 0)
        regime = "TREND" if pd.notna(chop_val) and pd.notna(adx_val) and chop_val < 45 and adx_val > 22 else "RANGE"

        atr = latest.get('ATR', np.nan)
        atr = float(atr) if pd.notna(atr) and atr > 0 else max(current * 0.0025, 0.25)

        dir_mult = 1 if bias == "BULLISH" else -1 if bias == "BEARISH" else 0

        likely = current + dir_mult * atr * 1.0
        possible = current + dir_mult * atr * 1.5
        stretch = current + dir_mult * atr * 2.5
        invalid = current - dir_mult * atr * 0.5

        why_list = []
        if squeeze:
            why_list.append("Bollinger squeeze → potential breakout")
        if div_rsi or div_macd:
            why_list.append("Divergence detected")
        if vol_surge:
            why_list.append("Volume surge leading price")
        if vix < 18:
            why_list.append("Low VIX → cleaner tape")

        ml_prob = predict_ml(df_5m)

        signal_text = "No Clear Signal"
        if pressure >= 75 and chop_val < 45 and bias != "NEUTRAL" and len(why_list) >= 1:
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
            'vix': float(vix),
            'es': es,
            'nq': nq,
            'why': why_list,
            'likely': float(likely),
            'possible': float(possible),
            'stretch': float(stretch),
            'invalid': float(invalid),
            'ml_prob_up': float(ml_prob),
            'signal': signal_text
        }

    except Exception as e:
        print(f"Generate signal error for {ticker}: {e}")
        return {"error": f"Signal error: {str(e)}"}