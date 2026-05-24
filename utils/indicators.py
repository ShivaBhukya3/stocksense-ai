import pandas as pd
import numpy as np


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    pct_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / sma
    return upper, sma, lower, pct_b, bandwidth


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                          k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    df['SMA_20'] = calculate_sma(close, 20)
    df['SMA_50'] = calculate_sma(close, 50)
    df['SMA_200'] = calculate_sma(close, 200)
    df['EMA_12'] = calculate_ema(close, 12)
    df['EMA_26'] = calculate_ema(close, 26)

    df['RSI'] = calculate_rsi(close)

    macd, signal, hist = calculate_macd(close)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    upper, mid, lower, pct_b, bw = calculate_bollinger_bands(close)
    df['BB_Upper'] = upper
    df['BB_Middle'] = mid
    df['BB_Lower'] = lower
    df['BB_PctB'] = pct_b
    df['BB_BW'] = bw

    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(high, low, close)
    df['ATR'] = calculate_atr(high, low, close)
    df['VWAP'] = calculate_vwap(high, low, close, volume)
    df['OBV'] = calculate_obv(close, volume)

    df['Vol_SMA20'] = calculate_sma(volume, 20)
    df['Vol_Ratio'] = volume / df['Vol_SMA20']

    return df


def get_signal_summary(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    signals = []

    rsi = latest.get('RSI', 50)
    if rsi < 30:
        signals.append(('RSI Oversold', 'BUY', rsi))
    elif rsi > 70:
        signals.append(('RSI Overbought', 'SELL', rsi))
    else:
        signals.append(('RSI Neutral', 'NEUTRAL', rsi))

    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    if macd > macd_signal:
        signals.append(('MACD Bullish', 'BUY', macd))
    else:
        signals.append(('MACD Bearish', 'SELL', macd))

    close = latest['Close']
    sma_50 = latest.get('SMA_50', close)
    sma_200 = latest.get('SMA_200', close)
    if close > sma_50 > sma_200:
        signals.append(('Above SMA 50/200', 'BUY', close))
    elif close < sma_50 < sma_200:
        signals.append(('Below SMA 50/200', 'SELL', close))
    else:
        signals.append(('Mixed MAs', 'NEUTRAL', close))

    buy_count = sum(1 for s in signals if s[1] == 'BUY')
    sell_count = sum(1 for s in signals if s[1] == 'SELL')

    if buy_count > sell_count:
        overall = 'BULLISH'
    elif sell_count > buy_count:
        overall = 'BEARISH'
    else:
        overall = 'NEUTRAL'

    return {'signals': signals, 'overall': overall, 'buy': buy_count, 'sell': sell_count}
