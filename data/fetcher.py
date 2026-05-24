import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

_cache = {}
_cache_timeout = 300  # 5 minutes


def _is_cache_valid(key):
    if key not in _cache:
        return False
    return (datetime.now() - _cache[key]['timestamp']).seconds < _cache_timeout


def get_historical_data(ticker: str, period: str = '2y') -> pd.DataFrame:
    cache_key = f"hist_{ticker}_{period}"
    if _is_cache_valid(cache_key):
        return _cache[cache_key]['data']

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = 'Date'

        _cache[cache_key] = {'data': df, 'timestamp': datetime.now()}
        return df
    except Exception:
        return pd.DataFrame()


def get_stock_info(ticker: str) -> dict:
    cache_key = f"info_{ticker}"
    if _is_cache_valid(cache_key):
        return _cache[cache_key]['data']

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        result = {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'eps': info.get('trailingEps', None),
            'dividend_yield': info.get('dividendYield', None),
            'beta': info.get('beta', None),
            'week_52_high': info.get('fiftyTwoWeekHigh', None),
            'week_52_low': info.get('fiftyTwoWeekLow', None),
            'avg_volume': info.get('averageVolume', None),
            'description': info.get('longBusinessSummary', ''),
            'website': info.get('website', ''),
            'country': info.get('country', 'N/A'),
            'employees': info.get('fullTimeEmployees', None),
        }
        _cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
        return result
    except Exception:
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A'}


def get_latest_quote(ticker: str) -> dict:
    try:
        df = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
        if df.empty:
            return {}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        price = float(latest['Close'])
        prev_close = float(prev['Close'])
        change = price - prev_close
        pct_change = (change / prev_close) * 100

        return {
            'price': price,
            'prev_close': prev_close,
            'change': change,
            'pct_change': pct_change,
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'volume': int(latest['Volume']),
            'timestamp': str(df.index[-1].date()),
        }
    except Exception:
        return {}


def format_market_cap(value):
    if value is None:
        return 'N/A'
    if value >= 1e12:
        return f'${value/1e12:.2f}T'
    if value >= 1e9:
        return f'${value/1e9:.2f}B'
    if value >= 1e6:
        return f'${value/1e6:.2f}M'
    return f'${value:,.0f}'


def format_volume(value):
    if value is None:
        return 'N/A'
    if value >= 1e9:
        return f'{value/1e9:.2f}B'
    if value >= 1e6:
        return f'{value/1e6:.2f}M'
    if value >= 1e3:
        return f'{value/1e3:.1f}K'
    return str(int(value))


POPULAR_STOCKS = [
    {'label': 'Apple Inc.', 'value': 'AAPL'},
    {'label': 'Microsoft Corp.', 'value': 'MSFT'},
    {'label': 'NVIDIA Corp.', 'value': 'NVDA'},
    {'label': 'Alphabet Inc.', 'value': 'GOOGL'},
    {'label': 'Amazon.com', 'value': 'AMZN'},
    {'label': 'Meta Platforms', 'value': 'META'},
    {'label': 'Tesla Inc.', 'value': 'TSLA'},
    {'label': 'Berkshire Hathaway', 'value': 'BRK-B'},
    {'label': 'SPDR S&P 500 ETF', 'value': 'SPY'},
    {'label': 'Invesco QQQ ETF', 'value': 'QQQ'},
    {'label': 'JPMorgan Chase', 'value': 'JPM'},
    {'label': 'Johnson & Johnson', 'value': 'JNJ'},
    {'label': 'Visa Inc.', 'value': 'V'},
    {'label': 'UnitedHealth Group', 'value': 'UNH'},
    {'label': 'Procter & Gamble', 'value': 'PG'},
    {'label': 'Exxon Mobil', 'value': 'XOM'},
    {'label': 'Netflix Inc.', 'value': 'NFLX'},
    {'label': 'Adobe Inc.', 'value': 'ADBE'},
    {'label': 'Salesforce Inc.', 'value': 'CRM'},
    {'label': 'PayPal Holdings', 'value': 'PYPL'},
]

INDEX_TICKERS = ['SPY', 'QQQ', 'DIA', 'GLD', 'BTC-USD']
