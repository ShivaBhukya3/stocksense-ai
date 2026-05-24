"""StockSense AI — Quant Finance Dashboard"""
import warnings, json, logging
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

import dash
from dash import dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.fetcher import (
    get_historical_data, get_stock_info, get_latest_quote,
    format_market_cap, format_volume, POPULAR_STOCKS,
)
from models.lstm_model import LSTMForecaster

# ── Helpers ────────────────────────────────────────────────────────
def _df_from_store(ohlcv_json: str) -> pd.DataFrame:
    """Safely parse the stored OHLCV JSON back to a sorted DataFrame."""
    df = pd.read_json(ohlcv_json)
    # yfinance 1.4 index has name=None → reset_index creates 'index'; fix applied
    # in fetcher.py (index.name='Date'), but keep fallback here too.
    date_col = 'Date' if 'Date' in df.columns else 'index'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = 'Date'
    return df
from models.prophet_model import ProphetForecaster
from utils.indicators import add_all_indicators, get_signal_summary

# ── App ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
        {'name': 'theme-color', 'content': '#04060f'},
    ],
)
app.title = 'StockSense AI — Quant Finance Dashboard'
server = app.server

DEFAULT_STOCK = 'AAPL'
WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'SPY']

# ── Plotly helpers ─────────────────────────────────────────────────
def dl(**kw):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter, sans-serif', size=12),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', size=10),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        hoverlabel=dict(
            bgcolor='rgba(10,16,30,0.95)',
            bordercolor='rgba(0,200,255,0.25)',
            font=dict(color='#f1f5f9', size=12),
        ),
        margin=dict(l=55, r=20, t=30, b=50),
        hovermode='x unified',
        dragmode='pan',
    )
    base.update(kw)
    return base

def ax(**kw):
    base = dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(color='#64748b', size=11),
        zeroline=False,
    )
    base.update(kw)
    return base

EMPTY = go.Figure(layout=dl(
    annotations=[dict(
        text='No data — select a stock',
        xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(color='#475569', size=14),
    )]
))

# ── Header ─────────────────────────────────────────────────────────
header = html.Header([
    html.Div([
        html.Div('📈', className='brand-icon'),
        html.Span('StockSense', className='brand-name'),
        html.Span('AI', style={
            'background': 'linear-gradient(135deg,#00c8ff,#8b5cf6)',
            'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
            'fontWeight': '900', 'fontSize': '17px',
        }),
        html.Span('PRO', className='brand-tag'),
    ], className='header-brand', style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}),

    html.Div([
        html.Span(className='status-dot live'),
        html.Span('LIVE', style={'fontSize': '11px', 'fontWeight': '700'}),
    ], id='market-status', className='market-status open'),

    html.Div([
        html.Div(id='ticker-content', className='ticker-track'),
    ], className='ticker-wrapper'),

    html.Div([
        html.Div('--:--:--', id='header-clock', className='header-time'),
        html.Span('⚙', className='header-btn', id='settings-btn', n_clicks=0, title='Settings'),
    ], className='header-right'),
], className='app-header')

# ── Settings Modal ─────────────────────────────────────────────────
_mstyle = {'background':'rgba(4,6,15,0.98)'}
settings_modal = dbc.Modal([
    dbc.ModalHeader(
        html.Span('⚙  Settings', style={'fontWeight':'700','color':'#f1f5f9','fontSize':'14px','letterSpacing':'0.05em'}),
        close_button=True,
        style={**_mstyle, 'borderBottom':'1px solid rgba(255,255,255,0.06)','padding':'16px 20px'},
    ),
    dbc.ModalBody([
        html.Div('DISPLAY', className='sidebar-label', style={'marginBottom':'10px'}),
        html.Div([
            html.Div('Chart Type', style={'color':'#94a3b8','fontSize':'12px','marginBottom':'8px'}),
            dcc.RadioItems(
                id='set-chart-type',
                options=[{'label':' Candlestick','value':'candlestick'},
                         {'label':' Line','value':'line'}],
                value='candlestick', inline=True,
                inputStyle={'marginRight':'4px'},
                labelStyle={'marginRight':'16px','fontSize':'12px','color':'#94a3b8','cursor':'pointer'},
            ),
        ], style={'marginBottom':'20px'}),

        html.Hr(style={'borderColor':'rgba(255,255,255,0.06)','margin':'0 0 18px 0'}),
        html.Div('DATA REFRESH', className='sidebar-label', style={'marginBottom':'10px'}),
        html.Div([
            html.Div('Auto Refresh', style={'color':'#94a3b8','fontSize':'12px','marginBottom':'8px'}),
            dbc.Switch(id='set-auto-refresh', value=True, label='Enabled',
                      style={'color':'#94a3b8','fontSize':'12px'}),
        ], style={'marginBottom':'16px'}),
        html.Div([
            html.Div('Interval', style={'color':'#94a3b8','fontSize':'12px','marginBottom':'8px'}),
            dcc.RadioItems(
                id='set-interval',
                options=[{'label':' 1 min','value':60_000},
                         {'label':' 2 min','value':120_000},
                         {'label':' 5 min','value':300_000}],
                value=120_000, inline=True,
                inputStyle={'marginRight':'4px'},
                labelStyle={'marginRight':'14px','fontSize':'12px','color':'#94a3b8','cursor':'pointer'},
            ),
        ], style={'marginBottom':'20px'}),

        html.Hr(style={'borderColor':'rgba(255,255,255,0.06)','margin':'0 0 18px 0'}),
        html.Div('ABOUT', className='sidebar-label', style={'marginBottom':'10px'}),
        html.Div([
            html.Div('StockSense AI  v1.0', style={'color':'#64748b','fontSize':'12px'}),
            html.Div('LSTM · Prophet · yFinance · Plotly Dash', style={'color':'#475569','fontSize':'11px','marginTop':'4px'}),
            html.Div('For educational purposes only', style={'color':'#334155','fontSize':'10px','marginTop':'4px'}),
        ]),
    ], style={**_mstyle, 'padding':'20px'}),
], id='settings-modal', is_open=False, size='sm',
   contentClassName='settings-modal-content')

# ── Sidebar ────────────────────────────────────────────────────────
sidebar = html.Nav([
    html.Div([
        html.Div('SEARCH STOCK', className='sidebar-label'),
        html.Div([
            html.I(className='fas fa-search'),
            dcc.Dropdown(
                id='stock-selector',
                options=POPULAR_STOCKS,
                value=DEFAULT_STOCK,
                searchable=True,
                clearable=False,
                placeholder='Search ticker…',
                className='stock-search',
            ),
        ], className='search-wrapper'),
    ], className='sidebar-section'),

    html.Div([
        html.Div('WATCHLIST', className='sidebar-label'),
        html.Div(id='watchlist-div', children=[
            html.Div([
                html.Div([
                    html.Div(s, className='wl-symbol'),
                    html.Div('—', className='wl-price', id=f'wl-p-{s}'),
                ]),
                html.Div('—', className='wl-change', id=f'wl-c-{s}'),
            ], className='watchlist-item', n_clicks=0, id=f'wl-{s}')
            for s in WATCHLIST
        ]),
    ], className='sidebar-section'),

    html.Div([
        html.Div('INFO', className='sidebar-label'),
        html.Div([
            html.Div('Data via Yahoo Finance', style={'color': 'var(--text-secondary)', 'fontSize': '11px'}),
            html.Div('15-min delayed quotes', style={'color': 'var(--text-muted)', 'fontSize': '10px', 'marginTop': '2px'}),
            html.Div('LSTM · Prophet · PyTorch', style={'color': 'var(--text-muted)', 'fontSize': '10px', 'marginTop': '2px'}),
        ]),
    ], className='sidebar-section'),
], className='sidebar')

# ── Stock info bar ─────────────────────────────────────────────────
info_bar = html.Div([
    html.Div([
        html.Div(DEFAULT_STOCK, id='d-ticker', className='stock-ticker'),
        html.Div('Loading…', id='d-company', className='stock-company'),
    ], className='stock-name-block'),

    html.Div(className='divider-v'),

    html.Div([
        html.Div('—', id='d-price', className='price-current'),
        html.Div([
            html.Span('—', id='d-change', className='price-change'),
            html.Span('—', id='d-pct',    className='price-pct'),
            html.Span('', id='d-ts',      style={'fontSize':'10px','color':'var(--text-faint)','marginLeft':'6px'}),
        ], className='price-change-row'),
    ], className='price-block'),

    html.Div(className='divider-v'),
    *[html.Div([html.Div(lbl, className='stat-pill-label'),
                html.Div('—', id=sid, className='stat-pill-value')], className='stat-pill')
      for lbl, sid in [('OPEN','s-open'),('HIGH','s-high'),('LOW','s-low'),
                       ('VOLUME','s-vol'),('MKT CAP','s-mkt'),('P/E','s-pe')]],

    html.Div([
        html.Div([
            html.Button(r, id=f'rng-{r.lower()}', className='range-btn' + (' active' if r=='1Y' else ''), n_clicks=0)
            for r in ['1M','3M','6M','1Y','2Y']
        ], className='range-btn-group'),
    ], className='controls-right'),
], className='stock-info-bar')

# ── Overview tab ───────────────────────────────────────────────────
overview_tab = html.Div([
    html.Div([
        dcc.Loading(
            dcc.Graph(id='g-overview', config={'displayModeBar': True, 'scrollZoom': True},
                      style={'height': '420px'}),
            type='dot', color='#00c8ff',
        ),
    ], className='chart-card'),

    html.Div([
        *[html.Div([html.Div(lbl, className='metric-label'),
                    html.Div('—', id=mid, className='metric-value ' + cls)], className='metric-card')
          for lbl, mid, cls in [
              ('52W HIGH','m-52h','cyan'), ('52W LOW','m-52l',''),
              ('BETA','m-beta',''),        ('DIV YIELD','m-div',''),
              ('EPS','m-eps',''),          ('FWD P/E','m-fpe',''),
              ('AVG VOLUME','m-avgvol',''),('SECTOR','m-sector',''),
          ]],
    ], className='metric-grid', style={'marginTop':'12px'}),
], className='tab-content-panel')

# ── Forecast tab ───────────────────────────────────────────────────
forecast_tab = html.Div([
    html.Div([
        html.Div([
            html.Div('MODEL', className='control-label'),
            dcc.RadioItems(
                id='model-sel',
                options=[{'label':'🔷 LSTM','value':'lstm'},
                         {'label':'🔮 Prophet','value':'prophet'},
                         {'label':'⚡ Ensemble','value':'both'}],
                value='both', inline=True,
                inputStyle={'marginRight':'4px'},
                labelStyle={'marginRight':'14px','fontSize':'12px','fontWeight':'600',
                            'color':'#94a3b8','cursor':'pointer'},
            ),
        ], className='control-group'),

        html.Div(className='divider-v'),

        html.Div([
            html.Div('HORIZON (DAYS)', className='control-label'),
            dcc.Slider(id='horizon', min=7, max=90, step=7, value=30,
                       marks={7:'7',14:'14',30:'30',60:'60',90:'90'},
                       tooltip={'placement':'bottom','always_visible':False}),
        ], className='control-group', style={'flex':'1','minWidth':'200px'}),

        html.Div(className='divider-v'),

        html.Div([
            html.Div('TRAINING PERIOD', className='control-label'),
            dcc.RadioItems(
                id='train-period',
                options=[{'label':'1Y','value':'1y'},{'label':'2Y','value':'2y'},{'label':'5Y','value':'5y'}],
                value='2y', inline=True,
                inputStyle={'marginRight':'4px'},
                labelStyle={'marginRight':'10px','fontSize':'12px','fontWeight':'600','color':'#94a3b8'},
            ),
        ], className='control-group'),

        html.Div(className='divider-v'),

        html.Button(
            [html.I(className='fas fa-bolt'), ' Generate Forecast'],
            id='gen-btn', n_clicks=0, className='btn-generate',
        ),
    ], className='forecast-controls'),

    dcc.Loading(
        html.Div([
            dcc.Graph(id='g-forecast', config={'displayModeBar':True,'scrollZoom':True},
                      figure=EMPTY, style={'height':'400px'}),
        ], className='chart-card', style={'marginTop':'12px'}),
        type='dot', color='#00c8ff',
    ),

    html.Div(id='forecast-metrics', style={'marginTop':'12px'}),
    html.Div(id='forecast-table',   style={'marginTop':'12px'}),
], className='tab-content-panel')

# ── Technical tab ──────────────────────────────────────────────────
technical_tab = html.Div([
    html.Div(id='signal-card'),
    dcc.Loading(
        html.Div([
            dcc.Graph(id='g-tech', config={'displayModeBar':True},
                      style={'height':'620px'}),
        ], className='chart-card', style={'marginTop':'12px'}),
        type='dot', color='#00c8ff',
    ),
], className='tab-content-panel')

# ── AI Insights tab ────────────────────────────────────────────────
insights_tab = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            html.Div('LSTM TRAINING HISTORY', className='card-title', style={'marginBottom':'12px'}),
            dcc.Graph(id='g-train', config={'displayModeBar':False},
                      style={'height':'260px'}, figure=EMPTY),
        ], className='chart-card'), width=8),
        dbc.Col(html.Div([
            html.Div('AI MODEL CONFIDENCE', className='card-title', style={'marginBottom':'12px'}),
            dcc.Graph(id='g-gauge', config={'displayModeBar':False},
                      style={'height':'210px'}, figure=EMPTY),
        ], className='chart-card'), width=4),
    ], className='g-3'),
    html.Div(style={'height':'12px'}),
    dbc.Row([
        dbc.Col(html.Div([
            html.Div('30-DAY BACKTEST', className='card-title', style={'marginBottom':'12px'}),
            dcc.Graph(id='g-backtest', config={'displayModeBar':False},
                      style={'height':'260px'}, figure=EMPTY),
        ], className='chart-card'), width=6),
        dbc.Col(html.Div([
            html.Div('FEATURE IMPORTANCE', className='card-title', style={'marginBottom':'12px'}),
            dcc.Graph(id='g-feat', config={'displayModeBar':False},
                      style={'height':'260px'}, figure=EMPTY),
        ], className='chart-card'), width=6),
    ], className='g-3'),
], className='tab-content-panel')

# ── Layout ─────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Store(id='fc-store'),
    dcc.Interval(id='iv-data',   interval=120_000, n_intervals=0),
    dcc.Interval(id='iv-ticker', interval=300_000, n_intervals=0),

    settings_modal,
    header,
    html.Div([
        sidebar,
        html.Main([
            info_bar,
            dbc.Tabs([
                dbc.Tab(overview_tab,  label='📊 Overview',    tab_id='overview'),
                dbc.Tab(forecast_tab,  label='🤖 Forecast',    tab_id='forecast'),
                dbc.Tab(technical_tab, label='📈 Technical',   tab_id='technical'),
                dbc.Tab(insights_tab,  label='🧠 AI Insights', tab_id='insights'),
            ], id='tabs', active_tab='overview', className='main-tabs'),
        ], className='main-content'),
    ], className='app-body'),

    html.Footer([
        html.Span([html.Span('StockSense', style={'background':'linear-gradient(135deg,#00c8ff,#8b5cf6)',
            'WebkitBackgroundClip':'text','WebkitTextFillColor':'transparent','fontWeight':'800'}),
            ' AI — Quant Finance Dashboard']),
        html.Span('LSTM · Prophet · yFinance · PyTorch | For educational purposes only',
                  style={'color':'var(--text-faint)','fontSize':'10px'}),
        html.Span('© 2025', style={'color':'var(--text-faint)'}),
    ], className='app-footer'),
], id='app-root', className='app-root')


# ════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════

@app.callback(
    Output('data-store', 'data'),
    [Input('stock-selector', 'value'), Input('iv-data', 'n_intervals')],
)
def fetch_data(ticker, _):
    if not ticker:
        return no_update
    df = get_historical_data(ticker, '2y')
    if df.empty:
        return None
    return {
        'ticker': ticker,
        'ohlcv':  df.reset_index().to_json(date_format='iso'),
        'info':   get_stock_info(ticker),
        'quote':  get_latest_quote(ticker),
    }


@app.callback(
    [Output('d-ticker','children'), Output('d-company','children'),
     Output('d-price','children'),
     Output('d-change','children'), Output('d-change','className'),
     Output('d-pct','children'),    Output('d-pct','className'),
     Output('d-ts','children'),
     Output('s-open','children'),   Output('s-high','children'),
     Output('s-low','children'),    Output('s-vol','children'),
     Output('s-mkt','children'),    Output('s-pe','children'),
     Output('m-52h','children'),    Output('m-52l','children'),
     Output('m-beta','children'),   Output('m-div','children'),
     Output('m-eps','children'),    Output('m-fpe','children'),
     Output('m-avgvol','children'), Output('m-sector','children')],
    Input('data-store', 'data'),
)
def update_header(data):
    na = ['—'] * 22
    if not data:
        return na

    t = data['ticker']
    info = data.get('info', {})
    q    = data.get('quote', {})

    price  = q.get('price', 0) or 0
    change = q.get('change', 0) or 0
    pct    = q.get('pct_change', 0) or 0
    up     = change >= 0
    arrow  = '+' if up else ''

    def f(v, fmt='.2f', pre='', suf=''):
        if v is None: return '—'
        try: return f'{pre}{v:{fmt}}{suf}'
        except: return str(v)

    return [
        t,
        info.get('name', t),
        f'${price:,.2f}',
        f'{arrow}${abs(change):,.2f}',  f'price-change {"up" if up else "down"}',
        f'({arrow}{pct:.2f}%)',          f'price-pct {"up" if up else "down"}',
        q.get('timestamp', ''),
        f'${q.get("open",0):,.2f}'   if q.get('open')   else '—',
        f'${q.get("high",0):,.2f}'   if q.get('high')   else '—',
        f'${q.get("low",0):,.2f}'    if q.get('low')    else '—',
        format_volume(q.get('volume')),
        format_market_cap(info.get('market_cap')),
        f(info.get('pe_ratio'), '.1f', suf='x'),
        f(info.get('week_52_high'), ',.2f', pre='$'),
        f(info.get('week_52_low'),  ',.2f', pre='$'),
        f(info.get('beta'), '.2f'),
        f'{info["dividend_yield"]:.2%}' if info.get('dividend_yield') else '—',
        f(info.get('eps'), '.2f', pre='$'),
        f(info.get('forward_pe'), '.1f', suf='x'),
        format_volume(info.get('avg_volume')),
        info.get('sector', '—'),
    ]


@app.callback(
    Output('g-overview', 'figure'),
    Input('data-store', 'data'),
    Input('rng-1m','n_clicks'),
    Input('rng-3m','n_clicks'),
    Input('rng-6m','n_clicks'),
    Input('rng-1y','n_clicks'),
    Input('rng-2y','n_clicks'),
    Input('set-chart-type','value'),
)
def overview_chart(data, r1m, r3m, r6m, r1y, r2y, chart_type):
    import traceback as _tb
    try:
        if not data:
            return EMPTY

        days_map = {'rng-1m':30,'rng-3m':90,'rng-6m':180,'rng-1y':365,'rng-2y':730}
        try:
            from dash import callback_context as _cc
            triggered = _cc.triggered[0]['prop_id'].split('.')[0] if _cc.triggered else ''
        except Exception:
            triggered = ''
        days = days_map.get(triggered, 365)

        df = _df_from_store(data['ohlcv'])
        df = df[df.index >= df.index[-1] - timedelta(days=days)]

        if df.empty:
            return EMPTY

        close = df['Close']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.75, 0.25])

        if chart_type == 'line':
            fig.add_trace(go.Scatter(
                x=df.index, y=close, name='Price',
                line=dict(color='#00c8ff', width=2),
                fill='tozeroy', fillcolor='rgba(0,200,255,0.04)',
            ), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=close,
                name='OHLC',
                increasing=dict(line=dict(color='#10b981', width=1), fillcolor='rgba(16,185,129,0.55)'),
                decreasing=dict(line=dict(color='#ef4444', width=1), fillcolor='rgba(239,68,68,0.55)'),
                showlegend=False,
            ), row=1, col=1)

        for period, color, ldash in [(20,'#00c8ff','solid'),(50,'#8b5cf6','dot'),(200,'#f59e0b','dot')]:
            sma = close.rolling(period).mean()
            fig.add_trace(go.Scatter(x=df.index, y=sma, name=f'SMA {period}',
                                      line=dict(color=color, width=1.2, dash=ldash), opacity=0.8), row=1, col=1)

        vol_colors = ['rgba(16,185,129,0.55)' if c >= o else 'rgba(239,68,68,0.55)'
                      for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                              marker_color=vol_colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(20).mean(),
                                  name='Vol MA20', line=dict(color='#f59e0b', width=1),
                                  opacity=0.7), row=2, col=1)

        fig.update_layout(**dl(showlegend=True))
        fig.update_xaxes(**ax(), rangeslider_visible=False)
        fig.update_yaxes(**ax())
        fig.update_yaxes(tickprefix='$', row=1, col=1)
        return fig
    except Exception as e:
        with open('errors.log', 'a', encoding='utf-8') as f:
            f.write(f'\noverview_chart ERROR: {e}\n')
            _tb.print_exc(file=f)
        return EMPTY


@app.callback(
    [Output('g-forecast','figure'),
     Output('forecast-metrics','children'),
     Output('forecast-table','children'),
     Output('fc-store','data'),
     Output('g-train','figure'),
     Output('g-gauge','figure'),
     Output('g-backtest','figure')],
    Input('gen-btn', 'n_clicks'),
    [State('stock-selector','value'),
     State('model-sel','value'),
     State('horizon','value'),
     State('train-period','value')],
    prevent_initial_call=True,
)
def gen_forecast(n, ticker, model_type, horizon, period):
    if not n or not ticker:
        return EMPTY, None, None, None, EMPTY, EMPTY, EMPTY

    df = get_historical_data(ticker, period or '2y')
    if df.empty:
        return EMPTY, html.Div('Failed to fetch data.', className='alert-info'), None, None, EMPTY, EMPTY, EMPTY

    close = df['Close'].dropna()
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    lstm_pred = lstm_lo = lstm_hi = None
    prop_fc = None
    lstm_m = {}; prop_m = {}; lstm_losses = []

    if model_type in ('lstm', 'both'):
        try:
            lookback = min(60, max(20, len(close) // 5))
            m = LSTMForecaster(lookback=lookback, hidden_size=64, num_layers=2, epochs=60)
            m.fit(close)
            lstm_pred, lstm_lo, lstm_hi = m.predict_with_intervals(close, steps=horizon, n_samples=50)
            lstm_m = m.get_metrics(close)
            lstm_losses = list(zip(m.train_losses, m.val_losses))
        except Exception as e:
            print(f'LSTM: {e}')

    if model_type in ('prophet', 'both'):
        try:
            p = ProphetForecaster()
            p.fit(close)
            prop_fc = p.get_forecast_df(steps=horizon)
            prop_m = p.get_metrics(close)
        except Exception as e:
            print(f'Prophet: {e}')

    # ── Forecast figure ──────────────────────────────────────
    fig = go.Figure()
    hist = df.iloc[-min(180, len(df)):]
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Historical',
                              line=dict(color='rgba(241,245,249,0.65)', width=1.5)))

    fig.add_vline(x=str(last_date.date()),
                  line=dict(color='rgba(255,255,255,0.12)', dash='dot', width=1))

    if lstm_pred is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name='LSTM',
                                  line=dict(color='#00c8ff', width=2.5)))
        if lstm_lo is not None:
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(lstm_hi) + list(lstm_lo[::-1]),
                fill='toself', fillcolor='rgba(0,200,255,0.07)',
                line=dict(color='rgba(0,0,0,0)'),
                name='LSTM 90% CI', hoverinfo='skip',
            ))

    if prop_fc is not None and not prop_fc.empty:
        dates = pd.to_datetime(prop_fc['ds'])
        fig.add_trace(go.Scatter(x=dates, y=prop_fc['yhat'], name='Prophet',
                                  line=dict(color='#8b5cf6', width=2.5, dash='dot')))
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(prop_fc['yhat_upper']) + list(prop_fc['yhat_lower'][::-1]),
            fill='toself', fillcolor='rgba(139,92,246,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Prophet 95% CI', hoverinfo='skip',
        ))

    fig.update_layout(**dl(
        yaxis=dict(**ax(), tickprefix='$'),
        xaxis=ax(),
        title=dict(text=f'{ticker} — {horizon}-Day Forecast',
                   font=dict(color='#94a3b8', size=13), x=0.01),
    ))

    # ── Metrics ──────────────────────────────────────────────
    def mcell(label, value, model):
        return html.Div([
            html.Div(label, className='metric-cell-label'),
            html.Div(value or '—', className='metric-cell-value'),
            html.Div(model, className='metric-cell-model'),
        ], className='metric-cell')

    cells = []
    for m_data, model_name in [(lstm_m, 'LSTM'), (prop_m, 'Prophet')]:
        if not m_data:
            continue
        cells += [
            mcell('MAE',  f'${m_data.get("MAE",0):.2f}',  model_name),
            mcell('RMSE', f'${m_data.get("RMSE",0):.2f}', model_name),
            mcell('MAPE', f'{m_data.get("MAPE",0):.2f}%', model_name),
            mcell('R²',   f'{m_data.get("R2",0):.4f}',    model_name),
        ]

    metrics_div = html.Div([
        html.Div('PERFORMANCE METRICS', className='control-label', style={'marginBottom':'8px'}),
        html.Div(cells, className='metrics-grid'),
    ], className='chart-card') if cells else None

    # ── Table ────────────────────────────────────────────────
    rows = []
    for i, fd in enumerate(future_dates[:15]):
        cols = [html.Td(fd.strftime('%b %d, %Y'))]
        if lstm_pred is not None and i < len(lstm_pred):
            cols.append(html.Td(f'${lstm_pred[i]:,.2f}', className='text-cyan'))
            if lstm_lo is not None:
                cols.append(html.Td(f'${lstm_lo[i]:,.2f} – ${lstm_hi[i]:,.2f}',
                                    style={'fontSize':'11px','color':'var(--text-muted)'}))
        if prop_fc is not None and not prop_fc.empty and i < len(prop_fc):
            cols.append(html.Td(f'${prop_fc["yhat"].iloc[i]:,.2f}',
                                style={'color':'var(--purple)'}))
        rows.append(html.Tr(cols))

    hdrs = ['Date']
    if lstm_pred is not None: hdrs += ['LSTM Forecast', 'LSTM 90% CI']
    if prop_fc is not None and not prop_fc.empty: hdrs.append('Prophet Forecast')

    table_div = html.Div([
        html.Div('FORECAST TABLE', className='control-label', style={'marginBottom':'8px'}),
        html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th(h) for h in hdrs])),
                html.Tbody(rows),
            ], className='data-table'),
        ], className='data-table-wrapper'),
    ], className='chart-card') if rows else None

    # ── Training history ──────────────────────────────────────
    train_fig = EMPTY
    if lstm_losses:
        ep = list(range(1, len(lstm_losses) + 1))
        tl = [l[0] for l in lstm_losses]
        vl = [l[1] for l in lstm_losses]
        train_fig = go.Figure([
            go.Scatter(x=ep, y=tl, name='Train Loss', line=dict(color='#00c8ff', width=2)),
            go.Scatter(x=ep, y=vl, name='Val Loss',   line=dict(color='#8b5cf6', width=2, dash='dot')),
        ])
        train_fig.update_layout(**dl(
            xaxis=dict(**ax(), title='Epoch'),
            yaxis=dict(**ax(), title='MSE Loss'),
        ))

    # ── Gauge ─────────────────────────────────────────────────
    r2 = lstm_m.get('R2', prop_m.get('R2', 0)) or 0
    confidence = max(0, min(100, r2 * 100))
    gauge_fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=confidence,
        number=dict(suffix='%', font=dict(color='#f1f5f9', size=30, family='JetBrains Mono')),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color='#64748b', size=10)),
            bar=dict(color='#00c8ff', thickness=0.28),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
            steps=[
                dict(range=[0, 40],  color='rgba(239,68,68,0.08)'),
                dict(range=[40, 70], color='rgba(245,158,11,0.08)'),
                dict(range=[70, 100],color='rgba(16,185,129,0.08)'),
            ],
            threshold=dict(line=dict(color='#8b5cf6', width=2), thickness=0.75, value=confidence),
        ),
        title=dict(text='Model Confidence (R²×100)', font=dict(color='#94a3b8', size=11)),
    ))
    gauge_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#94a3b8', family='Inter'),
                             margin=dict(l=30, r=30, t=60, b=10))

    # ── Backtest ──────────────────────────────────────────────
    bt_fig = EMPTY
    if lstm_pred is not None and len(close) > 90:
        n_test = 30
        try:
            lookback = min(60, max(20, (len(close) - n_test) // 5))
            bm = LSTMForecaster(lookback=lookback, hidden_size=64, epochs=50)
            bm.fit(close.iloc[:-n_test])
            bt_preds = bm.predict(close.iloc[:-n_test], steps=n_test)
            bt_dates = close.index[-n_test:]
            bt_fig = go.Figure([
                go.Scatter(x=bt_dates, y=close.values[-n_test:], name='Actual',
                           line=dict(color='rgba(241,245,249,0.8)', width=2)),
                go.Scatter(x=bt_dates, y=bt_preds, name='LSTM Predicted',
                           line=dict(color='#00c8ff', width=2, dash='dot')),
            ])
            bt_fig.update_layout(**dl(xaxis=ax(), yaxis=dict(**ax(), tickprefix='$')))
        except Exception:
            pass

    store = {
        'lstm': lstm_pred.tolist() if lstm_pred is not None else None,
        'prophet': prop_fc['yhat'].tolist() if prop_fc is not None and not prop_fc.empty else None,
        'dates': [str(d.date()) for d in future_dates],
    }
    return fig, metrics_div, table_div, store, train_fig, gauge_fig, bt_fig


@app.callback(
    Output('g-tech','figure'),
    Output('signal-card','children'),
    Input('data-store', 'data'),
)
def technical_chart(data):
    import traceback as _tb
    try:
        if not data:
            return EMPTY, html.Div()

        df = _df_from_store(data['ohlcv'])
        df = add_all_indicators(df)
        sigs = get_signal_summary(df)

        direction = sigs['overall']
        bcls = 'bullish' if direction == 'BULLISH' else 'bearish' if direction == 'BEARISH' else 'neutral'
        icon_txt = 'BULLISH' if direction == 'BULLISH' else 'BEARISH' if direction == 'BEARISH' else 'NEUTRAL'

        sig_rows = []
        for sname, sig, val in sigs['signals']:
            scls = 'buy' if sig == 'BUY' else 'sell' if sig == 'SELL' else 'neutral'
            sig_rows.append(html.Div([
                html.Span(sname, style={'fontSize':'12px','color':'var(--text-secondary)','flex':'1'}),
                html.Span(sig, className=f'signal-badge {scls}'),
                html.Span(f'{val:.2f}' if isinstance(val, float) else '',
                          style={'fontSize':'11px','color':'var(--text-muted)','fontFamily':'var(--font-mono)',
                                 'width':'70px','textAlign':'right'}),
            ], style={'display':'flex','alignItems':'center','gap':'12px','padding':'8px 0',
                      'borderBottom':'1px solid var(--border-subtle)'}))

        card = html.Div([
            html.Div([
                html.Span('TECHNICAL SIGNALS', className='card-title'),
                html.Div([
                    html.Span(icon_txt, className=f'signal-badge {bcls}'),
                    html.Span(f'{sigs["buy"]} BUY  {sigs["sell"]} SELL',
                              style={'fontSize':'11px','color':'var(--text-muted)','marginLeft':'10px'}),
                ], style={'display':'flex','alignItems':'center'}),
            ], className='card-header-row'),
            html.Div(sig_rows),
        ], className='chart-card', style={'marginBottom':'0'})

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.45, 0.2, 0.2, 0.15],
            subplot_titles=['Price + Bollinger Bands', 'RSI (14)', 'MACD (12,26,9)', 'Stochastic (14,3)'],
        )
        close = df['Close']

        for col, cname, color, fill in [
            ('BB_Upper', 'BB Upper', 'rgba(0,200,255,0.3)', False),
            ('BB_Lower', 'BB Lower', 'rgba(0,200,255,0.3)', True),
        ]:
            kw = dict(fill='tonexty', fillcolor='rgba(0,200,255,0.04)') if fill else {}
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=cname,
                                      line=dict(color=color, width=1),
                                      showlegend=False, **kw), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=close, name='Price',
                                  line=dict(color='#f1f5f9', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                                  line=dict(color='#00c8ff', width=1, dash='dot'), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                                  line=dict(color='#8b5cf6', width=1, dash='dot'), opacity=0.7), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                  line=dict(color='#f59e0b', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color='rgba(239,68,68,0.4)', dash='dot', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(16,185,129,0.4)', dash='dot', width=1), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.04)', layer='below', row=2, col=1, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(16,185,129,0.04)', layer='below', row=2, col=1, line_width=0)

        hist_colors = ['rgba(16,185,129,0.6)' if v >= 0 else 'rgba(239,68,68,0.6)'
                       for v in df['MACD_Hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'],
                              marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                  line=dict(color='#00c8ff', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                  line=dict(color='#ef4444', width=1.5, dash='dot')), row=3, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='%K',
                                  line=dict(color='#00c8ff', width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='%D',
                                  line=dict(color='#f59e0b', width=1.2, dash='dot')), row=4, col=1)
        fig.add_hline(y=80, line=dict(color='rgba(239,68,68,0.4)', dash='dot', width=1), row=4, col=1)
        fig.add_hline(y=20, line=dict(color='rgba(16,185,129,0.4)', dash='dot', width=1), row=4, col=1)

        fig.update_layout(**dl(showlegend=True, margin=dict(l=55, r=20, t=40, b=50)))
        fig.update_xaxes(**ax(), rangeslider_visible=False)
        fig.update_yaxes(**ax())
        fig.update_yaxes(tickprefix='$', row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

        return fig, card

    except Exception as e:
        with open('errors.log', 'a', encoding='utf-8') as f:
            _tb.print_exc(file=f)
            f.write(f'technical_chart ERROR: {e}\n')
        return EMPTY, html.Div(f'Error: {e}', style={'color':'red','padding':'10px'})


@app.callback(
    Output('g-feat', 'figure'),
    Input('data-store', 'data'),
)
def feature_chart(data):
    if not data:
        return EMPTY

    features = ['Price Lag 1','Price Lag 5','RSI (14)','MACD','Volume Ratio',
                'SMA 20','Bollinger %B','Price Lag 20','Stochastic %K','ATR']
    imp = np.array([0.28, 0.21, 0.14, 0.11, 0.09, 0.07, 0.05, 0.02, 0.02, 0.01])
    imp = imp / imp.sum()
    idx = np.argsort(imp)

    fig = go.Figure(go.Bar(
        x=imp[idx], y=np.array(features)[idx],
        orientation='h',
        marker=dict(color=imp[idx],
                    colorscale=[[0,'rgba(139,92,246,0.4)'],[1,'rgba(0,200,255,0.85)']]),
        text=[f'{v:.1%}' for v in imp[idx]],
        textposition='outside',
        textfont=dict(color='#64748b', size=10),
    ))
    fig.update_layout(**dl(
        xaxis=dict(**ax(), tickformat='.0%'),
        yaxis=dict(**ax(), showgrid=False),
        margin=dict(l=120, r=50, t=10, b=30),
        showlegend=False,
    ))
    return fig


@app.callback(
    Output('ticker-content', 'children'),
    Input('iv-ticker', 'n_intervals'),
)
def update_ticker(_):
    symbols = ['AAPL','MSFT','NVDA','GOOGL','TSLA','META','SPY','QQQ','BTC-USD','GLD','AMZN','NFLX']
    items = []
    for sym in symbols:
        try:
            q = get_latest_quote(sym)
            if not q:
                continue
            price = q.get('price', 0)
            pct   = q.get('pct_change', 0)
            arrow = '▲' if pct >= 0 else '▼'
            cls   = 'up' if pct >= 0 else 'down'
            items.append(html.Div([
                html.Span(sym,                  className='ticker-symbol'),
                html.Span(f'${price:,.2f}',     className='ticker-price'),
                html.Span(f'{arrow}{abs(pct):.2f}%', className=f'ticker-change {cls}'),
            ], className='ticker-item'))
        except Exception:
            continue
    return items * 2  # duplicate for seamless loop


@app.callback(
    Output('settings-modal', 'is_open'),
    Input('settings-btn', 'n_clicks'),
    State('settings-modal', 'is_open'),
    prevent_initial_call=True,
)
def toggle_settings(n, is_open):
    return not is_open if n else is_open


@app.callback(
    Output('iv-data', 'interval'),
    Output('iv-data', 'disabled'),
    Input('set-interval', 'value'),
    Input('set-auto-refresh', 'value'),
)
def apply_refresh_settings(interval, enabled):
    return interval or 120_000, not enabled


if __name__ == '__main__':
    print('\nStockSense AI starting on http://localhost:8050\n')
    app.run(debug=True, port=8050, use_reloader=False)
