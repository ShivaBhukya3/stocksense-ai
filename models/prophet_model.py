import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecaster:
    def __init__(self, changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 interval_width: float = 0.95):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.model = None
        self.forecast = None
        self.is_fitted = False

    def _to_prophet_df(self, prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'ds': prices.index, 'y': prices.values})
        df['ds'] = pd.to_datetime(df['ds'])
        if df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)
        return df.dropna()

    def fit(self, prices: pd.Series):
        if not PROPHET_AVAILABLE:
            self._fit_fallback(prices)
            return self

        df = self._to_prophet_df(prices)
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        try:
            self.model.add_country_holidays(country_name='US')
        except Exception:
            pass

        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

        self.model.fit(df)
        self.is_fitted = True
        return self

    def _fit_fallback(self, prices: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self._fallback_model = ExponentialSmoothing(
            prices.values, trend='add', seasonal=None
        ).fit(optimized=True)
        self._prices = prices
        self.is_fitted = True

    def predict(self, steps: int = 30) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if not PROPHET_AVAILABLE:
            return self._predict_fallback(steps)

        future = self.model.make_future_dataframe(periods=steps, freq='B')
        self.forecast = self.model.predict(future)
        return self.forecast

    def _predict_fallback(self, steps: int) -> pd.DataFrame:
        preds = self._fallback_model.forecast(steps)
        last_date = self._prices.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=steps + 1, freq='B')[1:]

        std = float(self._prices.pct_change().std() * self._prices.iloc[-1])
        ci_factor = np.sqrt(np.arange(1, steps + 1))

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': preds,
            'yhat_lower': preds - 1.96 * std * ci_factor,
            'yhat_upper': preds + 1.96 * std * ci_factor,
            'trend': preds,
        })

    def get_forecast_df(self, steps: int = 30) -> pd.DataFrame:
        fc = self.predict(steps)
        if fc is None or fc.empty:
            return pd.DataFrame()
        hist_len = len(self.model.history) if (PROPHET_AVAILABLE and self.model) else 0
        return fc.iloc[hist_len:].reset_index(drop=True)

    def get_components(self) -> dict:
        if not PROPHET_AVAILABLE or self.forecast is None:
            return {}
        components = {}
        for col in ['trend', 'yearly', 'weekly', 'holidays']:
            if col in self.forecast.columns:
                components[col] = self.forecast[col]
        return components

    def get_metrics(self, prices: pd.Series, test_size: int = 30) -> dict:
        if not self.is_fitted:
            return {}
        try:
            if PROPHET_AVAILABLE and self.model:
                hist = self.model.history.copy()
                hist_len = len(hist)
                future = self.model.make_future_dataframe(periods=0, freq='B')
                fc = self.model.predict(future)
                y_pred = fc['yhat'].values[-test_size:]
                y_actual = prices.values[-test_size:]
            else:
                fitted = self._fallback_model.fittedvalues[-test_size:]
                y_pred = fitted
                y_actual = prices.values[-test_size:]

            min_len = min(len(y_actual), len(y_pred))
            y_actual = y_actual[-min_len:]
            y_pred = y_pred[-min_len:]

            mae = float(np.mean(np.abs(y_actual - y_pred)))
            rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
            mape = float(np.mean(np.abs((y_actual - y_pred) / np.where(y_actual == 0, 1, y_actual))) * 100)
            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

            return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}
        except Exception:
            return {}
