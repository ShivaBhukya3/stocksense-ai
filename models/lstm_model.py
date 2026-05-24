import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _LSTMNet(object if not TORCH_AVAILABLE else __builtins__['object'] if isinstance(__builtins__, dict) else object):
    pass


if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


class LSTMForecaster:
    def __init__(self, lookback: int = 60, hidden_size: int = 64, num_layers: int = 2, epochs: int = 80):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        self.train_losses = []
        self.val_losses = []

    def _make_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i, 0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, prices: pd.Series, verbose: bool = False):
        scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))
        X, y = self._make_sequences(scaled)

        if not TORCH_AVAILABLE:
            self._fit_fallback(X, y)
            return self

        split = int(len(X) * 0.9)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.FloatTensor(y_train).unsqueeze(1)
        X_v = torch.FloatTensor(X_val)
        y_v = torch.FloatTensor(y_val).unsqueeze(1)

        self.model = _LSTMNet(1, self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_tr)
            loss = criterion(out, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_v), y_v).item()

            scheduler.step(val_loss)
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train: {loss.item():.6f} | Val: {val_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        return self

    def _fit_fallback(self, X, y):
        from sklearn.neural_network import MLPRegressor
        X_flat = X.reshape(X.shape[0], -1)
        self._fallback = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        self._fallback.fit(X_flat, y)
        self.is_trained = True

    def predict(self, prices: pd.Series, steps: int = 30) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        scaled = self.scaler.transform(prices.values.reshape(-1, 1))
        seq = scaled[-self.lookback:].tolist()
        predictions = []

        if not TORCH_AVAILABLE:
            for _ in range(steps):
                x = np.array(seq[-self.lookback:]).reshape(1, -1)
                pred = self._fallback.predict(x)[0]
                predictions.append(pred)
                seq.append([pred])
        else:
            self.model.eval()
            with torch.no_grad():
                for _ in range(steps):
                    x = torch.FloatTensor(seq[-self.lookback:]).unsqueeze(0)
                    pred = self.model(x).item()
                    predictions.append(pred)
                    seq.append([pred])

        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def predict_with_intervals(self, prices: pd.Series, steps: int = 30,
                               n_samples: int = 50) -> tuple:
        base = self.predict(prices, steps)

        if not TORCH_AVAILABLE or self.model is None:
            noise_std = prices.pct_change().std() * prices.iloc[-1]
            samples = np.array([base + np.random.normal(0, noise_std * np.sqrt(np.arange(1, steps + 1) / 5), steps)
                                 for _ in range(n_samples)])
        else:
            self.model.train()
            samples = []
            scaled = self.scaler.transform(prices.values.reshape(-1, 1))
            with torch.no_grad():
                for _ in range(n_samples):
                    seq = scaled[-self.lookback:].tolist()
                    preds = []
                    for _ in range(steps):
                        x = torch.FloatTensor(seq[-self.lookback:]).unsqueeze(0)
                        pred = self.model(x).item()
                        preds.append(pred)
                        seq.append([pred])
                    inv = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                    samples.append(inv)
            self.model.eval()
            samples = np.array(samples)

        lower = np.percentile(samples, 5, axis=0)
        upper = np.percentile(samples, 95, axis=0)
        return base, lower, upper

    def get_metrics(self, prices: pd.Series, test_size: int = 30) -> dict:
        if not self.is_trained:
            return {}
        scaled = self.scaler.transform(prices.values.reshape(-1, 1))
        X, y = self._make_sequences(scaled)
        X_test = X[-test_size:]
        y_test_scaled = y[-test_size:]

        if not TORCH_AVAILABLE:
            X_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_scaled = self._fallback.predict(X_flat)
        else:
            self.model.eval()
            with torch.no_grad():
                y_pred_scaled = self.model(torch.FloatTensor(X_test)).numpy().flatten()

        y_actual = self.scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mae = float(np.mean(np.abs(y_actual - y_pred)))
        rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_actual - y_pred) / np.where(y_actual == 0, 1, y_actual))) * 100)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}
