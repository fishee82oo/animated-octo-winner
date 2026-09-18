import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


class Naive:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.price_last_known.to_numpy()


class SeasonalNaive:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.price_lag_week.to_numpy()


class Sarimax:
    def fit(self, X, y):
        self.columns = [
            c
            for c in [
                "hour_sin",
                "hour_cos",
                "day_of_week",
                "load_forecast_mw",
                "solar_forecast_mw",
                "wind_forecast_mw",
                "weather_temperature_c",
                "weather_wind_100m_ms",
                "weather_radiation_wm2",
            ]
            if c in X
        ]
        self.scaler = StandardScaler().fit(X[self.columns])
        self.center = float(np.mean(y))
        self.scale = max(float(np.std(y)), 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.result = SM_SARIMAX(
                (np.asarray(y) - self.center) / self.scale,
                exog=self.scaler.transform(X[self.columns]),
                order=(1, 0, 0),
                trend="c",
                enforce_stationarity=False,
            ).fit(disp=False, maxiter=150)
        self.diagnostics = {
            "converged": bool(self.result.mle_retvals.get("converged", False)),
            "warnings": [str(w.message) for w in caught],
        }
        return self

    def predict(self, X):
        return (
            np.asarray(
                self.result.forecast(
                    len(X), exog=self.scaler.transform(X[self.columns])
                )
            )
            * self.scale
            + self.center
        )
