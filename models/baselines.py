import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


class Naive:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.price_lag_1.to_numpy()


class SeasonalNaive:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.price_lag_168.to_numpy()


class Sarimax:
    """Compact ARX baseline; daily shape supplied by exogenous features.

    Nonconvergence is surfaced through diagnostics, never silently hidden.
    """

    columns = [
        "hour",
        "day_of_week",
        "load_mw_forecast",
        "wind_mw_forecast",
        "solar_mw_forecast",
        "coal_index",
        "tou_peak",
        "tou_sharp_peak",
        "tou_valley",
    ]

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X[self.columns])
        exog = self.scaler.transform(X[self.columns])
        self.center = float(np.mean(y))
        self.scale = max(float(np.std(y)), 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.result = SM_SARIMAX(
                (np.asarray(y) - self.center) / self.scale,
                exog=exog,
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
