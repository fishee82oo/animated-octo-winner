import numpy as np
from models.gbm import GBM


class QuantileGBM:
    """Independent quantile regressors with monotone rearrangement."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.models = [
            GBM(objective="quantile", alpha=q, **self.params).fit(X, y)
            for q in [0.1, 0.5, 0.9]
        ]
        return self

    def predict(self, X):
        return np.sort(np.column_stack([m.predict(X) for m in self.models]), axis=1)
