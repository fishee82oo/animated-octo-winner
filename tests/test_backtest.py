import numpy as np
import pytest
from data.simulator import simulate
from models.baselines import Naive, SeasonalNaive, Sarimax
from backtest.walk_forward import walk_forward
from backtest.metrics import point_metrics


def test_metrics_hand_calculated():
    m = point_metrics([0, 2], [1, 3])
    assert m == dict(mae=1.0, rmse=1.0, wape=1.0)
    assert np.isnan(point_metrics([0], [1])["wape"])


def test_origins_and_training_boundary():
    df = simulate(years=0.2)

    class Spy(Naive):
        def fit(self, X, y):
            self.last = X.index.max()
            return self

        def predict(self, X):
            assert self.last < X.index.min()
            return super().predict(X)

    p, m, d = walk_forward(df, Spy, days=2)
    assert len(p) == 48 and len(m) == 5
    assert p.groupby("origin").prediction.nunique().eq(1).all()
    assert d[1]["train_rows"] == d[0]["train_rows"] + 24


def test_sarimax_interface():
    p, _, d = walk_forward(
        simulate(years=0.12), Sarimax, days=1, min_train_days=20, window_days=25
    )
    assert len(p) == 24 and np.isfinite(p.prediction).all()
    assert "converged" in d[0]


def test_insufficient_history():
    with pytest.raises(ValueError):
        walk_forward(simulate(years=0.02), SeasonalNaive)
