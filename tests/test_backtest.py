import numpy as np
import pandas as pd
import pytest
from backtest.walk_forward import walk_forward
from backtest.metrics import point_metrics, pinball_loss, spike_capture
from models.baselines import Naive, SeasonalNaive, Sarimax
from models.gbm import GBM
from models.quantile import QuantileGBM


@pytest.mark.parametrize("date,hours", [("2024-03-31", 23), ("2024-10-27", 25)])
def test_dst_horizons_and_training_labels(hourly, date, hours):
    target = pd.Timestamp(date, tz="Europe/Brussels").tz_convert("UTC")

    class Spy(Naive):
        def fit(self, X, y):
            assert X.index.max() < target
            return self

    p, m, d = walk_forward(hourly, Spy, days=1, end_date=date)
    assert len(p) == hours and d[0]["forecast_hours"] == hours
    assert (p.origin < p.timestamp).all()


def test_quantile_forecasts_and_threshold_do_not_use_target_labels(hourly):
    factory = lambda: QuantileGBM(n_estimators=20)
    p, _, _ = walk_forward(hourly, factory, days=1, end_date="2024-06-01")
    changed = hourly.copy()
    changed.loc[changed.timestamp >= p.timestamp.min(), "price"] = 99999
    p2, _, _ = walk_forward(changed, factory, days=1, end_date="2024-06-01")
    np.testing.assert_allclose(
        p[["p10", "p50", "p90", "spike_threshold"]],
        p2[["p10", "p50", "p90", "spike_threshold"]],
    )
    assert (p.p10 <= p.p50).all() and (p.p50 <= p.p90).all()


@pytest.mark.parametrize(
    "factory",
    [
        Naive,
        SeasonalNaive,
        Sarimax,
        lambda: GBM(n_estimators=20),
        lambda: QuantileGBM(n_estimators=20),
    ],
)
def test_all_models_end_to_end(hourly, factory):
    p, m, d = walk_forward(hourly.iloc[:1500], factory, days=1, use_exogenous=False)
    assert np.isfinite(p.prediction).all() and m.iloc[0].mae >= 0


def test_metrics_hand_cases():
    assert point_metrics([0, 2], [1, 3]) == dict(mae=1.0, rmse=1.0, wape=1.0)
    assert np.isnan(point_metrics([0], [1])["wape"])
    assert pinball_loss([10, 20], [12, 16], 0.1) == 1.1
    m = spike_capture([1, 9, 10], [8, 9, 1], 7)
    assert m["spike_precision"] == m["spike_recall"] == 0.5


def test_short_history_fails(hourly):
    with pytest.raises(ValueError):
        walk_forward(hourly.iloc[:100], Naive)


def test_minimum_training_window_counts_local_days(hourly):
    p, _, diagnostics = walk_forward(hourly, Naive, days=1,
        min_train_days=30, window_days=30, end_date="2024-04-01")
    assert p.delivery_date.iloc[0] == "2024-04-01"
    assert diagnostics[0]["train_rows"] == 719
