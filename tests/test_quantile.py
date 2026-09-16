import numpy as np
from data.simulator import simulate
from backtest.walk_forward import walk_forward
from backtest.metrics import pinball_loss, spike_capture
from models.quantile import QuantileGBM


def test_pinball_and_spikes():
    assert pinball_loss([10, 20], [12, 16], 0.1) == 1.1
    m = spike_capture([1, 9, 10], [8, 9, 1], 7)
    assert m["spike_precision"] == 0.5 and m["spike_recall"] == 0.5
    assert np.isnan(spike_capture([1], [1], 7)["spike_recall"])


def test_quantile_pipeline_and_train_only_threshold():
    df = simulate(years=0.15)
    p, m, d = walk_forward(df, lambda: QuantileGBM(n_estimators=20), days=1)
    assert (p.p10 <= p.p50).all() and (p.p50 <= p.p90).all()
    assert m.iloc[0].pinball_p50 >= 0
    changed = df.copy()
    changed.loc[changed.timestamp >= p.origin.iloc[0], "price"] = 1e6
    p2, _, _ = walk_forward(changed, lambda: QuantileGBM(n_estimators=20), days=1)
    np.testing.assert_allclose(
        p[["p10", "p50", "p90", "spike_threshold"]],
        p2[["p10", "p50", "p90", "spike_threshold"]],
    )
