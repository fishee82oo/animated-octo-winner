import numpy as np
from data.simulator import simulate
from models.gbm import GBM
from backtest.walk_forward import walk_forward


def test_point_backtest():
    p, metrics, _ = walk_forward(
        simulate(years=0.15), lambda: GBM(n_estimators=20), days=2
    )
    assert len(p) == 48 and np.isfinite(p.prediction).all()
    assert metrics.iloc[0].mae >= 0
