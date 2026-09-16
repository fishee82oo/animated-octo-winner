import numpy as np
import pytest
from analytics.hedging import (
    settlement,
    price_scenarios,
    tail_risk,
    hedge_analysis,
    suggest_hedge,
)


def test_hand_settlement_and_cfd_identity():
    p = np.array([100, 200])
    v = np.array([2, 3])
    h = 0.8
    c = 150
    np.testing.assert_allclose(settlement(p, v, c, h), p * v + (c - p) * h * v)
    np.testing.assert_allclose(settlement(p, v, c, 0), [200, 600])
    np.testing.assert_allclose(settlement(p, v, c, 1), [300, 450])
    assert settlement(-50, 2, 100, 0.5) == 50


def test_tail_mass_with_ties_and_fractional_boundary():
    assert tail_risk([1, 2, 3, 4], 0.5) == (2, 3.5)
    assert tail_risk([0, 0, 0, 100], 0.5) == (0, 50)
    assert tail_risk([1, 2, 3, 4], 0.625) == (3, (3 * 0.5 + 4) / 1.5)


def test_full_hedge_risk_and_seller_sign():
    p = np.array([[100, 200], [200, 300], [300, 400]])
    b = hedge_analysis(p, [2, 3], 150, ratios=[0, 1])
    s = hedge_analysis(p, [2, 3], 150, ratios=[0, 1], side="seller")
    assert b.iloc[1].expected_settlement == 750 and b.iloc[1].settlement_std == 0
    assert b.iloc[1].spot_loss_cvar == 0
    assert s.iloc[1].loss_cvar == -750
    assert b.iloc[0].expected_loss == -s.iloc[0].expected_loss


def test_scenarios_bounds_reproducibility_and_heuristic():
    q = np.array([[50, 100, 150], [60, 110, 160]])
    a = price_scenarios(q, -50, 300)
    b = price_scenarios(q, -50, 300)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= -50 and a.max() <= 300
    np.testing.assert_allclose(np.quantile(a, [0.1, 0.5, 0.9], axis=0).T, q, atol=7)
    t = hedge_analysis(a, 1, 100)
    assert 0 <= suggest_hedge(t, 0) <= 1
    with pytest.raises(ValueError):
        price_scenarios([[2, 1, 3]], 0, 10)
    with pytest.raises(ValueError):
        settlement(10, -1, 10, 0.5)
