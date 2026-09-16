"""Educational CfD scenarios in EUR. No execution or trading interfaces.

P10/P50/P90 do not identify a distribution: use explicitly assumed bounded
piecewise-linear inverse CDF tails and a Gaussian temporal copula.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr


def settlement(spot, volume, contract_price, hedge_ratio):
    """Buyer cost / seller gross revenue: PV + (C-P)H, H=hV.

    Volume is MWh per interval; a fixed metered profile is assumed. This is
    equivalent to CH + P(V-H). Generation costs/fees are excluded.
    """
    p, v = np.asarray(spot, float), np.asarray(volume, float)
    if (
        not np.isfinite(p).all()
        or not np.isfinite(v).all()
        or (v < 0).any()
        or not np.isfinite(contract_price)
        or not 0 <= hedge_ratio <= 1
    ):
        raise ValueError("Invalid settlement inputs")
    return v * (hedge_ratio * contract_price + (1 - hedge_ratio) * p)


def price_scenarios(
    quantiles, lower_bound, upper_bound, n_scenarios=5000, correlation=0.65, seed=42
):
    q = np.asarray(quantiles, float)
    if (
        q.ndim != 2
        or q.shape[1] != 3
        or len(q) == 0
        or not np.isfinite(q).all()
        or (np.diff(q, axis=1) < 0).any()
    ):
        raise ValueError("Expected finite ordered (hours,3) quantiles")
    if (
        not np.isfinite([lower_bound, upper_bound]).all()
        or lower_bound >= upper_bound
        or not 0 <= correlation <= 1
        or n_scenarios < 2
    ):
        raise ValueError("Invalid scenario parameters")
    # Physical price bounds also constrain extrapolated quantiles.
    q = np.clip(q, lower_bound, upper_bound)
    rng = np.random.default_rng(seed)
    z = np.sqrt(correlation) * rng.normal(size=(n_scenarios, 1)) + np.sqrt(
        1 - correlation
    ) * rng.normal(size=(n_scenarios, len(q)))
    u = ndtr(z)
    return np.column_stack(
        [
            np.interp(u[:, j], [0, 0.1, 0.5, 0.9, 1], [lower_bound, *q[j], upper_bound])
            for j in range(len(q))
        ]
    )


def tail_risk(losses, confidence=0.95):
    a = np.sort(np.asarray(losses, float))
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or not 0 < confidence < 1:
        raise ValueError("Invalid loss distribution")
    # Empirical inverse CDF VaR and exact worst (1-alpha) probability mass.
    n = len(a)
    cut = confidence * n
    weights = np.maximum(0, np.arange(1, n + 1) - np.maximum(np.arange(n), cut))
    return float(a[max(0, int(np.ceil(cut)) - 1)]), float(
        np.dot(a, weights) / weights.sum()
    )


def hedge_analysis(
    scenarios, volume, contract_price, ratios=None, side="buyer", confidence=0.95
):
    p = np.asarray(scenarios, float)
    v = np.asarray(volume, float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] == 0 or not np.isfinite(p).all():
        raise ValueError("Expected scenarios x hours")
    if (
        v.ndim > 1
        or (v.ndim == 1 and len(v) != p.shape[1])
        or not np.isfinite(v).all()
        or (v < 0).any()
    ):
        raise ValueError("Invalid MWh volume profile")
    if side not in ["buyer", "seller"]:
        raise ValueError("side must be buyer or seller")
    ratios = np.linspace(0, 1, 21) if ratios is None else np.asarray(ratios, float)
    rows = []
    for h in ratios:
        cash = settlement(p, v, contract_price, float(h)).sum(axis=1)
        exposure = (p * v * (1 - h)).sum(axis=1)
        loss = cash if side == "buyer" else -cash
        spot_loss = exposure if side == "buyer" else -exposure
        var, cvar = tail_risk(loss, confidence)
        sv, sc = tail_risk(spot_loss, confidence)
        rows.append(
            dict(
                hedge_ratio=h,
                expected_settlement=cash.mean(),
                expected_loss=loss.mean(),
                settlement_std=cash.std(),
                loss_var=var,
                loss_cvar=cvar,
                spot_loss_var=sv,
                spot_loss_cvar=sc,
            )
        )
    if not rows:
        raise ValueError("Provide at least one hedge ratio")
    return pd.DataFrame(rows)


def suggest_hedge(table, risk_tolerance=0.5):
    """Minimize expected loss + aversion * (CVaR - expected loss).

    tolerance=1: expected value only; tolerance=0: strong tail aversion.
    This is a scenario-dependent educational heuristic, not an optimal policy.
    """
    if not 0 <= risk_tolerance <= 1:
        raise ValueError("Risk tolerance must be [0,1]")
    score = table.expected_loss + 4 * (1 - risk_tolerance) * (
        table.loss_cvar - table.expected_loss
    )
    return float(table.loc[score.idxmin(), "hedge_ratio"])
