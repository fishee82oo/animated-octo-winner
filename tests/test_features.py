import numpy as np
import pandas as pd
from data.simulator import simulate
from features.build_features import build_features


def test_fixed_origin_lags_and_rolling():
    df = simulate(years=0.1)
    df["price"] = np.arange(len(df), dtype=float)
    x = build_features(df)
    i = 240
    assert x.iloc[i].price_lag_1 == 239
    assert x.iloc[i + 23].price_lag_1 == 239
    assert x.iloc[i + 23].price_lag_24 == 239
    assert x.iloc[i].price_lag_168 == 72
    assert x.iloc[i + 23].price_roll_24_mean == np.arange(216, 240).mean()
    assert np.isclose(x.iloc[i].price_roll_168_std, np.arange(72, 240).std(ddof=1))


def test_no_target_leakage_for_entire_day():
    df = simulate(years=0.1)
    a = build_features(df)
    df.loc[240:, "price"] = 999999
    b = build_features(df)
    pd.testing.assert_frame_equal(a.iloc[240:264], b.iloc[240:264])
    pd.testing.assert_frame_equal(a.iloc[:240], b.iloc[:240])


def test_observed_future_coal_does_not_leak():
    df = simulate(years=0.1)
    a = build_features(df)
    df.loc[240:, "coal_index"] *= 2
    b = build_features(df)
    pd.testing.assert_frame_equal(a.iloc[240:264], b.iloc[240:264])


def test_prefix_stability_and_forecast_contract():
    df = simulate(years=0.1)
    a = build_features(df)
    b = build_features(df.iloc[:300])
    pd.testing.assert_frame_equal(a.iloc[:300], b)
    assert not a.filter(like="_forecast").isna().any().any()
