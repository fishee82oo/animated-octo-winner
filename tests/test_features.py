import numpy as np
import pandas as pd
import pytest
from features.build_features import build_features, delivery_clock, issue_time


@pytest.mark.parametrize("date", ["2024-03-31", "2024-10-27"])
def test_no_target_day_or_future_price_leakage(hourly, date):
    local = delivery_clock(hourly.timestamp)
    mask = local.date == pd.Timestamp(date).date()
    a = build_features(hourly)
    changed = hourly.copy()
    changed.loc[local.date >= pd.Timestamp(date).date(), "price"] = 999999
    b = build_features(changed)
    pd.testing.assert_frame_equal(a.loc[mask], b.loc[mask])
    assert a.loc[mask].notna().all().all()


def test_same_wall_hour_lag_and_rolling(hourly):
    hourly["price"] = np.arange(len(hourly), dtype=float)
    x = build_features(hourly)
    i = 240
    assert x.iloc[i].price_lag_day == i - 24
    assert x.iloc[i + 23].price_last_known == 239
    assert x.iloc[i].price_roll_24_mean == np.arange(216, 240).mean()
    assert x.iloc[i].price_roll_168_std == np.arange(72, 240).std(ddof=1)


def test_price_only_excludes_driver_columns(hourly):
    a = build_features(hourly, False)
    changed = hourly.copy()
    changed.load_forecast_mw *= 100
    pd.testing.assert_frame_equal(a, build_features(changed, False))
    assert not any("forecast" in c for c in a)


def test_issue_clock_handles_dst():
    for date in ["2024-04-01", "2024-10-28"]:
        target = pd.Timestamp(date, tz="Europe/Brussels")
        origin = issue_time(target).tz_convert("Europe/Brussels")
        assert (
            origin.hour == 11
            and origin.date()
            == (target.tz_localize(None) - pd.Timedelta(days=1)).date()
        )
