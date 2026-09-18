"""Synthetic fixtures only; offline tests never query the provider."""
import copy
import numpy as np
import pandas as pd
import pytest
from data.weather import (
    aggregate_response, validate_weather, weather_settings, WEATHER_COLUMNS,
    download_weather, WeatherConnector,
)
from features.build_features import build_features, weather_features
from backtest.walk_forward import walk_forward
from models.gbm import GBM


def response(times, temperature=10):
    cfg = weather_settings()
    values = [temperature, 5, 50, 100, 0.2]
    units = {f"{s}_previous_day2": u for s, (_, u) in cfg["variables"].items()}
    hourly = {f"{s}_previous_day2": [v] * len(times) for s, v in zip(cfg["variables"], values)}
    hourly["time"] = [t.strftime("%Y-%m-%dT%H:%M") for t in times]
    return dict(utc_offset_seconds=0, hourly_units=units, hourly=hourly)


def test_aggregation_units_and_degree_features():
    times = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    a, b = response(times, 10), response(times, 30)
    frame = aggregate_response([a, b], [["a", 0, 0], ["b", 1, 1]])
    assert frame.weather_temperature_c.tolist() == [20, 20]
    assert frame.weather_heating_degree_c.tolist() == [4, 4]
    assert frame.weather_cooling_degree_c.tolist() == [4, 4]
    assert frame.weather_temperature_spread_c.tolist() == [20, 20]
    a["hourly_units"]["wind_speed_100m_previous_day2"] = "km/h"
    with pytest.raises(ValueError, match="unit"):
        aggregate_response([a, b], [["a", 0, 0], ["b", 1, 1]])


def test_one_missing_site_not_silently_ignored():
    times = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    a, b = response(times), response(times)
    b["hourly"]["temperature_2m_previous_day2"][0] = None
    out = aggregate_response([a, b], [["a", 0, 0], ["b", 1, 1]])
    assert pd.isna(out.weather_temperature_c.iloc[0])
    with pytest.raises(ValueError, match="nonfinite"):
        validate_weather(out)


@pytest.mark.parametrize("date,hours", [("2024-03-31", 23), ("2024-10-27", 25)])
def test_weather_dst_and_future_price_isolation(hourly, weather_hourly, date, hours):
    local = hourly.timestamp.dt.tz_convert("Europe/Brussels")
    mask = (local.dt.date == pd.Timestamp(date).date()).to_numpy()
    x = build_features(hourly, False, weather_hourly)
    changed = hourly.copy()
    changed.loc[local.dt.date >= pd.Timestamp(date).date(), "price"] = 999999
    y = build_features(changed, False, weather_hourly)
    pd.testing.assert_frame_equal(x.loc[mask], y.loc[mask])
    assert len(x.loc[mask]) == hours and x.loc[mask, WEATHER_COLUMNS].notna().all().all()


def test_reject_short_lead_and_changed_timing(hourly, weather_hourly):
    bad = weather_hourly.copy()
    bad["reference_time_bound"] = bad.timestamp - pd.Timedelta(hours=24)
    with pytest.raises(ValueError, match="lead"):
        weather_features(hourly.timestamp, bad)
    bad = weather_hourly.copy()
    bad["assumed_available_at"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="allowance"):
        weather_features(hourly.timestamp, bad)


def test_paired_ablation_uses_identical_rows(hourly, weather_hourly):
    weather = weather_hourly.drop(index=range(500, 524))
    factory = lambda: GBM(n_estimators=5)
    a, _, da = walk_forward(hourly, factory, days=2, end_date="2024-03-31", weather=weather, use_weather=False)
    b, _, db = walk_forward(hourly, factory, days=2, end_date="2024-03-31", weather=weather, use_weather=True)
    pd.testing.assert_series_equal(a.timestamp, b.timestamp)
    for left, right in zip(da, db):
        assert left["train_rows"] == right["train_rows"]
        assert left["train_start"] == right["train_start"]
        assert right["feature_count"] - left["feature_count"] == len(WEATHER_COLUMNS)


def test_download_manifest_and_corruption(tmp_path):
    def fetcher(params):
        t = pd.date_range(params["start_date"], periods=48, freq="h", tz="UTC")
        return response(t), {"url": "offline-fixture"}
    frame, manifest = download_weather("BE", "2025-01-01", "2025-01-02", tmp_path, fetcher)
    assert manifest["rows"] == 48 and manifest["missing_hours"] == 0
    connector = WeatherConnector("BE", tmp_path)
    assert len(connector.load()) == 48
    path = tmp_path / "BE.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        connector.load()


def test_all_market_profiles_present():
    from data.connectors import markets
    assert set(weather_settings()["zones"]) == set(markets())
