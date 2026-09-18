"""Deterministic toy fixtures for unit tests; never used by the dashboard."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly():
    t = pd.date_range("2023-12-31 23:00", "2024-12-31 22:00", freq="h", tz="UTC")
    i = np.arange(len(t))
    return pd.DataFrame(
        dict(
            timestamp=t,
            price=60 + 30 * np.sin(i / 24) + 10 * np.cos(i / 7),
            load_forecast_mw=1000 + 100 * np.sin(i / 24),
            solar_forecast_mw=np.maximum(0, 100 * np.sin((i % 24 - 6) * np.pi / 12)),
            wind_forecast_mw=100 + 30 * np.cos(i / 13),
        )
    )


@pytest.fixture
def weather_hourly(hourly):
    from data.weather import WEATHER_COLUMNS
    frame = pd.DataFrame({"timestamp": hourly.timestamp})
    frame["reference_time_bound"] = frame.timestamp - pd.Timedelta(hours=48)
    frame["assumed_available_at"] = frame.timestamp - pd.Timedelta(hours=40)
    for name, value in zip(WEATHER_COLUMNS, [12, 5, 50, 100, 0.2, 4, 6, 0]):
        frame[name] = float(value)
    return frame
