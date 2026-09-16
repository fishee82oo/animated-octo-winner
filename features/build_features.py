"""Delivery-day features with explicit European clock and auction information set."""

import numpy as np
import pandas as pd
from data.connectors import validate, settings


def delivery_clock(index):
    return pd.DatetimeIndex(index).tz_convert(settings()["market_timezone"])


def issue_time(delivery_start):
    """Research origin: D-1 11:00 Brussels, before the usual noon auction."""
    wall = (
        pd.Timestamp(delivery_start)
        .tz_convert(settings()["market_timezone"])
        .tz_localize(None)
        .normalize()
    )
    return (
        (wall - pd.Timedelta(days=1) + pd.Timedelta(hours=11))
        .tz_localize(settings()["market_timezone"])
        .tz_convert("UTC")
    )


def delivery_period(index):
    local = delivery_clock(index)
    return np.select(
        [local.dayofweek >= 5, (local.hour >= 8) & (local.hour < 20)],
        ["weekend", "weekday_peak"],
        default="weekday_offpeak",
    )


def build_features(frame, use_exogenous=True):
    df = validate(frame).set_index("timestamp")
    idx = df.index
    local = delivery_clock(idx)
    start = local.normalize().tz_convert("UTC")
    wall = local.tz_localize(None)
    x = pd.DataFrame(index=idx)
    x["hour"] = local.hour
    x["day_of_week"] = local.dayofweek
    x["month"] = local.month
    x["utc_offset_hours"] = [t.utcoffset().total_seconds() / 3600 for t in local]
    x["hour_sin"] = np.sin(local.hour * 2 * np.pi / 24)
    x["hour_cos"] = np.cos(local.hour * 2 * np.pi / 24)
    x["price_last_known"] = df.price.reindex(start - pd.Timedelta(hours=1)).to_numpy()
    # D-1 delivery prices are already known from the preceding day-ahead auction,
    # including D-1 evening hours beyond the D-1 11:00 forecast issue clock.
    historical_wall = pd.Series(df.price.to_numpy(), index=wall).groupby(level=0).mean()
    daily = (
        pd.Series(df.price.to_numpy(), index=wall.normalize()).groupby(level=0).mean()
    )
    for days, name in [(1, "day"), (7, "week")]:
        keys = wall - pd.Timedelta(days=days)
        values = historical_wall.reindex(keys).to_numpy()
        # Spring missing wall hour: previous reference day's mean. Fall duplicate
        # reference hours: their mean. Both use only the earlier delivery day.
        fallback = daily.reindex(keys.normalize()).to_numpy()
        x[f"price_lag_{name}"] = np.where(np.isnan(values), fallback, values)
    for window in [24, 168]:
        for stat in ["mean", "std"]:
            series = getattr(df.price.rolling(window), stat)()
            x[f"price_roll_{window}_{stat}"] = series.reindex(
                start - pd.Timedelta(hours=1)
            ).to_numpy()
    if use_exogenous:
        # PriceFM labels these as day-ahead forecasts; no generation/actual columns.
        for c in ["load_forecast_mw", "solar_forecast_mw", "wind_forecast_mw"]:
            x[c] = df[c]
        x["net_load_forecast_mw"] = (
            df.load_forecast_mw - df.solar_forecast_mw - df.wind_forecast_mw
        )
    return x.astype(float)
