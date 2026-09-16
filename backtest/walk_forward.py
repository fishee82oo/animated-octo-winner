"""D-1 origin, local delivery-day evaluation, including 23/25-hour DST days."""

import numpy as np
import pandas as pd
from data.connectors import validate, settings
from features.build_features import (
    build_features,
    delivery_clock,
    delivery_period,
    issue_time,
)
from backtest.metrics import report


def walk_forward(
    frame,
    model_factory,
    days=3,
    min_train_days=30,
    window_days=90,
    end_date=None,
    use_exogenous=True,
):
    if (
        days < 1
        or min_train_days < 1
        or (window_days is not None and window_days < min_train_days)
    ):
        raise ValueError("Invalid evaluation windows")
    frame = validate(frame)
    df = frame.set_index("timestamp")
    x = build_features(frame, use_exogenous)
    eligible = x.notna().all(axis=1)
    local = delivery_clock(df.index)
    day_index = local.normalize()
    origins = []
    for delivery in day_index.unique():
        next_day = delivery + pd.DateOffset(days=1)
        expected = pd.date_range(
            delivery, next_day, freq="h", inclusive="left"
        ).tz_convert("UTC")
        test = day_index == delivery
        if not df.index[test].equals(expected) or not eligible[test].all():
            continue
        train = eligible & (df.index < delivery.tz_convert("UTC"))
        if window_days is not None:
            train &= df.index >= (
                delivery - pd.DateOffset(days=window_days)
            ).tz_convert("UTC")
        if len(day_index[train].unique()) < min_train_days:
            continue
        if end_date is not None and delivery.date() > pd.Timestamp(end_date).date():
            continue
        origins.append(delivery)
    if len(origins) < days:
        raise ValueError("Insufficient complete delivery days after history warmup")
    results = []
    diagnostics = []
    for delivery in origins[-days:]:
        test = day_index == delivery
        train = eligible & (df.index < delivery.tz_convert("UTC"))
        if window_days is not None:
            train &= df.index >= (
                delivery - pd.DateOffset(days=window_days)
            ).tz_convert("UTC")
        model = model_factory().fit(x.loc[train], df.loc[train, "price"])
        a = np.asarray(model.predict(x.loc[test]))
        n = int(test.sum())
        part = pd.DataFrame(
            dict(
                timestamp=df.index[test],
                delivery_date=str(delivery.date()),
                origin=issue_time(delivery),
                actual=df.loc[test, "price"].values,
                delivery_period=delivery_period(df.index[test]),
            )
        )
        if a.shape == (n, 3):
            part[["p10", "p50", "p90"]] = a
            part["prediction"] = a[:, 1]
        elif a.shape == (n,):
            part["prediction"] = a
        else:
            raise ValueError("Model returned invalid shape")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite predictions")
        part["spike_threshold"] = df.loc[train, "price"].quantile(0.9)
        results.append(part)
        diagnostics.append(
            dict(
                delivery_date=str(delivery.date()),
                origin=str(issue_time(delivery)),
                forecast_hours=n,
                train_rows=int(train.sum()),
                **getattr(model, "diagnostics", {}),
            )
        )
    predictions = pd.concat(results, ignore_index=True)
    return predictions, report(predictions), diagnostics
