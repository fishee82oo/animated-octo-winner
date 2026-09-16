"""Daily rolling origin with fully separated targets and origin-safe features."""

import numpy as np
import pandas as pd
from data.connectors import validate
from features.build_features import build_features
from backtest.metrics import report


def walk_forward(
    frame, model_factory, days=7, min_train_days=30, window_days=None, feature_seed=123
):
    if (
        days < 1
        or min_train_days < 1
        or (window_days is not None and window_days < min_train_days)
    ):
        raise ValueError("Invalid training/evaluation window")
    df = validate(frame).set_index("timestamp")
    x = build_features(frame, seed=feature_seed)
    eligible = x.notna().all(axis=1)
    origins = [
        d
        for d in df.index.normalize().unique()
        if len(df.loc[df.index.normalize() == d]) == 24
        and (eligible & (x.index < d)).sum() >= min_train_days * 24
    ]
    if len(origins) < days:
        raise ValueError("Insufficient complete days after feature warmup and training")
    results = []
    diagnostics = []
    for origin in origins[-days:]:
        train = eligible & (x.index < origin)
        if window_days is not None:
            train &= x.index >= origin - pd.Timedelta(days=window_days)
        test = (x.index >= origin) & (x.index < origin + pd.Timedelta(days=1))
        if not eligible[test].all():
            raise ValueError("Missing forecast features")
        model = model_factory().fit(x.loc[train], df.loc[train, "price"])
        prediction = model.predict(x.loc[test])
        part = pd.DataFrame(
            dict(
                timestamp=x.index[test],
                origin=origin,
                actual=df.loc[test, "price"].values,
                tou_period=df.loc[test, "tou_period"].values,
            )
        )
        a = np.asarray(prediction)
        if a.ndim == 2 and a.shape == (24, 3):
            part[["p10", "p50", "p90"]] = a
            part["prediction"] = a[:, 1]
        elif a.shape == (24,):
            part["prediction"] = a
        else:
            raise ValueError("Model must return (24,) or (24,3)")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite forecast")
        part["spike_threshold"] = df.loc[train, "price"].quantile(0.9)
        results.append(part)
        diagnostics.append(
            dict(
                origin=str(origin),
                train_rows=int(train.sum()),
                **getattr(model, "diagnostics", {}),
            )
        )
    predictions = pd.concat(results, ignore_index=True)
    return predictions, report(predictions), diagnostics
