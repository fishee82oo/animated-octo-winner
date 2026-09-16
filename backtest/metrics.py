import numpy as np
import pandas as pd


def point_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (
        y.shape != pred.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("Invalid metric inputs")
    e = y - pred
    den = np.abs(y).sum()
    return dict(
        mae=float(np.abs(e).mean()),
        rmse=float(np.sqrt((e * e).mean())),
        wape=float(np.abs(e).sum() / den) if den else np.nan,
    )


def pinball_loss(y, pred, q):
    if not 0 < q < 1:
        raise ValueError("Quantile must be between zero and one")
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (
        y.shape != pred.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("Invalid pinball inputs")
    e = y - pred
    return float(np.maximum(q * e, (q - 1) * e).mean())


def spike_capture(y, pred, threshold):
    actual = np.asarray(y) > np.asarray(threshold)
    forecast = np.asarray(pred) > np.asarray(threshold)
    tp = int((actual & forecast).sum())
    predicted = int(forecast.sum())
    events = int(actual.sum())
    return dict(
        spike_precision=tp / predicted if predicted else np.nan,
        spike_recall=tp / events if events else np.nan,
        spike_events=events,
        spike_predictions=predicted,
    )


def report(predictions):
    rows = []
    for group, df in [
        ("overall", predictions),
        *list(predictions.groupby("tou_period")),
    ]:
        row = dict(
            tou_period=group,
            n=len(df),
            **point_metrics(df.actual, df.prediction),
            **spike_capture(df.actual, df.prediction, df.spike_threshold),
        )
        if {"p10", "p50", "p90"}.issubset(df):
            for name, q in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]:
                row["pinball_" + name] = pinball_loss(df.actual, df[name], q)
            row["coverage_80"] = float(df.actual.between(df.p10, df.p90).mean())
            row["interval_width"] = float((df.p90 - df.p10).mean())
        rows.append(row)
    return pd.DataFrame(rows)
