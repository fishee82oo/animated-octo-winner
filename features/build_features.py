"""Fixed-origin day-ahead features; target prices are never inputs.

Each day's 24 rows share a midnight origin. Price t-1 is available only for
hour zero and otherwise is the last known price (frozen, not realized).
Historical driver levels and coal are frozen; *_forecast are explicit noisy
synthetic ex-ante forecasts, not observations available in real deployment.
"""

import numpy as np
import pandas as pd
from data.connectors import validate
from data.simulator import tou_period

TOU = ["sharp_peak", "peak", "flat", "valley"]


def build_features(frame, seed=123, holidays=None):
    df = validate(frame).set_index("timestamp")
    idx = df.index
    origin = idx.normalize()
    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    # Explicit supplied dates; default is New Year only, not a Chinese calendar.
    dates = set(pd.to_datetime(holidays).date) if holidays is not None else set()
    out["holiday"] = np.array(
        [d.date() in dates or (d.month == 1 and d.day == 1) for d in idx], int
    )
    for bucket in TOU:
        out["tou_" + bucket] = (tou_period(idx.hour) == bucket).astype(int)

    def at(series, times):
        return series.reindex(times).to_numpy()

    out["price_lag_1"] = at(df.price, origin - pd.Timedelta(hours=1))
    for lag in [24, 168]:
        out[f"price_lag_{lag}"] = df.price.shift(lag)
    for window in [24, 168]:
        for stat in ["mean", "std"]:
            rolled = getattr(df.price.rolling(window), stat)()
            out[f"price_roll_{window}_{stat}"] = at(
                rolled, origin - pd.Timedelta(hours=1)
            )
    for j, col in enumerate(["load_mw", "wind_mw", "solar_mw"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, j]))
        out[col + "_last"] = at(df[col], origin - pd.Timedelta(hours=1))
        # Multiplicative error avoids fitting a noise scale on future data.
        out[col + "_forecast"] = np.maximum(
            0, df[col].to_numpy() * (1 + rng.normal(0, 0.12, len(df)))
        )
    out["hydro_mw_last"] = at(df.hydro_mw, origin - pd.Timedelta(hours=1))
    out["coal_index"] = at(df.coal_index, origin - pd.Timedelta(hours=1))
    out["coal_trend"] = at(
        df.coal_index - df.coal_index.shift(168), origin - pd.Timedelta(hours=1)
    )
    return out.astype(float)
