from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from data.simulator import simulate, SCHEMA


def validate(frame):
    missing = set(SCHEMA) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame[SCHEMA].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if (
        len(df) < 2
        or df.timestamp.duplicated().any()
        or not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Data must be unique, contiguous hourly observations")
    if not df.timestamp.eq(df.timestamp.dt.floor("h")).all():
        raise ValueError("Timestamps must align to whole hours")
    if df.timestamp.dt.tz is not None:
        raise ValueError("Use naive China local time (UTC+8)")
    for col in SCHEMA[1:-1]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Nonfinite {col}")
    if not df.tou_period.isin(["sharp_peak", "peak", "flat", "valley"]).all():
        raise ValueError("Unknown TOU bucket")
    if (df[["load_mw", "wind_mw", "solar_mw", "hydro_mw"]] < 0).any().any():
        raise ValueError("Negative physical output")
    return df


class MarketDataConnector(ABC):
    @abstractmethod
    def load(self, start=None, end=None):
        """Return [start, end) in naive China local time."""


class SyntheticConnector(MarketDataConnector):
    def __init__(self, province="shandong", years=3, seed=42):
        self.options = dict(province=province, years=years, seed=seed)

    def load(self, start=None, end=None):
        return _slice(validate(simulate(**self.options)), start, end)


class CsvConnector(MarketDataConnector):
    """Manual CSV adapter; no portal, network or credential integration."""

    def __init__(self, path):
        self.path = path

    def load(self, start=None, end=None):
        return _slice(validate(pd.read_csv(self.path)), start, end)


def _slice(df, start, end):
    if start is not None:
        df = df[df.timestamp >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.timestamp < pd.Timestamp(end)]
    return df.reset_index(drop=True)
