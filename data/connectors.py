"""Version-pinned PriceFM input. All timestamps are timezone-aware UTC."""

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "timestamp",
    "price",
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_forecast_mw",
]


def settings():
    return yaml.safe_load((ROOT / "config/dataset.yaml").read_text())


def markets():
    return yaml.safe_load((ROOT / "config/markets.yaml").read_text())


def validate(frame):
    if not set(COLUMNS).issubset(frame):
        raise ValueError("Missing required PriceFM columns")
    df = frame[COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    if df.timestamp.isna().any():
        raise ValueError("Missing timestamps")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if (
        len(df) < 2
        or df.timestamp.duplicated().any()
        or not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Expected unique contiguous hourly UTC observations")
    if not df.timestamp.eq(df.timestamp.dt.floor("h")).all():
        raise ValueError("Hours must be aligned")
    for c in COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"Nonfinite {c}; no filling is performed")
    return df


def hourly_zone(raw, zone):
    mapping = {
        f"{zone}-price": "price",
        f"{zone}-load": "load_forecast_mw",
        f"{zone}-solar": "solar_forecast_mw",
        f"{zone}-wind": "wind_forecast_mw",
    }
    if not {"time_utc", *mapping}.issubset(raw):
        raise ValueError(f"Missing source columns for {zone}")
    df = (
        raw[["time_utc", *mapping]]
        .rename(columns={"time_utc": "timestamp", **mapping})
        .copy()
    )
    df.timestamp = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    df = df.sort_values("timestamp")
    if (
        df.timestamp.isna().any()
        or df.timestamp.duplicated().any()
        or not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    ):
        raise ValueError("Source must be contiguous, unique quarter-hours")
    if not df.timestamp.eq(df.timestamp.dt.floor("15min")).all():
        raise ValueError("Misaligned source intervals")
    numeric = df.set_index("timestamp").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric).all().all():
        raise ValueError("Source contains missing or nonfinite data")
    counts = numeric.resample("h").size()
    incomplete = counts[counts != 4]
    # Only endpoint partial hours can be omitted, never interior gaps.
    if any(t not in (counts.index[0], counts.index[-1]) for t in incomplete.index):
        raise ValueError("Incomplete interior hour")
    hourly = numeric.resample("h").mean().loc[counts.eq(4)].reset_index()
    return validate(hourly), int((counts != 4).sum())


class MarketDataConnector(ABC):
    @abstractmethod
    def load(self, start=None, end=None):
        """Half-open UTC interval [start,end)."""


class PriceFMConnector(MarketDataConnector):
    def __init__(self, zone="DE_LU", cache_dir=None):
        if zone not in markets():
            raise ValueError("Unknown bidding zone")
        self.zone = zone
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "data/cache"

    def load(self, start=None, end=None):
        path = self.cache_dir / f"{self.zone}.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Run python data/download_dataset.py to download and prepare PriceFM"
            )
        metadata = json.loads((self.cache_dir / "provenance.json").read_text())
        if metadata.get("revision") != settings()["revision"]:
            raise ValueError("Cache revision mismatch; rebuild dataset")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != metadata["cache_sha256"][self.zone]
        ):
            raise ValueError("Cache checksum mismatch; rebuild dataset")
        df = validate(pd.read_csv(path))
        if start is not None:
            df = df[df.timestamp >= pd.to_datetime(start, utc=True)]
        if end is not None:
            df = df[df.timestamp < pd.to_datetime(end, utc=True)]
        return df.reset_index(drop=True)
