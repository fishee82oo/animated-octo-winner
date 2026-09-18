"""Fixed-lead archived forecasts. Never substitute reanalysis or recent analyses.

The provider exposes 48-hour offsets, not publication timestamps or exact run IDs.
Reference/availability columns below are explicit bounds/assumptions, not observed
publication metadata. The latest nominal reference plus eight hours precedes D-1
11:00 even for the last delivery hour on a 25-hour day.
"""

import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import pandas as pd
import yaml
from data.connectors import ROOT

CACHE = ROOT / "data/weather_cache"
RAW = ROOT / "data/weather_raw"
WEATHER_COLUMNS = [
    "weather_temperature_c", "weather_wind_100m_ms", "weather_cloud_pct",
    "weather_radiation_wm2", "weather_precipitation_mm",
    "weather_temperature_spread_c", "weather_heating_degree_c",
    "weather_cooling_degree_c",
]


def weather_settings():
    return yaml.safe_load((ROOT / "config/weather.yaml").read_text())


def config_digest():
    return hashlib.sha256((ROOT / "config/weather.yaml").read_bytes()).hexdigest()


def fetch_json(params, raw_dir=RAW):
    """Content-addressed requests, resumable raw cache, bounded retries."""
    url = weather_settings()["endpoint"] + "?" + urlencode(params)
    key = hashlib.sha256(url.encode()).hexdigest()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{key}.json"
    if path.exists():
        payload = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urlopen(url, timeout=60) as response:
                    payload = response.read()
                result = json.loads(payload)
                if isinstance(result, dict) and result.get("error"):
                    raise ValueError(result.get("reason", "Weather API error"))
                path.write_bytes(payload)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return json.loads(payload), {
        "url": url, "raw_file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def aggregate_response(payload, sites):
    """Equal-site means; any missing site/variable invalidates that hour.

    Instant radiation is aligned to the delivery interval start. Precipitation
    is the preceding-hour accumulation at that timestamp; it is not relabelled
    as the delivery interval total. Degree features are computed at each site
    before averaging (18 C heating and 22 C cooling illustrative thresholds).
    """
    cfg = weather_settings()
    responses = payload if isinstance(payload, list) else [payload]
    if len(responses) != len(sites):
        raise ValueError("Weather response location count mismatch")
    frames = []
    for response in responses:
        if response.get("utc_offset_seconds") != 0:
            raise ValueError("Weather must use UTC")
        hourly = response["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
        if idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ValueError("Duplicate or unordered weather times")
        frame = pd.DataFrame(index=idx)
        for source, (target, unit) in cfg["variables"].items():
            key = f"{source}_previous_day{cfg['lead_days']}"
            if response["hourly_units"].get(key) != unit:
                raise ValueError(f"Unexpected weather unit for {key}")
            frame[f"weather_{target}"] = pd.to_numeric(hourly[key], errors="raise")
        if frames and not frame.index.equals(frames[0].index):
            raise ValueError("Weather locations have different time axes")
        frames.append(frame)
    values = np.stack([f.to_numpy(dtype=float) for f in frames])
    out = pd.DataFrame(values.mean(axis=0), index=frames[0].index, columns=frames[0].columns)
    temperatures = values[:, :, 0]
    out["weather_temperature_spread_c"] = temperatures.max(axis=0) - temperatures.min(axis=0)
    out["weather_heating_degree_c"] = np.maximum(18 - temperatures, 0).mean(axis=0)
    out["weather_cooling_degree_c"] = np.maximum(temperatures - 22, 0).mean(axis=0)
    out.index.name = "timestamp"
    out = out.reset_index()
    out["reference_time_bound"] = out.timestamp - pd.Timedelta(days=cfg["lead_days"])
    out["assumed_available_at"] = out.reference_time_bound + pd.Timedelta(hours=cfg["availability_delay_hours"])
    return out


def validate_weather(frame):
    required = ["timestamp", "reference_time_bound", "assumed_available_at", *WEATHER_COLUMNS]
    if not set(required).issubset(frame):
        raise ValueError("Weather cache missing required fields")
    out = frame[required].copy()
    for c in required[:3]:
        out[c] = pd.to_datetime(out[c], utc=True, errors="raise")
        if out[c].isna().any():
            raise ValueError("Missing weather time metadata")
    if out.timestamp.duplicated().any() or not out.timestamp.is_monotonic_increasing:
        raise ValueError("Weather timestamps must be ordered and unique")
    if not out.timestamp.eq(out.timestamp.dt.floor("h")).all():
        raise ValueError("Weather timestamps must be hourly aligned")
    cfg = weather_settings()
    if not (out.timestamp - out.reference_time_bound).eq(pd.Timedelta(days=cfg["lead_days"])).all():
        raise ValueError("Weather lead does not match configured fixed lead")
    if not (out.assumed_available_at - out.reference_time_bound).eq(pd.Timedelta(hours=cfg["availability_delay_hours"])).all():
        raise ValueError("Weather availability allowance mismatch")
    values = out[WEATHER_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values).all().all():
        raise ValueError("Weather cache contains missing or nonfinite values")
    if not values.weather_cloud_pct.between(0, 100).all():
        raise ValueError("Invalid cloud percentage")
    if (values[["weather_wind_100m_ms", "weather_radiation_wm2", "weather_precipitation_mm"]] < 0).any().any():
        raise ValueError("Invalid negative weather quantity")
    out[WEATHER_COLUMNS] = values
    return out


def download_weather(zone="DE_LU", start=None, end=None, cache_dir=CACHE, fetcher=fetch_json):
    cfg = weather_settings()
    if zone not in cfg["zones"]:
        raise ValueError("No weather sampling profile for zone")
    start = pd.Timestamp(start or cfg["default_start"]).normalize()
    end = pd.Timestamp(end or cfg["default_end"]).normalize()
    if start.tzinfo or end.tzinfo or start > end:
        raise ValueError("Use ordered UTC calendar dates without timezone")
    sites = cfg["zones"][zone]
    chunks, requests = [], []
    current = start
    while current <= end:
        stop = min(current + pd.DateOffset(months=3) - pd.Timedelta(days=1), end)
        params = dict(
            latitude=",".join(str(s[1]) for s in sites),
            longitude=",".join(str(s[2]) for s in sites),
            start_date=str(current.date()), end_date=str(stop.date()),
            hourly=",".join(f"{v}_previous_day{cfg['lead_days']}" for v in cfg["variables"]),
            models=cfg["model"], wind_speed_unit="ms", timezone="GMT",
        )
        payload, provenance = fetcher(params)
        part = aggregate_response(payload, sites)
        expected = pd.date_range(current.tz_localize("UTC"), (stop + pd.Timedelta(days=1)).tz_localize("UTC"), freq="h", inclusive="left")
        if not pd.DatetimeIndex(part.timestamp).equals(expected):
            raise ValueError("API returned unexpected weather coverage")
        chunks.append(part)
        requests.append(provenance)
        print(f"Weather {zone}: {current.date()} to {stop.date()}", flush=True)
        current = stop + pd.Timedelta(days=1)
    full = pd.concat(chunks, ignore_index=True)
    valid = np.isfinite(full[WEATHER_COLUMNS]).all(axis=1)
    missing = full.loc[~valid, "timestamp"].astype(str).tolist()
    frame = validate_weather(full.loc[valid].reset_index(drop=True))
    if frame.empty:
        raise ValueError("No complete weather rows; choose a period with archive coverage")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{zone}.csv"
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    manifest = dict(
        provider=cfg["provider"], model=cfg["model"], zone=zone,
        config_sha256=config_digest(), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        requested_start=str(start.date()), requested_end=str(end.date()),
        first_timestamp=str(frame.timestamp.min()), last_timestamp=str(frame.timestamp.max()),
        rows=len(frame), missing_hours=len(missing), missing_timestamps=missing,
        sites=sites, sampling_note=cfg["sampling_note"], lead_days=cfg["lead_days"],
        availability_delay_hours=cfg["availability_delay_hours"],
        timing="Fixed-lead archive; nominal reference bound and assumed availability, not verified publication vintages or a common daily run.",
        license="Weather data CC BY 4.0; public API non-commercial research only; commercial API terms apply separately.",
        sources=[cfg["documentation"], "https://open-meteo.com/en/terms", "https://www.dwd.de/"],
        requests=requests,
    )
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return frame, manifest


class WeatherConnector:
    def __init__(self, zone="DE_LU", cache_dir=CACHE):
        self.zone, self.cache_dir = zone, Path(cache_dir)

    def load(self):
        path = self.cache_dir / f"{self.zone}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Weather cache missing. Run python data/download_weather.py --zone {self.zone}")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["zone"] != self.zone or metadata["config_sha256"] != config_digest():
            raise ValueError("Weather configuration changed; rebuild cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["cache_sha256"]:
            raise ValueError("Weather cache checksum mismatch")
        return validate_weather(pd.read_csv(path))

    def metadata(self):
        return json.loads((self.cache_dir / f"{self.zone}.json").read_text())
