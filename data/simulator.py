"""Hourly educational spot-market process; parameters are not calibrated tariffs."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = [
    "timestamp",
    "price",
    "load_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "coal_index",
    "tou_period",
]


def profiles():
    return yaml.safe_load((ROOT / "config/provinces.yaml").read_text())


def tou_period(hours):
    h = np.asarray(hours)
    return np.select(
        [np.isin(h, [18, 19, 20]), np.isin(h, [9, 10, 11, 16, 17, 21]), h < 7],
        ["sharp_peak", "peak", "valley"],
        default="flat",
    )


def simulate(province="shandong", years=3, seed=42, start="2021-01-01"):
    if not np.isfinite(years) or years <= 0:
        raise ValueError("years must be positive")
    p = profiles()[province]
    n = int(round(years * 365 * 24))
    if n < 24:
        raise ValueError("Generate at least one day")
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="h")
    h, doy = t.hour.to_numpy(), t.dayofyear.to_numpy()
    summer = np.maximum(0, np.cos(2 * np.pi * (doy - 205) / 365))
    winter = np.maximum(0, np.cos(2 * np.pi * (doy - 20) / 365))
    days = np.arange(n) // 24
    heat = rng.random(days.max() + 1) < 0.025
    cold = rng.random(days.max() + 1) < 0.02
    curtail = rng.random(days.max() + 1) < 0.018
    load = p["load_base"] * (
        0.83
        + 0.15 * np.sin(2 * np.pi * (h - 8) / 24)
        + 0.10 * np.exp(-(((h - 19) / 3) ** 2))
        + p["summer"] * summer
        + p["winter"] * winter
        - 0.09 * (t.dayofweek.to_numpy() >= 5)
        + 0.22 * heat[days] * (summer > 0.6)
        + 0.2 * cold[days] * (winter > 0.6)
        + rng.normal(0, 0.025, n)
    )
    cloud = rng.uniform(0.55, 1.05, days.max() + 1)
    solar = (
        p["load_base"]
        * p["solar_share"]
        * np.maximum(0, np.sin(np.pi * (h - 6) / 12)) ** 1.5
        * (0.75 + 0.25 * summer)
        * cloud[days]
    )
    wind_factor = np.empty(n)
    wind_factor[0] = 0.55
    for i in range(1, n):
        mean = 0.53 + 0.12 * winter[i]
        wind_factor[i] = np.clip(
            wind_factor[i - 1]
            + 0.06 * (mean - wind_factor[i - 1])
            + rng.normal(0, 0.075),
            0.03,
            1.2,
        )
    wind = p["load_base"] * p["wind_share"] * np.where(curtail[days], 1.3, wind_factor)
    wet = np.isin(t.month, [6, 7, 8, 9, 10])
    hydro = (
        p["load_base"]
        * p["hydro_share"]
        * np.where(wet, p["wet_hydro_multiplier"], 0.65)
        * (1 + rng.normal(0, 0.02, n))
    )
    coal = np.empty(n)
    coal[0] = 800
    for i in range(1, n):
        coal[i] = max(
            400,
            coal[i - 1]
            + 0.002 * (800 - coal[i - 1])
            + rng.normal(0, 0.7)
            + (rng.uniform(50, 130) if rng.random() < 1 / (24 * 150) else 0),
        )
    tou = tou_period(h)
    step = (
        pd.Series(tou).map(dict(sharp_peak=100, peak=55, flat=0, valley=-65)).to_numpy()
    )
    price = (
        p["benchmark_price"]
        + p["coal_sensitivity"] * (coal - 800)
        + p["load_sensitivity"] * (load / p["load_base"] - 0.95)
        - p["renewable_sensitivity"] * (wind + solar + 0.7 * hydro) / p["load_base"]
        + step
        + rng.normal(0, 22, n)
    )
    price += 220 * heat[days] * (summer > 0.6) + 150 * cold[days] * (winter > 0.6)
    price -= 350 * curtail[days] * (h < 7)
    return pd.DataFrame(
        dict(
            timestamp=t,
            price=np.clip(price, p["price_floor"], p["price_cap"]),
            load_mw=load,
            wind_mw=wind,
            solar_mw=solar,
            hydro_mw=hydro,
            coal_index=coal,
            tou_period=tou,
        )
    )
