"""Paired weather ablation: identical delivery dates and training sample masks."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
from backtest.walk_forward import walk_forward
from backtest.metrics import report
from data.connectors import ROOT, PriceFMConnector
from data.weather import WeatherConnector
from models.gbm import GBM
from models.quantile import QuantileGBM
from models.baselines import Naive, SeasonalNaive, Sarimax


def run(zone="DE_LU", days=30, endpoints=None, include_sarimax=True):
    endpoints = endpoints or ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    frame, weather = PriceFMConnector(zone).load(), WeatherConnector(zone).load()
    arms = [("Persistence", Naive, False, False), ("Weekly seasonal", SeasonalNaive, False, False)]
    if include_sarimax:
        arms += [("SARIMAX", Sarimax, False, False), ("SARIMAX", Sarimax, False, True)]
    for name, model in [("LightGBM", GBM), ("Quantile LightGBM", QuantileGBM)]:
        for exog in [False, True]:
            for wx in [False, True]:
                arms.append((name, model, exog, wx))
    predictions, metrics, diagnostics = [], [], []
    reference = {}
    for endpoint in endpoints:
        for name, model, exog, wx in arms:
            p, m, d = walk_forward(frame, model, days=days, end_date=endpoint,
                                  weather=weather, use_weather=wx, use_exogenous=exog)
            if p.delivery_date.max() != endpoint:
                raise ValueError(f"Incomplete endpoint {endpoint}; refusing silently shifted evaluation")
            key = (tuple(p.timestamp), tuple((v["train_rows"], v["train_start"], v["train_end"]) for v in d))
            if endpoint in reference and key != reference[endpoint]:
                raise AssertionError("Ablation arms have different samples")
            reference[endpoint] = key
            fields = dict(zone=zone, model=name, pricefm_forecasts=exog, weather=wx, evaluation_end=endpoint)
            predictions.append(p.assign(**fields))
            metrics.append(m.assign(**fields))
            diagnostics.extend([dict(**v, **fields) for v in d])
            print(f"{zone} {endpoint} {name} PriceFM={exog} weather={wx}: MAE={m.iloc[0].mae:.3f}", flush=True)
    out = ROOT / "reports/weather"
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.concat(predictions, ignore_index=True)
    if pred.duplicated(["timestamp", "model", "pricefm_forecasts", "weather"]).any():
        raise ValueError("Evaluation windows overlap; use disjoint endpoints")
    summaries = []
    for (model, exog, wx), group in pred.groupby(["model", "pricefm_forecasts", "weather"]):
        # Concatenation introduces empty quantile columns for point models.
        if group.p10.isna().all():
            group = group.drop(columns=["p10", "p50", "p90"])
        summaries.append(report(group).assign(zone=zone, model=model, pricefm_forecasts=exog, weather=wx))
    summary = pd.concat(summaries, ignore_index=True)
    pred.to_csv(out / f"{zone}_predictions.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(out / f"{zone}_by_window.csv", index=False)
    summary.to_csv(out / f"{zone}_summary.csv", index=False)
    manifest = dict(zone=zone, days_per_window=days, endpoints=endpoints, window_days=90,
                    paired_samples_verified=True, weather_provenance=WeatherConnector(zone).metadata(),
                    diagnostics=diagnostics,
                    limitations=["Retrospective research, not trading P&L", "Weather availability is a conservative assumption, not observed publication metadata", "PriceFM forecast vintages unavailable", "Regional sites are geographic proxies", "No model tuning on evaluation windows"])
    (out / f"{zone}_benchmark.json").write_text(json.dumps(manifest, indent=2))
    overall = summary[summary.delivery_period == "overall"].copy()
    overall["inputs"] = overall.apply(lambda r: ("Prices + PriceFM" if r.pricefm_forecasts else "Prices") + (" + weather" if r.weather else ""), axis=1)
    fig = px.bar(overall, x="model", y="mae", color="inputs", barmode="group", title=f"{zone}: paired seasonal evaluation ({days * len(endpoints)} delivery days)", labels={"mae": "MAE (EUR/MWh)"})
    fig.write_html(out / f"{zone}_comparison.html", include_plotlyjs=True)
    week = weather.tail(168)
    fig = px.line(week, x="timestamp", y="weather_temperature_c", title=f"{zone}: archived 48-hour-lead regional temperature forecast", labels={"weather_temperature_c": "°C", "timestamp": "Valid time (UTC)"})
    fig.write_html(out / f"{zone}_weather_week.html", include_plotlyjs=True)
    print(overall[["model", "inputs", "mae", "rmse", "coverage_80"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", default="DE_LU")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--end-dates", nargs="+")
    parser.add_argument("--skip-sarimax", action="store_true")
    args = parser.parse_args()
    run(args.zone, args.days, args.end_dates, not args.skip_sarimax)
