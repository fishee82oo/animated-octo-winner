from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.connectors import markets, settings, PriceFMConnector
from models.baselines import Naive, SeasonalNaive, Sarimax
from models.gbm import GBM
from models.quantile import QuantileGBM
from backtest.walk_forward import walk_forward
from data.weather import WeatherConnector, CACHE as WEATHER_CACHE

MODELS = {
    "Persistence": Naive,
    "Weekly seasonal naive": SeasonalNaive,
    "SARIMAX": Sarimax,
    "LightGBM point": GBM,
    "LightGBM quantile": QuantileGBM,
}


def setup(title):
    st.set_page_config(
        page_title=title + " · Europe Power Spot Lab", page_icon="⚡", layout="wide"
    )
    st.title(title)
    st.caption("PriceFM historical research data · EUR/MWh · No trading or execution")
    choices = markets()
    selected = st.session_state.get("selected_zone", "DE_LU")
    zone = st.sidebar.selectbox(
        "Bidding zone",
        list(choices),
        index=list(choices).index(selected),
        format_func=lambda x: f"{x} · {choices[x]}",
        key="_zone",
    )
    st.session_state["selected_zone"] = zone
    st.sidebar.info(
        "Historical day-ahead prices. Source: PriceFM / Runyao Yu et al. CC BY 4.0. Delivery days: Europe/Brussels."
    )
    try:
        df = dataset(zone)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.code("python data/download_dataset.py")
        st.stop()
    return zone, df


@st.cache_data(show_spinner="Loading verified PriceFM cache…")
def dataset(zone):
    return PriceFMConnector(zone).load()


def weather_dataset(zone):
    return WeatherConnector(zone).load()


def weather_available(zone):
    return (WEATHER_CACHE / f"{zone}.csv").exists()


def evaluate(zone, model, days=3, end_date="2025-12-31", use_exogenous=True, use_weather=False, match_weather=False):
    manifest = WEATHER_CACHE / f"{zone}.json"
    version = manifest.read_text() if (use_weather or match_weather) and manifest.exists() else ""
    return _evaluate(zone, model, days, end_date, use_exogenous, use_weather, match_weather, version)


@st.cache_data(show_spinner="Running delivery-day walk-forward evaluation…")
def _evaluate(zone, model, days, end_date, use_exogenous, use_weather, match_weather, weather_version):
    return walk_forward(
        dataset(zone),
        MODELS[model],
        days=days,
        end_date=end_date,
        use_exogenous=use_exogenous,
        weather=weather_dataset(zone) if use_weather or match_weather else None,
        use_weather=use_weather,
    )


evaluate.clear = _evaluate.clear


def forecast_chart(pred):
    fig = go.Figure()
    if "p10" in pred:
        fig.add_trace(
            go.Scatter(x=pred.timestamp, y=pred.p90, name="P90", line=dict(width=0))
        )
        fig.add_trace(
            go.Scatter(
                x=pred.timestamp,
                y=pred.p10,
                name="P10–P90",
                fill="tonexty",
                fillcolor="rgba(20,160,170,.2)",
                line=dict(width=0),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=pred.timestamp,
            y=pred.actual,
            name="Historical price",
            line=dict(color="#536379"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred.timestamp,
            y=pred.prediction,
            name="Forecast",
            line=dict(color="#009eab"),
        )
    )
    fig.update_layout(
        yaxis_title="EUR/MWh",
        xaxis_title="Delivery timestamp (UTC)",
        legend=dict(orientation="h"),
    )
    return fig


def controls(df):
    days = st.slider("Evaluation delivery days", 1, 90, 7)
    local = df.timestamp.dt.tz_convert("Europe/Brussels")
    end = st.date_input(
        "Last delivery date",
        value=local.iloc[-1].date(),
        min_value=local.iloc[0].date(),
        max_value=local.iloc[-1].date(),
    )
    exog = st.checkbox("Include PriceFM load, wind and solar forecasts", value=False)
    zone = st.session_state.get("selected_zone", "DE_LU")
    available = weather_available(zone)
    wx = st.checkbox("Include archived weather forecasts", value=available, disabled=not available)
    if available:
        st.caption("Weather: DWD ICON via Open-Meteo, fixed 48-hour lead; eight-hour assumed release allowance. Regional sites use equal weights, not power-system weights. Missing weather hours are excluded.")
    else:
        st.caption(f"Add weather for this zone: python data/download_weather.py --zone {zone}")
    st.caption(
        "Research origin: 11:00 Europe/Brussels on D−1. PriceFM has no issue-time vintages or per-value imputation flags; this is a retrospective benchmark, not a verified auction-time replay. Disable both input groups for prices/calendar only."
    )
    return days, str(end), exog, wx
