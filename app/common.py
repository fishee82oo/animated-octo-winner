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


@st.cache_data(show_spinner="Running delivery-day walk-forward evaluation…")
def evaluate(zone, model, days=3, end_date="2025-12-31", use_exogenous=True):
    return walk_forward(
        dataset(zone),
        MODELS[model],
        days=days,
        end_date=end_date,
        use_exogenous=use_exogenous,
    )


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
    days = st.slider("Evaluation delivery days", 1, 14, 3)
    local = df.timestamp.dt.tz_convert("Europe/Brussels")
    end = st.date_input(
        "Last delivery date",
        value=local.iloc[-1].date(),
        min_value=local.iloc[0].date(),
        max_value=local.iloc[-1].date(),
    )
    exog = st.checkbox("Include PriceFM load, wind and solar forecasts", value=True)
    st.caption(
        "Research origin: 11:00 Europe/Brussels on D−1. PriceFM has no issue-time vintages or per-value imputation flags; this is a retrospective benchmark, not a verified auction-time replay. Uncheck forecast inputs for a price-only comparison."
    )
    return days, str(end), exog
