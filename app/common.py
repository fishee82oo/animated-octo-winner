from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.simulator import profiles
from data.connectors import CsvConnector, SyntheticConnector
from models.baselines import Naive, SeasonalNaive, Sarimax
from models.gbm import GBM
from models.quantile import QuantileGBM
from backtest.walk_forward import walk_forward

MODELS = {
    "Naive persistence": Naive,
    "Seasonal naive": SeasonalNaive,
    "SARIMAX": Sarimax,
    "LightGBM point": GBM,
    "LightGBM quantile": QuantileGBM,
}


def setup(title):
    st.set_page_config(
        page_title=title + " · Power Spot Lab", page_icon="⚡", layout="wide"
    )
    st.title(title)
    st.caption(
        "Research & education · Synthetic data · CNY/MWh · China local time (UTC+8) · No execution facilities"
    )
    choices = profiles()
    selected = st.session_state.get("selected_province", "shandong")
    province = st.sidebar.selectbox(
        "Province",
        list(choices),
        index=list(choices).index(selected),
        format_func=lambda k: f"{choices[k]['label']} · {k}",
        key="_province",
    )
    st.session_state["selected_province"] = province
    st.sidebar.info(
        "Hourly synthetic spot market. Tariffs and price bands are illustrative assumptions."
    )
    return province, choices[province], dataset(province)


@st.cache_data(show_spinner="Loading synthetic market…")
def dataset(province):
    file = ROOT / "data/cache" / f"{province}_3y_seed42.csv"
    if file.exists():
        return CsvConnector(file).load()
    return SyntheticConnector(province, years=3, seed=42).load()


@st.cache_data(show_spinner="Running daily walk-forward evaluation…")
def evaluate(province, model, days=3):
    return walk_forward(
        dataset(province), MODELS[model], days=days, min_train_days=30, window_days=90
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
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(32,160,160,.2)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=pred.timestamp, y=pred.actual, name="Actual", line=dict(color="#536379")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred.timestamp,
            y=pred.prediction,
            name="Forecast",
            line=dict(color="#00a6a6"),
        )
    )
    fig.update_layout(
        yaxis_title="CNY/MWh",
        xaxis_title="Settlement hour",
        legend=dict(orientation="h"),
    )
    return fig
