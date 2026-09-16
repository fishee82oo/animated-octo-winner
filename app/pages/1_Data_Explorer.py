import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import plotly.express as px
import streamlit as st
from app.common import setup, dataset
from data.connectors import markets

zone, df = setup("Data Explorer")
local = df.timestamp.dt.tz_convert("Europe/Brussels")
months = local.dt.strftime("%Y-%m")
month = st.selectbox(
    "Delivery month", sorted(months.unique()), index=len(months.unique()) - 1
)
view = df[months == month].copy()
st.plotly_chart(
    px.line(
        view, x="timestamp", y="price", labels={"price": "EUR/MWh", "timestamp": "UTC"}
    ),
    width="stretch",
)
st.plotly_chart(
    px.line(
        view,
        x="timestamp",
        y=["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"],
        labels={"value": "Forecast MW", "timestamp": "UTC"},
        title="Source day-ahead forecasts, not measured generation",
    ),
    width="stretch",
)
view["hour"] = view.timestamp.dt.tz_convert("Europe/Brussels").dt.hour
view["net_load_forecast_mw"] = (
    view.load_forecast_mw - view.wind_forecast_mw - view.solar_forecast_mw
)
hourly = view.groupby("hour").price.mean().reset_index()
a, b = st.columns(2)
a.plotly_chart(
    px.line(hourly, x="hour", y="price", title="Average price by Brussels clock hour"),
    width="stretch",
)
b.plotly_chart(
    px.scatter(
        view,
        x="net_load_forecast_mw",
        y="price",
        color="hour",
        title="Forecast net load vs price",
        labels={"price": "EUR/MWh"},
    ),
    width="stretch",
)
other = st.selectbox(
    "Compare bidding zone", [k for k in markets() if k != zone], index=0
)
comparison = dataset(other)
comparison = comparison[comparison.timestamp.isin(view.timestamp)]
joined = pd.concat(
    [
        view[["timestamp", "price"]].assign(zone=zone),
        comparison[["timestamp", "price"]].assign(zone=other),
    ]
)
st.plotly_chart(
    px.line(
        joined,
        x="timestamp",
        y="price",
        color="zone",
        title="Same UTC delivery intervals",
    ),
    width="stretch",
)
st.caption(
    "Repeated autumn clock hours remain separate in the time-series plots; the average daily profile groups both by clock hour. Zero forecast series may be upstream placeholders."
)
st.dataframe(view.head(168), hide_index=True)
st.download_button(
    "Download selected month",
    view.to_csv(index=False),
    f"{zone}_{month}.csv",
    "text/csv",
)
