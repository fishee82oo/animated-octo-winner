import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import plotly.express as px
import streamlit as st
from app.common import setup, dataset, weather_available, weather_dataset
from data.weather import WeatherConnector
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

st.subheader("Archived weather forecasts")
if weather_available(zone):
    try:
        weather = weather_dataset(zone)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()
    joined_weather = view[["timestamp", "price"]].merge(weather, on="timestamp", how="inner")
    st.caption(f"{len(joined_weather)} of {len(view)} selected hours have complete weather. Forecasts use a fixed 48-hour lead; geographic sampling is illustrative.")
    variable = st.selectbox("Weather variable", [c for c in weather.columns if c.startswith("weather_")])
    if not joined_weather.empty:
        st.plotly_chart(px.line(joined_weather, x="timestamp", y=variable, title="Archived forecast values, not observed weather"), width="stretch")
        st.plotly_chart(px.scatter(joined_weather, x=variable, y="price", labels={"price": "EUR/MWh"}, title="Weather forecast and historical price"), width="stretch")
        st.download_button("Download aligned weather and prices", joined_weather.to_csv(index=False), f"{zone}_{month}_weather.csv", "text/csv")
    with st.expander("Weather source, sites, coverage and timing assumptions"):
        st.json(WeatherConnector(zone).metadata())
else:
    st.info(f"Download weather with: python data/download_weather.py --zone {zone}")
