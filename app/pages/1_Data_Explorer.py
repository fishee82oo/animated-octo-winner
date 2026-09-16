import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
import plotly.express as px
from app.common import setup

province, p, df = setup("Data Explorer")
month = st.selectbox(
    "Month to inspect",
    sorted(df.timestamp.dt.to_period("M").astype(str).unique()),
    index=6,
)
view = df[df.timestamp.dt.to_period("M").astype(str) == month]
st.plotly_chart(
    px.line(view, x="timestamp", y="price", labels={"price": "Price (CNY/MWh)"}),
    width="stretch",
)
st.plotly_chart(
    px.line(
        view,
        x="timestamp",
        y=["load_mw", "wind_mw", "solar_mw", "hydro_mw"],
        labels={"value": "MW"},
    ),
    width="stretch",
)
daily = (
    view.assign(hour=view.timestamp.dt.hour)
    .groupby("hour")[["price", "load_mw", "wind_mw", "solar_mw"]]
    .mean()
    .reset_index()
)
a, b = st.columns(2)
a.plotly_chart(
    px.line(
        daily,
        x="hour",
        y="price",
        title="Average daily price: duck curve",
        labels={"price": "CNY/MWh"},
    ),
    width="stretch",
)
b.plotly_chart(
    px.scatter(
        view,
        x="load_mw",
        y="price",
        color="tou_period",
        opacity=0.5,
        labels={"load_mw": "Load MW", "price": "CNY/MWh"},
    ),
    width="stretch",
)
monthly = (
    df.assign(month=df.timestamp.dt.month)
    .groupby("month")[["price", "hydro_mw"]]
    .mean()
    .reset_index()
)
st.plotly_chart(
    px.line(monthly, x="month", y="price", title="Seasonal average price"),
    width="stretch",
)
st.dataframe(view.head(168), hide_index=True)
st.download_button(
    "Download selected month CSV",
    view.to_csv(index=False),
    "synthetic_month.csv",
    "text/csv",
)
