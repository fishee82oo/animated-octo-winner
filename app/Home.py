import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import streamlit as st
from app.common import setup, ROOT
from data.connectors import markets

zone, df = setup("Europe Power Spot Lab")
st.markdown(
    "Forecast European **day-ahead electricity prices** with PriceFM prices and archived **DWD ICON weather forecasts via Open-Meteo**. Compare baselines and LightGBM, test the incremental value of weather, inspect uncertainty, and explore fixed-price hedge scenarios."
)
a, b, c = st.columns(3)
a.metric("Bidding zones", len(markets()))
b.metric("Hourly records in selected zone", f"{len(df):,}")
c.metric("Negative-price hours", f"{(df.price < 0).mean():.1%}")
st.markdown("""### Research workflow
1. **Data Explorer:** prices, forecast supply/demand and differences between zones.
2. **Forecast & Backtest:** delivery-day point forecasts and P10/P50/P90 bands.
3. **Model Leaderboard:** compare models on identical dates and information sets.
4. **Hedging Simulator:** test fixed-for-floating contracts against scenario risk.

Day-ahead prices are quoted in EUR/MWh. Timestamps are stored in UTC; delivery days and calendar features use Europe/Brussels, with 23/25-hour daylight-saving days preserved. Hourly prices are averages of four source quarter-hours. This hourly target does not reproduce every quarter-hour price spike.

Prices for D−1 delivery are already known before the D−1 auction for D delivery. The backtest uses those historical day-ahead prices, not future intraday or imbalance prices.""")
st.warning(
    "PriceFM is a processed historical research dataset: upstream interpolation and zero placeholders remain. Forecast issue timestamps are unavailable. No live prices, broker connections, real money or order execution."
)
st.caption(
    "PriceFM is the data source. This app runs its own baselines and LightGBM; it does not run the PriceFM neural model or reproduce its published scores."
)
st.info("Weather inputs: temperature, 100 m wind, cloud cover, instantaneous solar radiation, precipitation, and heating/cooling degree features. Fixed 48-hour-lead forecasts are checked against D−1 11:00 using an eight-hour availability allowance. This is an explicit timing assumption, not verified historical publication metadata.")
st.markdown("[Weather archive documentation](https://open-meteo.com/en/docs/previous-runs-api) · [DWD](https://www.dwd.de/) · [Open-Meteo usage terms](https://open-meteo.com/en/terms)")
st.markdown(
    "[PriceFM repository](https://github.com/runyao-yu/PriceFM) · [Dataset and CC BY 4.0 license](https://huggingface.co/datasets/RunyaoYu/PriceFM) · [Research paper](https://arxiv.org/abs/2508.04875)"
)
with st.expander("Pinned source and transformation provenance"):
    st.json(json.loads((ROOT / "reports/data_provenance.json").read_text()))
st.caption(
    f"Selected zone: {zone}. UTC coverage: {df.timestamp.min()} to {df.timestamp.max()}."
)
