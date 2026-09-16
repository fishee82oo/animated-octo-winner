import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import streamlit as st
from app.common import setup, ROOT
from data.connectors import markets

zone, df = setup("Europe Power Spot Lab")
st.markdown(
    "Forecast European **day-ahead electricity prices** with historical data from PriceFM. Compare simple baselines and LightGBM, inspect uncertainty, and explore fixed-price hedge scenarios."
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
st.markdown(
    "[PriceFM repository](https://github.com/runyao-yu/PriceFM) · [Dataset and CC BY 4.0 license](https://huggingface.co/datasets/RunyaoYu/PriceFM) · [Research paper](https://arxiv.org/abs/2508.04875)"
)
with st.expander("Pinned source and transformation provenance"):
    st.json(json.loads((ROOT / "reports/data_provenance.json").read_text()))
st.caption(
    f"Selected zone: {zone}. UTC coverage: {df.timestamp.min()} to {df.timestamp.max()}."
)
