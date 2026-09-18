import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import plotly.express as px
import streamlit as st
from app.common import setup, evaluate, weather_available
from analytics.hedging import price_scenarios, hedge_analysis, suggest_hedge

zone, df = setup("Hedging Simulator")
st.caption(
    "Illustrative fixed-for-floating contract against day-ahead prices. Historical final delivery day; fixed energy profile, not intraday/imbalance settlement."
)
wx = st.checkbox("Use weather-informed quantiles", value=weather_available(zone), disabled=not weather_available(zone))
try:
    pred, _, _ = evaluate(zone, "LightGBM quantile", 1, use_exogenous=False, use_weather=wx)
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()
history = df[df.timestamp < pred.timestamp.min()].tail(90 * 24)
a, b, c = st.columns(3)
contract = a.number_input(
    "Fixed contract price (EUR/MWh)", value=round(float(history.price.mean()), 2)
)
volume = b.number_input(
    "Energy per hourly interval (MWh)", min_value=0.1, value=100.0, step=10.0
)
side = c.selectbox("Perspective", ["buyer", "seller"])
ratio = st.slider("Contracted share (%)", 0, 100, 90, step=5)
tolerance = st.slider("Risk tolerance", 0.0, 1.0, 0.5, step=0.05)
correlation = st.slider(
    "Assumed hourly scenario correlation", 0.0, 1.0, 0.65, step=0.05
)
confidence = st.selectbox("VaR / CVaR confidence", [0.9, 0.95, 0.99], index=1)
width = max(float(history.price.std()), 10)
low_default = float(min(history.price.min(), pred.p10.min()) - 2 * width)
high_default = float(max(history.price.max(), pred.p90.max()) + 2 * width)
a, b = st.columns(2)
low = a.number_input("Scenario lower bound (EUR/MWh)", value=round(low_default, 2))
high = b.number_input("Scenario upper bound (EUR/MWh)", value=round(high_default, 2))
st.caption(
    "Bounds are editable stress assumptions derived from past prices and forecast quantiles, not market price limits. Three quantiles alone do not determine tail risk."
)
if low >= high:
    st.error("Lower bound must be below upper bound.")
    st.stop()
scenarios = price_scenarios(
    pred[["p10", "p50", "p90"]], low, high, correlation=correlation
)
table = hedge_analysis(scenarios, volume, contract, side=side, confidence=confidence)
selected = table.iloc[np.abs(table.hedge_ratio - ratio / 100).argmin()]
a, b, c = st.columns(3)
a.metric(
    "Expected cost" if side == "buyer" else "Expected gross revenue",
    f"€{selected.expected_settlement:,.0f}",
)
b.metric("Loss VaR", f"€{selected.loss_var:,.0f}")
c.metric("Loss CVaR", f"€{selected.loss_cvar:,.0f}")
st.info(
    f"Educational heuristic: {suggest_hedge(table, tolerance):.0%} contracted. Delivery: {pred.delivery_date.iloc[0]}, {len(pred)} hours; total volume {len(pred) * volume:,.0f} MWh."
)
st.plotly_chart(
    px.line(
        table,
        x="hedge_ratio",
        y=["expected_loss", "loss_var", "loss_cvar"],
        labels={"value": "EUR", "hedge_ratio": "Contracted share"},
    ),
    width="stretch",
)
st.dataframe(table, hide_index=True)
st.markdown("""Settlement = spot × energy + (fixed price − spot) × contracted energy. Buyer loss is cost; seller loss is negative gross revenue. Generation costs, fees, volume mismatch, basis and credit risks are excluded.

Scenarios use a bounded piecewise-linear inverse CDF and Gaussian hourly dependence (5,000 paths, seed 42). Quantiles are clipped to the selected stress bounds. This is a scenario assumption, not a calibrated joint distribution or a trading recommendation. A 100% hedge removes price risk only for matched, fixed volumes and reference prices.""")
