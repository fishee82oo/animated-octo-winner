import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import streamlit as st
import plotly.express as px
from app.common import setup, evaluate
from analytics.hedging import price_scenarios, hedge_analysis, suggest_hedge

province, p, df = setup("Hedging Simulator")
st.caption(
    "Scenario exercise for the final held-out day, viewed from its midnight forecast origin. Fixed volume profile; no dispatch optimization or execution."
)
a, b, c = st.columns(3)
contract = a.number_input("Contract price (CNY/MWh)", value=float(p["benchmark_price"]))
volume = b.number_input(
    "Metered volume per hour (MWh)", min_value=0.1, value=100.0, step=10.0
)
side = c.selectbox("Perspective", ["buyer", "seller"])
ratio = st.slider("Contracted share (%)", 0, 100, 90, step=5)
tolerance = st.slider(
    "Risk tolerance (0 = cautious, 1 = expected value)", 0.0, 1.0, 0.5, step=0.05
)
confidence = st.selectbox("VaR / CVaR confidence", [0.9, 0.95, 0.99], index=1)
correlation = st.slider(
    "Assumed hourly scenario correlation", 0.0, 1.0, 0.65, step=0.05
)
pred, _, _ = evaluate(province, "LightGBM quantile", 1)
scenarios = price_scenarios(
    pred[["p10", "p50", "p90"]],
    p["price_floor"],
    p["price_cap"],
    correlation=correlation,
)
table = hedge_analysis(scenarios, volume, contract, side=side, confidence=confidence)
selected = table.iloc[np.abs(table.hedge_ratio - ratio / 100).argmin()]
a, b, c = st.columns(3)
a.metric(
    "Expected cost" if side == "buyer" else "Expected gross revenue",
    f"¥{selected.expected_settlement:,.0f}",
)
b.metric("Loss VaR", f"¥{selected.loss_var:,.0f}")
c.metric("Loss CVaR", f"¥{selected.loss_cvar:,.0f}")
st.info(
    f"Scenario heuristic suggests {suggest_hedge(table, tolerance):.0%} contracted share. Forecast origin: {pred.origin.iloc[0]}."
)
st.plotly_chart(
    px.line(
        table,
        x="hedge_ratio",
        y=["expected_loss", "loss_var", "loss_cvar"],
        labels={"value": "CNY", "hedge_ratio": "Contracted share"},
        title="Expected loss and upper-tail loss",
    ),
    width="stretch",
)
st.dataframe(table, hide_index=True)
st.markdown("""**Assumptions:** piecewise-linear inverse CDF through the floor, P10, P50, P90 and cap; Gaussian hourly dependence; 5,000 scenarios with seed 42. Tails and dependence are imposed, not learned. Quantiles are clipped to province bounds.

Buyer loss is settlement cost; seller loss is negative gross revenue. Negative seller VaR is possible and is not profit after production costs. Spot-exposure columns exclude the fixed contract leg. A 100% hedge removes price risk only under the assumed fixed volume and matched reference price; volume, basis, credit and fee risks are excluded.

The heuristic minimizes expected loss plus a tolerance-dependent penalty on CVaR above the mean. It is not a trading recommendation.""")
