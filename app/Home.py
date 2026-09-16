import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.common import setup

province, p, df = setup("China Power Spot Lab")
st.markdown(
    "A research prototype for short-term provincial electricity prices and contract hedging. Explore synthetic supply and demand, compare next-day forecasts, and examine settlement risk."
)
a, b, c = st.columns(3)
a.metric("Hourly observations", f"{len(df):,}")
b.metric("Illustrative benchmark", f"{p['benchmark_price']} CNY/MWh")
c.metric("Typical contract share assumption", f"{p['contract_share']:.0%}")
st.markdown("""### Two coupled market layers
**中长期 — medium-and-long-term contracts:** bilateral or listed contracts commonly cover most volume. This simulator assumes a benchmark plus a configurable floating band; those settings are not official market rules.

**现货 — spot market:** hourly day-ahead prices guide this prototype. Real-time markets operate at finer intervals; this project does not simulate a separate real-time clearing process.

**价差结算 — contract for differences:** spot settlement plus a financial contract adjustment gives `cost = spot × metered volume + (contract − spot) × contracted volume`.

Use the sidebar pages to explore data, run forecasts and backtests, compare all five models, and simulate contract ratios from 0% to 100%.""")
st.warning(
    "All prices are synthetic and directionally designed, not calibrated to observed provincial prices. This is not a live trading system or a financial recommendation."
)
with st.expander("Selected illustrative province parameters"):
    st.json(p)
st.caption(
    f"Data range: {df.timestamp.min()} to {df.timestamp.max()}. Seed 42. Daily origins at midnight; final 90 days of history used for each model fit."
)
