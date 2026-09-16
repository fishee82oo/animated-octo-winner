import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
import plotly.express as px
from app.common import setup, MODELS, controls, evaluate

zone, df = setup("Model Leaderboard")
days, end, exog = controls(df)
st.caption(
    "90-day sliding fits. Weekday peak = 08:00–20:00 Brussels, off-peak = other weekday hours; weekends separate. These are research groups, not retail tariffs. WAPE is a fraction; undefined metrics appear blank."
)
if st.button("Compare all models", type="primary"):
    rows = []
    issues = []
    for name in MODELS:
        try:
            _, metrics, diag = evaluate(zone, name, days, end, exog)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        rows.append(metrics.assign(model=name))
        if any(d.get("converged") is False for d in diag):
            issues.append(name)
    table = pd.concat(rows, ignore_index=True)
    overall = table[table.delivery_period == "overall"].sort_values("mae")
    st.dataframe(overall, hide_index=True)
    st.plotly_chart(
        px.bar(overall, x="model", y="mae", title="MAE (EUR/MWh)"), width="stretch"
    )
    st.subheader("By delivery period")
    st.dataframe(table[table.delivery_period != "overall"], hide_index=True)
    if issues:
        st.warning("Nonconverged fits: " + ", ".join(issues))
    st.download_button(
        "Download leaderboard", table.to_csv(index=False), "leaderboard.csv", "text/csv"
    )
