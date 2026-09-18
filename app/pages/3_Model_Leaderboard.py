import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
import plotly.express as px
from app.common import setup, MODELS, controls, evaluate

zone, df = setup("Model Leaderboard")
days, end, exog, wx = controls(df)
st.caption(
    "90-day sliding fits. Weekday peak = 08:00–20:00 Brussels, off-peak = other weekday hours; weekends separate. These are research groups, not retail tariffs. WAPE is a fraction; undefined metrics appear blank."
)
if st.button("Compare all models", type="primary"):
    rows = []
    issues = []
    for name in MODELS:
        try:
            _, metrics, diag = evaluate(zone, name, days, end, exog, wx)
        except (ValueError, FileNotFoundError) as exc:
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

if wx and st.button("Compare LightGBM with / without weather"):
    rows = []
    timestamps = None
    for name in ["LightGBM point", "LightGBM quantile"]:
        for include in [False, True]:
            try:
                p, m, _ = evaluate(zone, name, days, end, exog, include, match_weather=True)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
                st.stop()
            if timestamps is not None and not p.timestamp.equals(timestamps):
                st.error("Comparison dates differ; cannot compare fairly.")
                st.stop()
            timestamps = p.timestamp
            rows.append(m.assign(model=name, weather=include))
    paired = pd.concat(rows, ignore_index=True)
    st.subheader("Paired weather ablation")
    st.caption("Both arms use identical training eligibility and delivery dates. Weather improvement is measured, not assumed.")
    st.dataframe(paired, hide_index=True)
    st.download_button("Download weather comparison", paired.to_csv(index=False), "weather_ablation.csv", "text/csv")
