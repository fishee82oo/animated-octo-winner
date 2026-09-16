import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
import plotly.express as px
from app.common import setup, MODELS, evaluate

province, p, df = setup("Model Leaderboard")
days = st.slider("Shared evaluation days", 1, 14, 3)
st.caption(
    "All models use identical origins, a 90-day sliding training window and the same synthetic driver forecasts. WAPE is a fraction; undefined spike precision/recall appears blank."
)
if st.button("Compare all models", type="primary"):
    rows = []
    issues = []
    for model in MODELS:
        _, metrics, diagnostics = evaluate(province, model, days)
        metrics["model"] = model
        rows.append(metrics)
        if any(d.get("converged") is False for d in diagnostics):
            issues.append(model)
    table = pd.concat(rows, ignore_index=True)
    overall = table[table.tou_period == "overall"].sort_values("mae")
    st.dataframe(overall, hide_index=True)
    st.plotly_chart(
        px.bar(overall, x="model", y="mae", title="Overall MAE (CNY/MWh)"),
        width="stretch",
    )
    st.subheader("By TOU bucket")
    st.dataframe(table[table.tou_period != "overall"], hide_index=True)
    if issues:
        st.warning("Nonconverged fits: " + ", ".join(issues))
    st.download_button(
        "Download leaderboard", table.to_csv(index=False), "leaderboard.csv", "text/csv"
    )
