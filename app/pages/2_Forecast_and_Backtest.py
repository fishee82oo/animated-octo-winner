import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.common import setup, MODELS, evaluate, forecast_chart

province, p, df = setup("Forecast & Backtest")
model = st.selectbox("Model", list(MODELS), index=3)
days = st.slider("Evaluation days", 1, 14, 3)
st.caption(
    "Historical replay: each midnight origin forecasts the next 24 hours. Training uses the preceding 90 days. Future load, wind and solar inputs are noisy synthetic forecasts; actual target prices are excluded."
)
if st.button("Run backtest", type="primary"):
    pred, metrics, diagnostics = evaluate(province, model, days)
    st.plotly_chart(forecast_chart(pred), width="stretch")
    st.dataframe(metrics, hide_index=True)
    if "p10" in pred:
        coverage = metrics.iloc[0].coverage_80
        st.info(
            f"Observed P10–P90 coverage: {coverage:.1%}; nominal target 80%. Quantiles are not automatically calibrated."
        )
    if any(d.get("converged") is False for d in diagnostics):
        st.warning("One or more SARIMAX fits did not converge. Inspect diagnostics.")
    with st.expander("Fit diagnostics"):
        st.json(diagnostics)
    st.download_button(
        "Download forecasts", pred.to_csv(index=False), "forecasts.csv", "text/csv"
    )
