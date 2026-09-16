import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.common import setup, MODELS, controls, evaluate, forecast_chart

zone, df = setup("Forecast & Backtest")
model = st.selectbox("Model", list(MODELS), index=3)
days, end, exog = controls(df)
if st.button("Run backtest", type="primary"):
    try:
        pred, metrics, diag = evaluate(zone, model, days, end, exog)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.plotly_chart(forecast_chart(pred), width="stretch")
    st.dataframe(metrics, hide_index=True)
    if "coverage_80" in metrics:
        st.info(
            f"Observed P10–P90 coverage: {metrics.iloc[0].coverage_80:.1%}; nominal 80%, not automatically calibrated."
        )
    if any(d.get("converged") is False for d in diag):
        st.warning("At least one SARIMAX fit did not converge.")
    with st.expander("Origins, horizon lengths and fit diagnostics"):
        st.json(diag)
    st.download_button(
        "Download forecasts", pred.to_csv(index=False), "forecasts.csv", "text/csv"
    )
