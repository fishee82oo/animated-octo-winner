# Validation record

Environment: Python 3.11.16, Apple Silicon. Exact Python package versions are in `requirements-lock.txt`.

- Phase 1: scaffold and five distinct parameter profiles checked.
- Phase 2: required CLI produced 26,280 rows. Synthetic-pattern assertions passed. Plotly report generated; charts rendered in the dashboard and executed notebook. Midday average 204.8 versus evening 469.7 CNY/MWh; Sichuan wet/dry hydro ratio 2.77; Gansu/Mengxi overnight wind/price correlation −0.73. Shandong has six midday hours at or below 20 CNY/MWh over three years.
- Phase 3: origin-based lag/rolling, target perturbation, coal boundary and prefix-stability tests passed.
- Phase 4: persistence, seasonal naive and SARIMAX ran through daily evaluation. SARIMAX converged on the smoke-test origins.
- Phase 5: LightGBM point forecast ran; MAE 37.38 versus seasonal naive 51.05 on the shared 72-hour summer smoke test.
- Phase 6: ordered quantiles, pinball, spike metrics and train-only threshold checks passed. P50 MAE 35.28; interval coverage 66.7% versus nominal 80%.
- Phase 7: settlement identity, zero/full hedge, negative prices, seller loss sign, tail ties/fractional mass and scenario reproducibility tests passed.
- Phase 8: Streamlit AppTest loaded Home and all four pages; both GBM UI runs and the all-model leaderboard passed. Live server health returned `ok`; browser displayed the data and charts.
- Phase 9: all 29 pytest tests passed; compileall succeeded. Six notebook code cells executed top to bottom with no notebook error outputs. A macOS sandbox restriction emitted an ipykernel child-process enumeration warning during shutdown, after execution; the runner exited successfully and saved every cell's outputs.

The supplied environment resolves macOS OpenMP using the bundled scikit-learn OpenMP library copied into the workspace-local Python runtime. A fresh machine should install the platform OpenMP runtime as explained in README.

Tests establish internal mechanics, not numerical agreement with real markets, real ex-ante forecast availability, quantile calibration, or suitability for financial decisions.
