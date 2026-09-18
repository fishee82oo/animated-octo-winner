# Weather integration — results and validation

Public source: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), DWD ICON Global. Weather profiles and caches cover all 38 PriceFM zones, April 2024–December 2025: 15,360 UTC hours each, 583,680 zone-hours total, no missing input hours in this downloaded interval. Geographic sites are illustrative equal-weight proxies.

## Matched Germany–Luxembourg evaluation

Four 30-day evaluation windows end 2025-03-31, 2025-06-30, 2025-09-30 and 2025-12-31: 120 local delivery days and 2,879 hours. Each fit uses a 90-local-day sliding window. Identical training-row masks and target timestamps were asserted across all arms. Fixed model hyperparameters; no tuning on evaluation periods.

| Model | PriceFM forecasts | Weather | MAE EUR/MWh | RMSE EUR/MWh | P10–P90 coverage |
|---|---|---|---:|---:|---:|
| LightGBM | False | False | 21.71 | 31.58 | — |
| LightGBM | False | True | 18.43 | 26.61 | — |
| LightGBM | True | False | 12.16 | 18.93 | — |
| LightGBM | True | True | 11.95 | 18.65 | — |
| Persistence | False | False | 33.11 | 48.85 | — |
| Quantile LightGBM | False | False | 21.04 | 31.64 | 56.4% |
| Quantile LightGBM | False | True | 18.12 | 27.56 | 59.6% |
| Quantile LightGBM | True | False | 11.63 | 18.74 | 61.6% |
| Quantile LightGBM | True | True | 11.44 | 18.65 | 61.6% |
| SARIMAX | False | False | 29.12 | 42.37 | — |
| SARIMAX | False | True | 22.74 | 32.01 | — |
| Weekly seasonal | False | False | 32.50 | 47.49 | — |

Adding weather to price/calendar LightGBM reduces MAE from 21.71 to 18.43 EUR/MWh (15.1%). With PriceFM load/wind/solar covariates already present, incremental reduction is smaller: 12.16 to 11.95 (1.7%). Effects differ by seasonal window; no universal improvement is claimed. The historical three-day 5.93 result uses a different evaluation sample.

Quantile price/calendar plus weather has pinball losses P10=4.70, P50=9.06 and P90=4.89; nominal 80% interval coverage is only 59.6%. The bands are under-calibrated. This is not a validated VaR/CVaR risk model or a tradable alpha/P&L result.

## Timing and source limits

Each valid hour uses fixed 48-hour-lead archive data. Reference time is a nominal bound; availability is reference plus an assumed eight-hour delay. The as-of check runs against each historical sample’s own D−1 11:00 Brussels origin, including DST. These timestamps are not observed publication metadata. Source forecasts may use different runs across the target day. PriceFM supply/demand forecast vintages are also unavailable. Upstream PriceFM processing remains.

Temperature, wind at 100 m, cloud cover, instantaneous radiation and preceding-hour precipitation are used directly. Heating/cooling proxies and cross-site temperature spread are derived. No target-day reanalysis or synthetic weather substitutes are used. Missing data would remove an hour rather than be filled.

## Validation

- `python -m pytest -q`: 56 passed, including the pre-existing duplicate local test files. Tests cover weather units, missing sites, checksum corruption, timing metadata, DST, target-price isolation, matched ablation rows and weather-enabled Streamlit model runs.
- Notebook: seven code cells executed top to bottom without cell errors. The local sandbox emitted a harmless IPython temporary-home notice and a process-enumeration warning during kernel shutdown; the runner exited successfully.
- All 38 weather caches reloaded and verified; timing guards and complete weather feature coverage passed.
- Streamlit launched at http://127.0.0.1:8503. Automated tests exercise all four main pages and Home, both LightGBM model actions and the weather comparison action.
- Two SARIMAX fits reported nonconvergence; scores are retained transparently, not silently replaced: [('2025-03-29', False), ('2025-06-13', False)].

## Files

Additional real-cache smoke checks passed for Home, Data Explorer, Hedging Simulator, and both LightGBM forecast actions. The running server returned health `ok` during validation.

- [Overall and delivery-period metrics](DE_LU_summary.csv)
- [Metrics by seasonal window](DE_LU_by_window.csv)
- [Hourly predictions](DE_LU_predictions.csv)
- [Run configuration, provenance and diagnostics](DE_LU_benchmark.json)
- [38-zone coverage audit](coverage.csv)
- [Interactive model comparison](DE_LU_comparison.html)
- [Weather week plot](DE_LU_weather_week.html)

## Reproduce

```sh
python data/download_dataset.py
python data/download_weather.py --all-zones
python scripts/weather_benchmark.py --zone DE_LU
python scripts/run_notebook.py
python -m pytest -q
streamlit run app/Home.py
```

Weather data: CC BY 4.0, Open-Meteo / DWD attribution. The public API is non-commercial; use appropriate commercial access for research inside a fund. No trading connections, orders or real-money execution were added.
