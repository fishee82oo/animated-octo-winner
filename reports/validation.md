# Europe restart validation

## Weather extension validation — 2026-09-18

- 56 pytest tests passed, including the pre-existing duplicate local test files. New tests cover weather aggregation, units, missing-site rejection, timing assumptions, checksum corruption, DST, target-price isolation, paired samples and weather-enabled UI actions.
- Downloaded and verified all 38 weather zone caches: 15,360 hourly rows each, 583,680 zone-hours, 2024-04-01 through 2025-12-31 UTC, zero missing hours in this interval. `reports/weather/coverage.csv` records the audit.
- Four paired seasonal 30-day DE_LU windows completed for point/quantile LightGBM, SARIMAX and naive controls, with identical training eligibility and test timestamps. Price/calendar LightGBM MAE: 21.71 without weather vs 18.43 with weather. Two price-only SARIMAX fits did not converge; full diagnostics retained.
- Notebook rebuilt and all seven code cells executed without cell errors; the previously observed sandbox kernel-shutdown warning remains non-fatal.
- Streamlit launched on localhost:8503 and returned health `ok`. Real-cache AppTest smoke checks passed for Home, Data Explorer, Hedging Simulator, and both LightGBM forecast actions. Offline UI tests also cover the leaderboard and paired weather comparison.
- README, data attribution, provenance manifests and `reports/weather/RESULTS.md` explain publication-time assumptions, geographic proxies, API usage terms, seasonal results and under-calibrated prediction intervals.
- Original local duplicate files were preserved and excluded from the weather update.

## Original price-only restart checks

Python 3.11.16; exact versions in requirements-lock.txt.

- PriceFM source SHA-256 verified against config/dataset.yaml. 140,257 source rows; 38 zone caches, each with 35,064 hourly rows. One partial endpoint hour omitted per zone. No source missing values; upstream interpolation/zero placeholders remain.
- 33 pytest tests passed. Includes duplicate/gap rejection, source and cache integrity, negative-price preservation, local-day price-lag leakage checks, 23/25-hour horizons, and a 719-hour 30-local-day training window.
- All five models completed on DE_LU actual dataset (2025-12-29 through 2025-12-31); SARIMAX converged on the smoke-test origins. Metrics in DE_LU_benchmark.csv.
- Additional LightGBM smoke tests completed for FR, ES and NL; results in cross_zone_smoke.csv.
- Actual DE_LU data produced 23 forecast hours on 2025-03-30 and 25 on 2025-10-26.
- Streamlit AppTest loaded every page, ran both LightGBM UI workflows, and computed the five-model leaderboard using offline toy fixtures. Toy data is isolated to tests.
- Live Europe dashboard on localhost:8502 returned health `ok`; browser verified the actual 38-zone source inventory and rendered historical price chart.
- The exploration notebook executed all six code cells and saved output without cell errors. Kernel shutdown emits a macOS sandbox child-process enumeration warning after execution; the runner exits successfully.

These checks establish pipeline mechanics, not raw-exchange accuracy, forecast vintage availability, PriceFM neural-model equivalence, or interval calibration. Nominal 80% interval coverage on the three-day DE_LU example is 63.9%.
