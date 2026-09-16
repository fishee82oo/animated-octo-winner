# Europe restart validation

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
