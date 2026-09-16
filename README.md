# Europe Power Spot Forecast

A Python 3.11 research and educational project for European **day-ahead electricity-price forecasting** and illustrative fixed-price hedging. The primary data source is [PriceFM](https://github.com/runyao-yu/PriceFM), published by Runyao Yu and coauthors. This project uses its dataset, not its TensorFlow model weights, architecture or reported performance.

**No brokers, real money, automated orders or execution facilities. No live data feed.**

## Start here

From this repository's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/download_dataset.py
python -m pytest -q
streamlit run app/Home.py
```

The download is approximately 195 MB. No Hugging Face account or token is required for the pinned public file. Download once; dashboard operation and tests require no network. On macOS, LightGBM needs OpenMP (`brew install libomp` if its import reports a missing library). Recreate the environment when moving the repository. `requirements-lock.txt` records the tested package versions.

```sh
python scripts/benchmark.py --zone DE_LU --days 3 --end-date 2025-12-31
python scripts/benchmark.py --zone FR --days 7 --price-only
python scripts/run_notebook.py
```

`notebooks/01_exploration.ipynb` is an executable walkthrough with data inspection, forecasts, metrics and hedge scenarios. The supplied workspace has already downloaded and validated the data. Tests use labeled toy fixtures for isolation; those fixtures are never connected to the app.

## Data source and provenance

- Dataset: [RunyaoYu/PriceFM on Hugging Face](https://huggingface.co/datasets/RunyaoYu/PriceFM), `FINAL.csv`.
- Pinned revision: `68b032a923e518bcf88edde906035ea223870aee`.
- SHA-256: `98f596deba7ffaf0edd21e78e1a779256ab24dda5463d445f081e1ee4ab3a54a`.
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: Runyao Yu et al., *PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting*, [arXiv:2508.04875](https://arxiv.org/abs/2508.04875).
- Source file inspected: 140,257 quarter-hour timestamps, 191 columns and 38 zones. UTC timestamps run from 2021-12-31 23:00 to 2025-12-31 23:00, inclusive.
- Our transformation: arithmetic hourly means of four complete quarter-hours. The final partial hour is omitted. This gives **35,064 hourly observations per zone**, covering delivery dates 2022-01-01 through 2025-12-31 on the common Europe/Brussels clock.

The upstream research dataset includes resampling, interpolation and zero-filled unavailable features. It is not untouched exchange data. See the source paper's data section for upstream processing. There are no per-value imputation flags or forecast publication vintages in the supplied CSV, so this project cannot undo that processing or establish the exact auction-time availability of each value. Results are retrospective research benchmarks.

Our reader selects only `{zone}-price`, `{zone}-load`, `{zone}-solar`, and `{zone}-wind`, following PriceFM's forecast-variable definitions. The additional `generation` columns are excluded because this implementation does not establish their ex-ante availability. Values and units are retained without price clipping, synthetic noise, or added interpolation. An all-zero forecast feature may be an upstream placeholder rather than verified absence of generation.

`config/dataset.yaml` pins the source; `reports/data_provenance.json` records transformations and checksums; `reports/market_catalog.csv` records coverage, negative hours and all-zero generation-forecast series by zone. `data/raw/` and `data/cache/` are ignored by Git; reproducible download code and provenance are committed. Each loaded zone CSV is checked against its cache checksum.

## Scope and units

Select any of the 38 source zones, including Germany–Luxembourg (`DE_LU`), France (`FR`), Spain (`ES`), Netherlands (`NL`), Denmark's two zones, Norway's five zones, Sweden's four zones, and Italy's seven zones. They are bidding-zone series, not interchangeable national averages. Labels are in `config/markets.yaml`.

Prices are EUR/MWh, source load/wind/solar forecasts are MW, scenario volumes are MWh per hourly interval, and settlement totals are EUR. An hourly average price is an equal-duration research target; it does not preserve every quarter-hour price spike or reproduce a bill with varying quarter-hour energy.

The project no longer applies Chinese benchmark tariffs, provincial price caps, TOU steps, coal-index simulation, or mandatory contract-share assumptions. Negative European price observations are preserved. There is no real-time, intraday or imbalance-price model.

## Forecast information set and time

Storage is timezone-aware UTC. Delivery dates, calendar features and research periods use **Europe/Brussels for every zone**, a common comparison clock rather than each country's civil time. Spring and autumn delivery days contain 23 and 25 hourly observations. Both repeated autumn hours remain distinct in storage, forecasts and settlement volumes.

The assumed issue time is D−1 at 11:00 Brussels, before the usual noon day-ahead auction for D. We use D−1 delivery prices, already published at the prior auction, including D−1 evening delivery prices. These are known day-ahead prices, not unobserved realized spot prices. Target-day price labels are excluded from feature construction and fitting.

Features comprise local clock/calendar variables, UTC offset, same wall-hour prices from one/seven local days ago, previous delivery day's final known price, and trailing 24/168-hour price statistics frozen before D delivery. Prior-day price features reference the previous local date, avoiding the target-label leak a naive `shift(24)` can cause on a 25-hour day. A duplicated reference hour is averaged; a spring-missing reference hour uses that earlier day's mean. There is no holiday-calendar claim.

Optionally include PriceFM's target-day load/wind/solar forecasts and their net-load combination. This assumes the source forecast values were available by the issue time; **the dataset lacks vintages to verify that assumption**. Disable these features in the UI or use `--price-only` for a comparison without forecast covariates. Upstream interpolation limitations still apply to prices themselves. This is not a fully audited historical auction replay.

## Models, evaluation and metrics

Models share `fit(X, y)` / `predict(X)`:

- Persistence: previous delivery day's final known price.
- Weekly seasonal naive: same wall hour one week earlier, with the DST reference rules above.
- SARIMAX: compact AR(1) plus calendar and available forecast regressors; scaling fit only on the training window, convergence surfaced in diagnostics.
- LightGBM point forecast.
- LightGBM quantile forecasts: P10/P50/P90, monotonically sorted per row to remove crossings.

Each delivery day is held out as one batch. Fits use a 90-local-day sliding window by default; expanding windows are available with `window_days=None`. Horizons can be 23, 24 or 25 hours. Rows with insufficient lag history are excluded before fitting. Metrics include MAE, RMSE, WAPE, pinball loss, interval coverage/width, and precision/recall for prices above the training window's 90th percentile. Undefined denominators return NaN.

Metrics are overall and by research delivery groups: weekday 08:00–20:00, other weekday hours, and weekend. These groups are not regulated tariffs or exact exchange-contract definitions.

A three-day DE_LU smoke test ending 2025-12-31, with PriceFM forecast inputs enabled, produced MAE of 9.05 (persistence), 13.54 (weekly), 13.86 (SARIMAX), 5.93 (LightGBM point), and 6.94 EUR/MWh (quantile P50). The nominal 80% interval covered 63.9%. This small slice verifies execution, not stable rankings, calibrated risk, or PriceFM paper reproduction. Reports are saved under `reports/`.

## Hedge scenarios

For price P, energy V, fixed price C and matched contracted energy H = hV:

```text
buyer cost / seller gross revenue = P V + (C − P) H
```

This is an illustrative fixed-for-floating hedge indexed to the modeled day-ahead price. It is not a claim about a specific European regulated CfD or an exchange product. Buyer loss is cost; seller loss is negative gross revenue. Generation costs, fees, margining, basis, volume and counterparty risks are outside scope.

P10/P50/P90 are converted into an assumed piecewise-linear inverse CDF with editable lower/upper stress bounds and Gaussian hourly dependence. Those bounds are **scenario assumptions, not exchange limits**; forecast quantiles outside them are clipped. VaR uses the empirical loss quantile; CVaR uses exact worst-tail probability mass with fractional boundary weighting. The heuristic minimizes expected loss plus `4 × (1 − risk_tolerance) × (CVaR − expected_loss)` over contract shares. Full hedging removes only matched fixed-volume price risk under this simplified setup. No orders are generated.

## Dashboard

`Home` provides scope and provenance. `Data Explorer` shows source prices, forecast drivers, hourly profiles and cross-zone comparisons. `Forecast & Backtest` runs every model and displays quantile bands. `Model Leaderboard` compares identical dates and feature choices. `Hedging Simulator` exposes contract price, volume, share, risk tolerance, confidence, dependence and stress bounds.

All charts use Plotly. No scraping or account connections are needed. The historical China prototype remains recoverable in Git history; this Europe restart replaces its active code and documentation. The connected GitHub repository URL is unchanged.
