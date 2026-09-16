# china-power-spot-forecast

Python 3.11 research and educational prototype for next-day electricity spot-price forecasts and medium/long-term contract hedging in illustrative Chinese provincial markets.

**All price data is synthetic, generated from hand-chosen parameters. It is not live market data and has not been calibrated against observed provincial prices. There are no broker connections, real-money facilities, automated orders, exchange integrations, or portal scrapers.** Forecasts and hedge suggestions are scenario exercises, not financial recommendations.

## Quick start

Run from this repository's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/generate_dataset.py --province shandong --years 3 --seed 42
python -m pytest -q
streamlit run app/Home.py
```

Open http://localhost:8501. The generated Shandong CSV is loaded by the dashboard; other province selections use the same three-year, seed-42 simulator. On Apple Silicon, LightGBM requires OpenMP: install `libomp` with your package manager (for example `brew install libomp`) if its import reports a missing `libomp.dylib`. The supplied local environment has already been configured and tested with Python 3.11.16. Linux environments may need their distribution's OpenMP runtime (`libgomp1`).

The existing environment can be used immediately with `.venv/bin/python` and `.venv/bin/streamlit`. To reproduce the exact tested package versions, install `requirements-lock.txt` instead of the version ranges in `requirements.txt`. The environment itself is local and should be recreated when moving the repository.

```sh
python data/check_patterns.py
python scripts/benchmark.py --gbm --quantile --days 3
python scripts/run_notebook.py
```

The first command saves a self-contained Plotly report of a summer week, daily duck curve, seasonal load and Sichuan hydro. Add `--show` to open it in your browser. The benchmark writes forecasts and an overall/by-TOU metrics table to `reports/`. The notebook runner executes `notebooks/01_exploration.ipynb` top to bottom and saves its outputs in place; an interactive Jupyter frontend is optional.

## Market assumptions

The project encodes the task's two-layer market abstraction: medium/long-term bilateral or listed contracts typically account for roughly 80–95% of volume; spot prices determine dispatch and settle deviations around the contracted position. Contract prices are benchmark-plus-floating-band assumptions. Configured default contract share is 90%, with an illustrative ±20% band. The hedge simulator accepts any contract price to support sensitivity studies; the band is a reference, not an enforced legal constraint.

This abstraction simplifies actual province-specific settlement rules. Historical pilot context supplied for the project includes Guangdong, Mengxi, Zhejiang, Shanxi, Shandong, Fujian, Sichuan and Gansu in batch one, with Shanghai, Jiangsu, Anhui, Hubei and Liaoning among batch-two markets. This is context, not a current operational-status registry. Actual floors, caps, settlement rules and TOU hours vary by place and date; the YAML values are deliberately illustrative. Caps need not be a fixed multiple of the benchmark. This project models hourly day-ahead prices, not separate 5–15-minute real-time clearing, transmission constraints, or nodal prices.

| Profile key | Main synthetic dynamics |
|---|---|
| `shandong` | High solar, coal-sensitive base, summer load peak, low midday prices |
| `gansu_mengxi` | Combined illustrative wind-heavy profile, low overnight prices and curtailment events |
| `sichuan` | Hydro-dominated with abundant June–October hydro and tighter dry season |
| `guangdong` | Large industrial load and strong summer peak; hydro contribution is aggregated, nuclear/import schedules are not separately modeled |
| `shanxi` | Coal-heavy price response and winter heating load |

The coal index is an arbitrary synthetic index centered at 800, not a downloaded Qinhuangdao or Bohai Rim series. Load includes daily, weekly, summer and winter components; solar has daylight and cloud variation; wind is bounded mean-reverting noise with a seasonal mean and no forced daily cycle. Coal follows a slow stochastic process with scarcity jumps. Random heatwaves, winter cold snaps and high-wind curtailment days create extremes. Price is the clipped sum of benchmark, coal/load effects, renewable/hydro effects, TOU steps, events and noise. TOU buckets are simplified fixed hours, shared across profiles.

## Data and connectors

`python data/generate_dataset.py --province shandong --years 3 --seed 42` produces `data/cache/shandong_3y_seed42.csv`, 26,280 hourly rows from 2021-01-01 through 2023-12-31. A simulator year means 365 days, including when another start date is used. Fractional years are supported, with at least 24 hours required. Prices use CNY/MWh, power uses MW, and timestamps are timezone-naive China local time (UTC+8).

Required CSV schema:

```text
timestamp,price,load_mw,wind_mw,solar_mw,hydro_mw,coal_index,tou_period
```

Allowed TOU values: `sharp_peak`, `peak`, `flat`, `valley`. `CsvConnector(path).load(start, end)` validates finite values, nonnegative physical output, unique contiguous hourly observations, and TOU labels. The interval is `[start, end)`. Input rows are sorted. Missing or duplicate hours fail explicitly; no hidden imputation occurs. Supply manually exported data through this adapter; no credentials or scraping are needed.

```python
from data.connectors import CsvConnector
from models.gbm import GBM
from backtest.walk_forward import walk_forward
frame = CsvConnector('my_hourly_export.csv').load()
predictions, metrics, diagnostics = walk_forward(frame, GBM, days=7)
```

**Real-data limitation:** the default feature builder creates noisy synthetic forecasts using future realized load/wind/solar. These are intentionally labeled simulated ex-ante forecasts for this research experiment. To claim real historical forecasting performance, replace these columns with archived forecasts actually issued before each origin. Merely replacing the CSV does not establish a realistic information set.

## Features and leakage controls

Each forecast origin is midnight, forecasting the next 24 hours as one batch. Models see only training labels strictly before the origin. Training rows use the same origin-based conventions as evaluation rows.

- Calendar: hour, weekday, month, holiday flag and four TOU one-hot columns. Default holiday flag marks New Year only; pass explicit dates via `holidays=` for a researched Chinese holiday calendar. Rescheduled working weekends are not encoded.
- Price: hourly `t-24` and `t-168` lags. `price_lag_1` is frozen at the last observed pre-origin price for all 24 horizons, since actual intra-day prices would leak future information. At hour zero this is the ordinary `t-1` lag.
- Rolling 24/168-hour price mean and sample standard deviation use only observations before the origin and are frozen across the day.
- Last-known load/wind/solar/hydro and coal level/trend are frozen at the origin. Synthetic driver forecasts use multiplicative 12% standard-deviation noise, fixed independent random streams, and no fit to future sample statistics.
- The first eight days are warmup for the coal trend and weekly price history. Incomplete test days are excluded. Tests mutate all future target prices and verify unchanged forecast inputs, fitted quantiles and spike thresholds.

## Models and backtests

All models implement `fit(X, y)` / `predict(X)`. Persistence repeats the origin's last observed price; seasonal naive uses the same hour last week. SARIMAX uses a compact AR(1) with standardized exogenous drivers and TOU/calendar variables; daily structure is supplied through those regressors. Scaling is fit on training data only. Its convergence state is included in diagnostics. LightGBM provides point and separate 0.1/0.5/0.9 quantile objectives. Quantile outputs are sorted per row to remove crossing; this is not coverage calibration.

`walk_forward` supports expanding windows (`window_days=None`) or sliding windows, daily retraining, a minimum training-history requirement, and a selected number of final complete test days. The UI uses 90-day sliding windows and 1–14 test days. It replays historical origins; it does not claim to have real forecasts for tomorrow. GBM is direct multi-horizon with shared parameters and calendar/origin-safe lag features. No actual test-day price feeds back into later forecast hours.

Metrics are MAE, RMSE, WAPE (sum absolute errors divided by sum absolute actuals), three pinball losses, P10–P90 coverage/width, and spike precision/recall. Spikes exceed the training window's 90th price percentile, recomputed at each origin. Quantile spike classification uses P50. Overall and each TOU bucket are reported with observation counts; undefined WAPE or precision/recall is NaN rather than a misleading zero.

The included three-day summer smoke test (2,160 training rows per origin) produced:

| Model | MAE, CNY/MWh |
|---|---:|
| Persistence | 93.88 |
| Weekly seasonal naive | 51.05 |
| SARIMAX | 58.16 |
| LightGBM point | 37.38 |
| LightGBM quantile P50 | 35.28 |

The quantile interval covered 66.7% rather than the nominal 80%. These results demonstrate the pipeline, not stable rankings, calibrated uncertainty or market profitability. The UI uses the end of the full three-year dataset, so its results differ from this half-year smoke test. No hyperparameters were selected against this test slice. Broader multi-season evaluation and archived ex-ante inputs are prerequisites for further research.

## CfD hedging

For price `P`, metered energy `V`, contract price `C` and contracted energy `H = hV`:

```text
settlement = P V + (C − P) H = C H + P (V − H)
```

Settlement is buyer cost or seller gross revenue. Buyer loss is cost; seller loss is negative revenue. Production costs and fees are excluded. Price risk disappears at 100% contracted share only under fixed known volumes and a perfectly matching settlement reference. Volume, basis and counterparty risks remain outside this model.

P10/P50/P90 do not determine a complete distribution. The scenario model assumes a piecewise-linear inverse CDF through the province floor, the three quantiles and cap, and a Gaussian copula with adjustable hourly correlation (default 0.65). Out-of-bound model quantiles are clipped to the configured bounds. VaR is the empirical loss quantile and CVaR the average worst tail probability mass, with fractional boundary weighting so ties are handled correctly. Both full-settlement loss risk and unhedged-spot loss risk are reported. All amounts are CNY per selected settlement period, not percentages or incremental losses against a budget.

The heuristic minimizes `expected_loss + 4 * (1 − risk_tolerance) * (CVaR − expected_loss)` on a 0–100% grid. Higher tolerance emphasizes expected value. Endpoint suggestions are normal for this fixed-volume linear setup. Results depend strongly on imposed tails, dependence and uncertain forecasts.

## Dashboard and walkthrough

- Home: overview, province selector and illustrative parameters.
- Data Explorer: raw hourly data, generation/load, average daily price, load/price scatter and seasonality.
- Forecast & Backtest: choose any model; plot actual/forecast and quantile bands, with metrics and downloadable forecasts.
- Model Leaderboard: all five models on identical origins, overall and by TOU.
- Hedging Simulator: contract price, volume, buyer/seller side, hedge ratio, confidence, dependence and risk tolerance.

`notebooks/01_exploration.ipynb` loads the cached data, explores a summer week and duck curve, builds features, evaluates point and quantile forecasts against baselines, displays metrics and prediction bands, and computes a CfD sensitivity table. Charts use Plotly throughout.

## Repository map

`config/`: province parameters. `data/`: simulator, connector and generation CLI. `features/`: origin-safe features. `models/`: baseline, point and quantile estimators. `backtest/`: daily evaluation and metrics. `analytics/`: settlement and scenario risk. `app/`: Streamlit entry point and pages. `tests/`: data, leakage, models, metrics, hedging and dashboard tests. `scripts/`: benchmark and notebook execution. `reports/`: reproducible research outputs.

## Glossary

| Term | Meaning in this prototype |
|---|---|
| 中长期 | Medium/long-term financial contract layer |
| 现货市场 | Spot market |
| 日前 / 实时 | Day-ahead / real-time; only hourly day-ahead is modeled |
| 基准价 + 上下浮动比例 | Benchmark plus an allowed proportional band; illustrative here |
| 价差结算 | CfD adjustment between spot and contract prices |
| 尖峰 / 高峰 / 平段 / 低谷 | Sharp peak / peak / flat / valley TOU buckets |
| 弃风 / 弃光 | Wind / solar curtailment |
| 环渤海动力煤价格指数 | Bohai Rim thermal-coal index; inspiration only, not a data feed |

Deep learning is intentionally omitted: the complete requested non-neural pipeline is the maintained prototype.
