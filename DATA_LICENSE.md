# PriceFM dataset attribution

Source: Runyao Yu and coauthors, **PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting**.

- Dataset: https://huggingface.co/datasets/RunyaoYu/PriceFM
- Source repository: https://github.com/runyao-yu/PriceFM
- Paper: https://arxiv.org/abs/2508.04875
- Dataset license as declared by its publisher: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

This project downloads a pinned version of the data. Changes made here: aggregate complete quarter-hours to hourly arithmetic means, omit incomplete endpoint hours, select source price and forecast covariates, rename fields, compute features, and produce research summaries and forecasts. No endorsement by PriceFM's authors is implied.

This attribution applies to source data and derivatives; it does not assert a license for upstream model code or weights. Neither is copied into this project.

## Weather attribution and API terms

Weather forecasts: **Open-Meteo.com**, using the **Deutscher Wetterdienst (DWD) ICON Global** model. Data license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Archive documentation: https://open-meteo.com/en/docs/previous-runs-api
- Provider: https://www.dwd.de/
- API terms: https://open-meteo.com/en/terms
- Commercial access: https://open-meteo.com/en/pricing

Transformations: request fixed 48-hour-lead variables in UTC with m/s wind units; average configured geographic sites equally; derive site-level heating/cooling degrees and regional temperature spread; exclude hours missing any site/variable; assign a documented availability-delay assumption; align with hourly PriceFM prices. Per-request URLs, retrieval times, checksums, site coordinates and missing-hour counts are recorded in provenance manifests. No provider endorsement is implied.

The public API is restricted to non-commercial use, separately from the data license. In particular, the provider classifies undisclosed research at commercial entities as commercial. This repository uses the public endpoint for the user's educational prototype; deployment or research inside a commercial fund requires appropriately licensed access. No paid account is created and no API key is stored. The availability fields we derive are assumptions, not verified publication timestamps.
