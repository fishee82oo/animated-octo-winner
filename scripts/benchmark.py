"""Small reproducible daily backtest; flags broaden coverage when desired."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from data.connectors import SyntheticConnector
from data.simulator import ROOT
from models.baselines import Naive, SeasonalNaive, Sarimax
from backtest.walk_forward import walk_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gbm", action="store_true")
    p.add_argument("--quantile", action="store_true")
    p.add_argument("--days", type=int, default=3)
    args = p.parse_args()
    models = {"Naive": Naive, "Seasonal naive": SeasonalNaive, "SARIMAX": Sarimax}
    if args.gbm:
        from models.gbm import GBM

        models["LightGBM"] = GBM
    if args.quantile:
        from models.quantile import QuantileGBM

        models["Quantile LightGBM"] = QuantileGBM
    frame = SyntheticConnector(years=0.5).load()
    rows = []
    for name, factory in models.items():
        pred, metrics, diagnostics = walk_forward(
            frame, factory, days=args.days, min_train_days=30, window_days=90
        )
        metrics["model"] = name
        rows.append(metrics)
        print(name, metrics.iloc[0].to_dict(), diagnostics[-1], flush=True)
        pred.to_csv(
            ROOT / "reports" / f"{name.lower().replace(' ', '_')}_predictions.csv",
            index=False,
        )
    pd.concat(rows, ignore_index=True).to_csv(
        ROOT / "reports/benchmark.csv", index=False
    )


if __name__ == "__main__":
    main()
