import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from data.connectors import ROOT, PriceFMConnector, markets
from models.baselines import Naive, SeasonalNaive, Sarimax
from models.gbm import GBM
from models.quantile import QuantileGBM
from backtest.walk_forward import walk_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zone", choices=list(markets()), default="DE_LU")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--price-only", action="store_true")
    args = p.parse_args()
    frame = PriceFMConnector(args.zone).load()
    rows = []
    for name, cls in {
        "Persistence": Naive,
        "Weekly seasonal": SeasonalNaive,
        "SARIMAX": Sarimax,
        "LightGBM": GBM,
        "Quantile LightGBM": QuantileGBM,
    }.items():
        pred, metrics, diag = walk_forward(
            frame,
            cls,
            days=args.days,
            end_date=args.end_date,
            use_exogenous=not args.price_only,
        )
        metrics["model"] = name
        metrics["zone"] = args.zone
        rows.append(metrics)
        print(name, metrics.iloc[0].to_dict(), diag[-1], flush=True)
        pred.to_csv(
            ROOT
            / "reports"
            / f"{args.zone}_{name.lower().replace(' ', '_')}_predictions.csv",
            index=False,
        )
    pd.concat(rows, ignore_index=True).to_csv(
        ROOT / "reports" / f"{args.zone}_benchmark.csv", index=False
    )


if __name__ == "__main__":
    main()
