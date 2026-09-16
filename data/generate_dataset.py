import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from data.connectors import SyntheticConnector
from data.simulator import ROOT, profiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hourly prices, CNY/MWh"
    )
    parser.add_argument("--province", choices=list(profiles()), default="shandong")
    parser.add_argument("--years", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = SyntheticConnector(args.province, args.years, args.seed).load()
    target = (
        ROOT / "data/cache" / f"{args.province}_{args.years:g}y_seed{args.seed}.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print(f"{len(df):,} rows saved to {target}")


if __name__ == "__main__":
    main()
