"""Download and verify the pinned CC BY 4.0 PriceFM dataset, then cache all zones."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from datetime import datetime, timezone
import hashlib
import json
from urllib.request import urlopen
import pandas as pd
from data.connectors import ROOT, settings, markets, hourly_zone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(raw_path=None, cache_dir=None):
    cfg = settings()
    raw_path = Path(raw_path) if raw_path else ROOT / "data/raw/FINAL.csv"
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw_path.with_suffix(".part")
        with urlopen(cfg["url"], timeout=60) as response, tmp.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if digest(tmp) != cfg["sha256"]:
            tmp.unlink()
            raise ValueError("Downloaded source checksum mismatch")
        tmp.replace(raw_path)
    if digest(raw_path) != cfg["sha256"]:
        raise ValueError("Source checksum mismatch; refusing a different dataset")
    raw = pd.read_csv(raw_path)
    cache = Path(cache_dir) if cache_dir else ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    summary = []
    checksums = {}
    for zone, name in markets().items():
        df, dropped = hourly_zone(raw, zone)
        path = cache / f"{zone}.csv"
        df.to_csv(path, index=False)
        checksums[zone] = digest(path)
        summary.append(
            dict(
                zone=zone,
                name=name,
                rows=len(df),
                start=str(df.timestamp.min()),
                end=str(df.timestamp.max()),
                min_price=df.price.min(),
                max_price=df.price.max(),
                negative_hours=int((df.price < 0).sum()),
                dropped_partial_hours=dropped,
                zero_solar=bool(df.solar_forecast_mw.eq(0).all()),
                zero_wind=bool(df.wind_forecast_mw.eq(0).all()),
            )
        )
    provenance = dict(
        provider="Runyao Yu et al., PriceFM",
        **cfg,
        raw_rows=len(raw),
        raw_columns=len(raw.columns),
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        cache_sha256=checksums,
        transformation="Hourly arithmetic means of four complete quarter-hours; endpoint partial hours omitted; no interpolation, clipping, scaling or synthetic filling added.",
        caveat="Upstream data is resampled/interpolated and may contain zero placeholders. Per-value flags and issue timestamps are absent. Not live raw exchange data.",
    )
    (cache / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (ROOT / "reports").mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(ROOT / "reports/market_catalog.csv", index=False)
    (ROOT / "reports/data_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    args = p.parse_args()
    summary = prepare(args.input)
    print(
        f"Prepared {len(summary)} zones, {summary.rows.iloc[0]:,} hourly records per zone."
    )
    print(
        summary[["zone", "rows", "negative_hours", "dropped_partial_hours"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
