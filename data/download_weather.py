"""Download public fixed-lead weather forecasts; no credentials or scraping."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.weather import download_weather, weather_settings
from data.connectors import ROOT
import json


def main():
    cfg = weather_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", choices=list(cfg["zones"]), default="DE_LU")
    parser.add_argument("--all-zones", action="store_true")
    parser.add_argument("--start", default=cfg["default_start"])
    parser.add_argument("--end", default=cfg["default_end"])
    args = parser.parse_args()
    for zone in cfg["zones"] if args.all_zones else [args.zone]:
        frame, metadata = download_weather(zone, args.start, args.end)
        report = ROOT / "reports/weather"
        report.mkdir(parents=True, exist_ok=True)
        (report / f"{zone}_provenance.json").write_text(json.dumps(metadata, indent=2))
        print(f"{zone}: {len(frame)} complete hours, {metadata['missing_hours']} omitted; no filling.", flush=True)


if __name__ == "__main__":
    main()
