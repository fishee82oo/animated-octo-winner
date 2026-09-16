import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from data.connectors import hourly_zone, validate, PriceFMConnector, settings, markets


def native():
    return pd.DataFrame(
        {
            "time_utc": pd.date_range("2025-01-01", periods=9, freq="15min", tz="UTC"),
            "DE_LU-price": [-20, 0, 20, 40, 100, 100, 100, 100, 999],
            "DE_LU-load": 1000,
            "DE_LU-solar": 0,
            "DE_LU-wind": 100,
        }
    )


def test_complete_hour_aggregation_and_partial_endpoint():
    df, dropped = hourly_zone(native(), "DE_LU")
    assert dropped == 1 and df.price.tolist() == [10.0, 100.0]
    assert str(df.timestamp.dt.tz) == "UTC"


def test_source_gaps_and_duplicates_fail():
    a = native()
    with pytest.raises(ValueError):
        hourly_zone(a.drop(index=2), "DE_LU")
    with pytest.raises(ValueError):
        hourly_zone(pd.concat([a, a.iloc[:1]]), "DE_LU")
    a.loc[0, "DE_LU-price"] = np.nan
    with pytest.raises(ValueError):
        hourly_zone(a, "DE_LU")


def test_negative_prices_preserved(hourly):
    hourly.loc[0, "price"] = -500
    assert validate(hourly).price.iloc[0] == -500


def test_cached_connector_integrity_and_slice(tmp_path, hourly):
    path = tmp_path / "DE_LU.csv"
    hourly.to_csv(path, index=False)
    meta = dict(
        revision=settings()["revision"],
        cache_sha256={"DE_LU": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    (tmp_path / "provenance.json").write_text(json.dumps(meta))
    assert (
        len(PriceFMConnector(cache_dir=tmp_path).load("2024-01-02", "2024-01-03")) == 24
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        PriceFMConnector(cache_dir=tmp_path).load()


def test_source_catalog():
    assert len(markets()) == 38 and "DE_LU" in markets()
    assert len(settings()["revision"]) == 40 and len(settings()["sha256"]) == 64
