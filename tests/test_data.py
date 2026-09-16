import numpy as np
import pandas as pd
import pytest
from data.simulator import simulate, profiles
from data.connectors import CsvConnector, validate


@pytest.mark.parametrize("province", list(profiles()))
def test_simulator_profiles(province):
    df = simulate(province, years=0.1)
    p = profiles()[province]
    pd.testing.assert_frame_equal(df, simulate(province, years=0.1))
    assert df.price.between(p["price_floor"], p["price_cap"]).all()
    assert df.loc[df.timestamp.dt.hour < 6, "solar_mw"].eq(0).all()
    assert np.isfinite(df.select_dtypes("number")).all().all()


def test_csv_validation_and_exclusive_end(tmp_path):
    df = simulate(years=0.02)
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    got = CsvConnector(path).load("2021-01-02", "2021-01-03")
    assert len(got) == 24
    with pytest.raises(ValueError):
        validate(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError):
        validate(df.drop(index=5))
    with pytest.raises(ValueError):
        validate(df.drop(columns="price"))
