from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
import app.common as common

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_app_data(monkeypatch, hourly):
    monkeypatch.setattr(common, "dataset", lambda zone: hourly.copy())
    monkeypatch.setattr(common, "weather_available", lambda zone: False)
    common.evaluate.clear()


@pytest.mark.parametrize(
    "page",
    [
        "Home.py",
        "pages/1_Data_Explorer.py",
        "pages/2_Forecast_and_Backtest.py",
        "pages/3_Model_Leaderboard.py",
        "pages/4_Hedging_Simulator.py",
    ],
)
def test_pages_load(page):
    app = AppTest.from_file(str(ROOT / "app" / page), default_timeout=60).run()
    assert not app.exception


@pytest.mark.parametrize("model", ["LightGBM point", "LightGBM quantile"])
def test_ui_model_runs(model):
    app = AppTest.from_file(
        str(ROOT / "app/pages/2_Forecast_and_Backtest.py"), default_timeout=60
    ).run()
    app.selectbox[0].select(model).run()
    app.button[0].click().run()
    assert not app.exception and len(app.dataframe) == 1


def test_ui_leaderboard():
    app = AppTest.from_file(
        str(ROOT / "app/pages/3_Model_Leaderboard.py"), default_timeout=60
    ).run()
    app.button[0].click().run()
    assert not app.exception and len(app.dataframe[0].value) == 5


@pytest.mark.parametrize("model", ["LightGBM point", "LightGBM quantile"])
def test_weather_ui_models(monkeypatch, weather_hourly, model):
    monkeypatch.setattr(common, "weather_available", lambda zone: True)
    monkeypatch.setattr(common, "weather_dataset", lambda zone: weather_hourly.copy())
    app = AppTest.from_file(str(ROOT / "app/pages/2_Forecast_and_Backtest.py"), default_timeout=60).run()
    app.selectbox[0].select(model).run()
    app.button[0].click().run()
    assert not app.exception and len(app.dataframe) == 1


def test_weather_ui_ablation(monkeypatch, weather_hourly):
    monkeypatch.setattr(common, "weather_available", lambda zone: True)
    monkeypatch.setattr(common, "weather_dataset", lambda zone: weather_hourly.copy())
    app = AppTest.from_file(str(ROOT / "app/pages/3_Model_Leaderboard.py"), default_timeout=60).run()
    app.button[1].click().run()
    assert not app.exception
    table = app.dataframe[0].value
    assert set(table.weather) == {True, False}
    assert len(table[table.delivery_period == "overall"]) == 4
