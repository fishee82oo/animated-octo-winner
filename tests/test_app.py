from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


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
def test_ui_runs_models(model):
    app = AppTest.from_file(
        str(ROOT / "app/pages/2_Forecast_and_Backtest.py"), default_timeout=60
    ).run()
    app.selectbox[0].select(model).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) == 1


def test_leaderboard_runs():
    app = AppTest.from_file(
        str(ROOT / "app/pages/3_Model_Leaderboard.py"), default_timeout=60
    ).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 5
