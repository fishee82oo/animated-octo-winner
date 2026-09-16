import sys
import argparse
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.simulator import simulate, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    s = simulate()
    h = s.groupby(s.timestamp.dt.hour).price.mean()
    assert h.loc[12:14].mean() < h.loc[18:20].mean()
    assert s.loc[s.timestamp.dt.hour < 6, "solar_mw"].eq(0).all()
    c = simulate("sichuan")
    wet = c.timestamp.dt.month.between(6, 10)
    assert c.loc[wet, "hydro_mw"].mean() > c.loc[~wet, "hydro_mw"].mean() * 2
    g = simulate("gansu_mengxi")
    night = g[g.timestamp.dt.hour < 7]
    assert night.price.corr(night.wind_mw) < 0
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Shandong summer week: price",
            "Average daily price: duck curve + TOU",
            "Monthly average load",
            "Sichuan monthly hydro and price",
        ],
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
    )
    week = s[s.timestamp.between("2021-07-12", "2021-07-18 23:00")]
    fig.add_trace(
        go.Scatter(x=week.timestamp, y=week.price, name="Price CNY/MWh"), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Hourly price"), row=1, col=2)
    month = s.groupby(s.timestamp.dt.month).load_mw.mean()
    fig.add_trace(
        go.Scatter(x=month.index, y=month.values, name="Load MW"), row=2, col=1
    )
    cm = c.groupby(c.timestamp.dt.month)[["hydro_mw", "price"]].mean()
    fig.add_trace(go.Scatter(x=cm.index, y=cm.hydro_mw, name="Hydro MW"), row=2, col=2)
    fig.add_trace(
        go.Scatter(x=cm.index, y=cm.price, name="Sichuan price"),
        row=2,
        col=2,
        secondary_y=True,
    )
    fig.update_layout(
        height=800,
        title="Synthetic market sanity checks — illustrative, not live prices",
    )
    fig.write_html(ROOT / "reports/simulator_patterns.html", include_plotlyjs=True)
    if args.show:
        webbrowser.open((ROOT / "reports/simulator_patterns.html").as_uri())
    print(
        f"Midday {h.loc[12:14].mean():.1f}, evening {h.loc[18:20].mean():.1f} CNY/MWh; Sichuan wet/dry hydro {c.loc[wet, 'hydro_mw'].mean() / c.loc[~wet, 'hydro_mw'].mean():.2f}x; wind/night-price correlation {night.price.corr(night.wind_mw):.2f}"
    )


if __name__ == "__main__":
    main()
