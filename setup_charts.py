"""
Builds setup candidates for one or more days and draws a review chart for each day.

    python setup_charts.py --dates 2026-10-20
    python setup_charts.py --dates 2026-10-20,2026-09-15,2025-06-10
    python setup_charts.py --dates 2026-10-20 --no-m1      (skip the 1-minute data, faster)

For each day you get (in the "charts" folder):
  setups_<date>.png   5M candles; for every setup #n:
                        star     the sweep (red = high swept, look short; green = low swept, look long)
                        box      the inversion gap (IFVG) that confirmed the reversal
                        diamond  entry variant A (close of the inversion candle)
                        dotted   stop (dark red) and take profits TP1-TP3 (grey)
  setups_<date>.csv   every field and tag for every setup
Times are broker SERVER time.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from detector_charts import load_m1, price_range, style_axis, window
from detectors import build_timeframes, find_fvgs
from levels import build_levels
from setups import build_setups
from swings import draw_candles, load_m5

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, DARKRED = "#2e9e6b", "#d9534f", "#8b0000"


def run_day(m5_all, m1_all, day):
    m5 = m5_all[(m5_all["time"] >= day - pd.Timedelta(days=45)) &
                (m5_all["time"] < day + pd.Timedelta(days=2))].reset_index(drop=True)
    m1 = None
    if m1_all is not None:
        m1 = m1_all[(m1_all["time"] >= day - pd.Timedelta(days=1)) &
                    (m1_all["time"] < day + pd.Timedelta(days=2))].reset_index(drop=True)
    tfs = build_timeframes(m5, m1)
    fvgs = {tf: find_fvgs(df, tf) for tf, df in tfs.items()}
    points, legs, piv = build_levels(tfs, m5)
    setups = build_setups(points, tfs, legs, piv, fvgs, day, day + pd.Timedelta(days=1))
    return tfs, setups


def draw_day(tfs, setups, day, path):
    view = window(tfs["5M"], day, day + pd.Timedelta(days=1))
    vt = view["time"].to_numpy()

    def xc(ts):
        return max(int(np.searchsorted(vt, np.datetime64(ts), side="right")) - 1, 0)

    fig, ax = plt.subplots(figsize=(22, 9))
    draw_candles(ax, view)
    y_range = price_range(view)
    for _, s in setups.iterrows():
        short = s["trade"] == "short"
        color = RED if short else GREEN
        x_sw, x_en = xc(s["sweep_time"]), xc(s["entry_time"])
        ax.scatter([x_sw], [s["level"]], marker="*", s=260, color=color, edgecolor="black", zorder=6)
        ax.text(x_sw, s["level"] + (3 if short else -3), f"#{s['id']}", fontsize=11, fontweight="bold",
                ha="center", va="bottom" if short else "top", zorder=7)
        x0, x1 = xc(s["ifvg_formed"]), xc(s["ifvg_invert"]) + 6
        ax.add_patch(Rectangle((x0, s["ifvg_bottom"]), max(x1 - x0, 1), s["ifvg_top"] - s["ifvg_bottom"],
                               facecolor=color, alpha=0.30, edgecolor=color, zorder=2))
        ax.scatter([x_en], [s["entry_price"]], marker="D", s=70, color=color, edgecolor="black", zorder=6)
        xe = min(x_en + 40, len(view) - 1)
        ax.hlines(s["stop"], x_en, xe, colors=DARKRED, linestyles=":", linewidth=1.6, zorder=3)
        for k in (1, 2, 3):
            tp = s[f"tp{k}"]
            if pd.notna(tp):
                ax.hlines(tp, x_en, xe, colors="#555555", linestyles=":", linewidth=1.1, zorder=3)
                ax.text(xe, tp, f" TP{k}", fontsize=7, va="center", color="#555555", clip_on=True)
    ax.set_title(f"{day.date()} 5M | #n = setup number | star = sweep, box = inversion gap, diamond = entry A, "
                 f"dotted: dark red = stop, grey = TP1-3 | setups: {len(setups)}", fontsize=12, loc="left")
    style_axis(ax, view, 12, "%H:%M")
    ax.set_ylim(*y_range)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--dates", required=True, help="comma separated, e.g. 2026-10-20,2026-09-15")
    parser.add_argument("--no-m1", action="store_true")
    parser.add_argument("--suffix", default="", help="added to output file names, e.g. _v2")
    args = parser.parse_args()

    days = [pd.Timestamp(d.strip()) for d in args.dates.split(",")]
    m5_all = load_m5(args.data_dir)
    first = min(days) - pd.Timedelta(days=1)
    m1_all = None if args.no_m1 else load_m1(args.data_dir, first)
    out = os.path.join(HERE, "charts")
    os.makedirs(out, exist_ok=True)

    show = ["id", "sweep_time", "trade", "level", "tfs", "window_extreme", "rank", "htf_tap", "vs_1h", "vs_4h",
            "entry_tf", "entry_time", "entry_price", "stop", "tp1", "tp2", "tp3", "rr1"]
    for day in days:
        tfs, setups = run_day(m5_all, m1_all, day)
        png = os.path.join(out, f"setups_{day.date()}{args.suffix}.png")
        draw_day(tfs, setups, day, png)
        setups.to_csv(os.path.join(out, f"setups_{day.date()}{args.suffix}.csv"), index=False)
        print(f"\n=== {day.date()}: {len(setups)} setups (server time) ===")
        if len(setups):
            print(setups[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("Saved", png)


if __name__ == "__main__":
    main()
