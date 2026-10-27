"""
Draws trend legs, liquidity levels and trigger sweeps so you can check them by eye.

    python level_charts.py                    latest full day
    python level_charts.py --date 2026-10-20

What you see:
  thick lines     trend legs of the chart timeframe (red = down leg, green = up leg)
  dashed lines    unswept liquidity levels, from where they formed until price swept them
                  (colour = timeframe: 5M grey, 15M blue, 30M purple, 1H orange, 4H black)
  big star        TRIGGER: an eligible sweep (red star = a high was swept, so look for shorts;
                  green star = a low was swept, so look for longs)
  small grey x    a sweep that does NOT count (continuation inside the current trend, or too old)

Charts and a CSV of the day's triggers go into the "charts" folder. Times are SERVER time.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detectors import build_timeframes
from detector_charts import price_range, style_axis, window, x_of
from levels import LOOKBACK_HOURS, build_levels
from swings import draw_candles, load_m5

HERE = os.path.dirname(os.path.abspath(__file__))
TF_COLOR = {"5M": "#7f8c8d", "15M": "#1f4fd8", "30M": "#8e44ad", "1H": "#e08a00", "4H": "#000000"}
TF_WIDTH = {"5M": 0.9, "15M": 1.0, "30M": 1.2, "1H": 1.5, "4H": 1.8}
GREEN, RED = "#2e9e6b", "#d9534f"


def chart(tf_name, tfs, points, legs_by_tf, start, end, path, every, fmt):
    view = window(tfs[tf_name], start, end)
    vt = view["time"].to_numpy()
    y_range = price_range(view)
    fig, ax = plt.subplots(figsize=(20, 9))
    draw_candles(ax, view)

    # trend legs of this timeframe
    for lg in legs_by_tf[tf_name]:
        a, b = lg["start"]["time"], lg["extreme"]["time"]
        if b < start or a > end:
            continue
        xa, xb = max(x_of(vt, a), 0), min(x_of(vt, b), len(view) - 1)
        if xb <= xa:
            continue
        ax.plot([xa, xb], [lg["start"]["price"], lg["extreme"]["price"]],
                color=RED if lg["dir"] == -1 else GREEN, linewidth=2.6, alpha=0.55, zorder=2)

    n_trig = 0
    for _, p in points.iterrows():
        if p["known_time"] > end or p["time"] < start - pd.Timedelta(hours=LOOKBACK_HOURS[p["tf"]]):
            continue
        if pd.notna(p["sweep_time"]) and p["sweep_time"] < start:
            continue
        if p["level"] < y_range[0] or p["level"] > y_range[1]:
            continue
        swept_in_view = pd.notna(p["sweep_time"]) and p["sweep_time"] <= end
        x0 = max(x_of(vt, p["time"]), 0)
        x1 = x_of(vt, p["sweep_time"]) if swept_in_view else len(view) - 1
        if x1 <= x0:
            continue
        ax.hlines(p["level"], x0, min(x1, len(view) - 1), colors=TF_COLOR[p["tf"]],
                  linestyles="--", linewidth=TF_WIDTH[p["tf"]], zorder=1)
        if swept_in_view and x1 < len(view):
            if p["eligible"]:
                ax.scatter([x1], [p["extreme"]], marker="*", s=260, edgecolor="black", zorder=5,
                           color=RED if p["kind"] == "high" else GREEN)
                n_trig += 1
            else:
                ax.scatter([x1], [p["extreme"]], marker="x", s=25, color="#888888", zorder=4)

    ax.set_title(f"{tf_name} candles | thick lines = {tf_name} trend legs | dashed = unswept liquidity by timeframe | "
                 f"stars = triggers (red: high swept, look short; green: low swept, look long) | triggers shown: {n_trig}",
                 fontsize=11, loc="left")
    style_axis(ax, view, every, fmt)
    ax.set_ylim(*y_range)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("Saved", path, "| triggers:", n_trig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    m5 = load_m5(args.data_dir)
    counts = m5.groupby(m5["time"].dt.normalize()).size()
    day = pd.Timestamp(args.date) if args.date else counts[counts >= 200].index[-1]
    m5 = m5[m5["time"] >= day - pd.Timedelta(days=45)].reset_index(drop=True)

    tfs = build_timeframes(m5)
    points, legs_by_tf, _ = build_levels(tfs, m5)

    out = os.path.join(HERE, "charts")
    os.makedirs(out, exist_ok=True)
    chart("15M", tfs, points, legs_by_tf, day - pd.Timedelta(days=2) + pd.Timedelta(hours=1),
          day + pd.Timedelta(hours=23), os.path.join(out, f"levels_15M_{day.date()}.png"), 8, "%d %H:%M")
    chart("5M", tfs, points, legs_by_tf, day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=22),
          os.path.join(out, f"levels_5M_{day.date()}.png"), 12, "%H:%M")

    trig = points[points["eligible"] & (points["sweep_time"] >= day) & (points["sweep_time"] < day + pd.Timedelta(days=1))]
    trig = trig.sort_values("sweep_time")
    cols = ["sweep_time", "tf", "kind", "level", "time", "rank", "age_hours", "merged"]
    csv_path = os.path.join(out, f"triggers_{day.date()}.csv")
    trig[cols].to_csv(csv_path, index=False)
    print(f"\nTriggers on {day.date()} (server time): {len(trig)}")
    print(trig[cols].to_string(index=False))
    print("\nSaved", csv_path)


if __name__ == "__main__":
    main()
