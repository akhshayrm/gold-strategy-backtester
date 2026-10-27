"""
Draws the detectors on charts so you can compare them with your own markings.

    python detector_charts.py                    latest full day
    python detector_charts.py --date 2026-10-20
    python detector_charts.py --m1               also draw the 1-minute chart (slower)

Charts go into the "charts" folder. Times are broker SERVER time.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from detectors import build_timeframes, find_fvgs, find_sweeps
from swings import draw_candles, file_stem, load_m5

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_MIN_ATR = 0.25                      # only DRAW gaps bigger than this x ATR (less clutter)
GREEN, RED, BLUE, ORANGE = "#2e9e6b", "#d9534f", "#1f4fd8", "#e08a00"
TF_ALPHA = {"15M": 0.15, "30M": 0.22, "1H": 0.30, "4H": 0.40}


def load_m1(data_dir, since):
    path = os.path.join(data_dir, f"{file_stem()}_M1.csv")
    df = pd.read_csv(path, parse_dates=["time"], usecols=["time", "open", "high", "low", "close"])
    return df[df["time"] >= since].reset_index(drop=True)


def x_of(view_times, ts):
    return int(np.searchsorted(view_times, np.datetime64(ts)))


def style_axis(ax, view, every, fmt):
    x = np.arange(len(view))
    ax.set_xticks(x[::every])
    ax.set_xticklabels(view["time"].iloc[::every].dt.strftime(fmt), rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-1, len(view))
    ax.grid(alpha=0.2)


def window(df, start, end):
    return df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)


def price_range(view, pad_share=0.25):
    lo, hi = view["low"].min(), view["high"].max()
    pad = (hi - lo) * pad_share
    return lo - pad, hi + pad


def draw_gaps(ax, gaps, vt, start, end, n_view, alpha, label=None, edge=None, y_range=None):
    drawn = 0
    for _, g in gaps.iterrows():
        if y_range and (g["top"] < y_range[0] or g["bottom"] > y_range[1]):
            continue
        stop = g["fill_time"] if pd.notna(g["fill_time"]) else end
        stop = min(stop, end)
        if stop <= start or g["formed_time"] >= end:
            continue
        x0 = max(x_of(vt, max(g["formed_time"], start)), 0)
        x1 = min(x_of(vt, stop), n_view)
        if x1 <= x0:
            continue
        color = GREEN if g["direction"] == "bull" else RED
        ax.add_patch(Rectangle((x0, g["bottom"]), x1 - x0, g["top"] - g["bottom"],
                               facecolor=color, alpha=alpha, edgecolor=edge or color, linewidth=0.6))
        if label:
            ax.text(x0 + 0.3, g["top"], label, fontsize=7, va="bottom", color="#333", clip_on=True)
        drawn += 1
    return drawn


def chart_htf_fvg(tfs, fvgs, day, path):
    start, end = day - pd.Timedelta(days=2) + pd.Timedelta(hours=1), day + pd.Timedelta(hours=23)
    view = window(tfs["15M"], start, end)
    vt = view["time"].to_numpy()
    fig, ax = plt.subplots(figsize=(20, 9))
    draw_candles(ax, view)
    y_range = price_range(view)
    counts = {}
    for tf in ("4H", "1H", "30M", "15M"):
        f = fvgs[tf]
        f = f[f["gap_size"] >= PLOT_MIN_ATR * f["atr"]]
        counts[tf] = draw_gaps(ax, f, vt, start, end, len(view), TF_ALPHA[tf], label=tf, y_range=y_range)
    ax.set_title("Higher-timeframe FVGs on 15M candles (green = bullish gap, red = bearish; darker = higher timeframe; "
                 "a gap stops when price fills it) | drawn: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
                 fontsize=12, loc="left")
    style_axis(ax, view, 8, "%d %H:%M")
    ax.set_ylim(*y_range)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("Saved", path, counts)


def chart_sweeps(tf_name, tfs, sweeps, day, path, hours=(8, 22)):
    start, end = day + pd.Timedelta(hours=hours[0]), day + pd.Timedelta(hours=hours[1])
    view = window(tfs[tf_name], start, end)
    vt = view["time"].to_numpy()
    fig, ax = plt.subplots(figsize=(20, 8))
    draw_candles(ax, view)
    sw = sweeps[tf_name]
    sw = sw[(sw["known_time"] < end) & (sw["swing_time"] > start - pd.Timedelta(hours=12))]
    n_sweeps = 0
    for _, s in sw.iterrows():
        swept = pd.notna(s["sweep_time"]) and s["sweep_time"] <= end
        x0 = max(x_of(vt, s["swing_time"]), 0)
        x1 = x_of(vt, s["sweep_time"]) if swept else len(view) - 1
        if x1 <= 0:
            continue
        color = BLUE if s["kind"] == "high" else ORANGE
        ax.hlines(s["level"], x0, x1, colors=color, linestyles="--", linewidth=0.9)
        if swept and x1 < len(view):
            ax.scatter([x1], [s["extreme"]], marker="X", s=70, color=RED if s["kind"] == "high" else GREEN, zorder=4)
            n_sweeps += 1
    ax.set_title(f"{tf_name} swing levels (dashed: blue = highs, orange = lows) and sweeps "
                 f"(X: red = highs swept, green = lows swept) | sweeps shown: {n_sweeps}", fontsize=12, loc="left")
    style_axis(ax, view, 12 if tf_name == "5M" else 60, "%H:%M")
    ax.set_ylim(*price_range(view))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("Saved", path, "sweeps:", n_sweeps)


def chart_ifvg(tf_name, tfs, fvgs, day, path, hours=(8, 22)):
    start, end = day + pd.Timedelta(hours=hours[0]), day + pd.Timedelta(hours=hours[1])
    view = window(tfs[tf_name], start, end)
    vt = view["time"].to_numpy()
    fig, ax = plt.subplots(figsize=(20, 8))
    draw_candles(ax, view)
    f = fvgs[tf_name]
    f = f[(f["gap_size"] >= PLOT_MIN_ATR * f["atr"]) & (f["formed_time"] >= start - pd.Timedelta(hours=1))]
    y_range = price_range(view)
    draw_gaps(ax, f, vt, start, end, len(view), 0.25, y_range=y_range)
    n_inv = 0
    for _, g in f.iterrows():
        if pd.isna(g["invert_time"]) or not (start <= g["invert_time"] <= end):
            continue
        x = x_of(vt, g["invert_time"])
        if x >= len(view):
            continue
        # bullish gap closed through downward -> bearish IFVG (red); bearish gap closed up -> bullish IFVG (green)
        ax.scatter([x], [view["close"].iloc[x]], marker="D", s=60,
                   color=RED if g["direction"] == "bull" else GREEN, edgecolor="black", zorder=4)
        n_inv += 1
    ax.set_title(f"{tf_name} FVGs (rectangles until filled) and inversions (diamond at the closing candle: "
                 f"red = bearish IFVG, green = bullish IFVG) | inversions shown: {n_inv}", fontsize=12, loc="left")
    style_axis(ax, view, 12 if tf_name == "5M" else 60, "%H:%M")
    ax.set_ylim(*y_range)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("Saved", path, "inversions:", n_inv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--date", default=None)
    parser.add_argument("--m1", action="store_true")
    args = parser.parse_args()

    m5 = load_m5(args.data_dir)
    counts = m5.groupby(m5["time"].dt.normalize()).size()
    day = pd.Timestamp(args.date) if args.date else counts[counts >= 200].index[-1]

    # 45 days of history so 1H/4H gaps formed earlier are known
    since = day - pd.Timedelta(days=45)
    m5 = m5[m5["time"] >= since].reset_index(drop=True)
    m1 = load_m1(args.data_dir, day - pd.Timedelta(days=3)) if args.m1 else None
    tfs = build_timeframes(m5, m1)

    fvgs = {tf: find_fvgs(df, tf) for tf, df in tfs.items()}
    sweeps = {tf: find_sweeps(df, tf) for tf, df in tfs.items()}

    out = os.path.join(HERE, "charts")
    os.makedirs(out, exist_ok=True)
    chart_htf_fvg(tfs, fvgs, day, os.path.join(out, f"fvg_htf_{day.date()}.png"))
    chart_sweeps("5M", tfs, sweeps, day, os.path.join(out, f"sweeps_5M_{day.date()}.png"))
    chart_ifvg("5M", tfs, fvgs, day, os.path.join(out, f"ifvg_5M_{day.date()}.png"))
    if args.m1:
        chart_ifvg("1M", tfs, fvgs, day, os.path.join(out, f"ifvg_1M_{day.date()}.png"), hours=(16, 19))
    print("\nDone. Open the PNG files in the 'charts' folder.")


if __name__ == "__main__":
    main()
