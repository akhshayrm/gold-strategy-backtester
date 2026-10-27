"""
Swing detector + comparison charts.

A swing high is a candle whose high is higher than the N candles before it and
at least as high as the N candles after it. Swing lows are the mirror image.
A swing is only KNOWN N candles after it forms (the backtester must respect that).

Run:
    python swings.py                       charts for the latest full day
    python swings.py --date 2026-10-20     charts for a specific day
    python swings.py --data-dir C:\\path\\to\\data

Charts are saved in a "charts" folder next to this script.
Times shown are your broker's SERVER time.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Which instrument to work on. Set the AGENT_SYMBOL environment variable to use another,
# e.g.  set AGENT_SYMBOL=NAS100  (Windows)  before running any script.
SYMBOL = os.environ.get("AGENT_SYMBOL", "XAUUSD.vx")


def file_stem(symbol=None):
    """CSV name for a symbol: XAUUSD.vx -> XAUUSD_vx"""
    return (symbol or SYMBOL).replace(".", "_").replace("/", "_")

N_VALUES = (1, 2, 3)


def load_m5(data_dir):
    path = os.path.join(data_dir, f"{file_stem()}_M5.csv")
    return pd.read_csv(path, parse_dates=["time"], usecols=["time", "open", "high", "low", "close"])


def resample(df, rule):
    out = (df.set_index("time")
             .resample(rule)
             .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
             .dropna()
             .reset_index())
    return out


def find_swings(df, n):
    """Return (is_swing_high, is_swing_low) as boolean numpy arrays."""
    high, low = df["high"], df["low"]
    left_max = high.shift(1).rolling(n).max()
    right_max = high.shift(-n).rolling(n).max()
    left_min = low.shift(1).rolling(n).min()
    right_min = low.shift(-n).rolling(n).min()
    is_high = (high > left_max) & (high >= right_max)
    is_low = (low < left_min) & (low <= right_min)
    return is_high.fillna(False).to_numpy(), is_low.fillna(False).to_numpy()


def swings_per_day(df, start="2025-01-01"):
    recent = df[df["time"] >= start].reset_index(drop=True)
    days = recent["time"].dt.date.nunique()
    rows = []
    for n in N_VALUES:
        h, l = find_swings(recent, n)
        rows.append((n, (h.sum() + l.sum()) / days))
    return rows


def draw_candles(ax, view):
    x = np.arange(len(view))
    up = view["close"] >= view["open"]
    colors = np.where(up, "#2e9e6b", "#d9534f")
    ax.vlines(x, view["low"], view["high"], color=colors, linewidth=0.8)
    body_low = np.minimum(view["open"], view["close"])
    body_h = np.maximum((view["close"] - view["open"]).abs(), 0.01)
    ax.bar(x, body_h, bottom=body_low, color=colors, width=0.7)
    return x


def make_chart(df, label, start, end, path, tick_every, tick_fmt):
    fig, axes = plt.subplots(len(N_VALUES), 1, figsize=(20, 5.2 * len(N_VALUES)), sharex=False)
    for ax, n in zip(axes, N_VALUES):
        is_high, is_low = find_swings(df, n)
        mask = (df["time"] >= start) & (df["time"] <= end)
        idx = np.where(mask.to_numpy())[0]
        view = df.iloc[idx].reset_index(drop=True)
        x = draw_candles(ax, view)
        pad = (view["high"].max() - view["low"].min()) * 0.03
        sh = np.where(is_high[idx])[0]
        sl = np.where(is_low[idx])[0]
        ax.scatter(sh, view["high"].iloc[sh] + pad, marker="v", color="#1f4fd8", s=45, zorder=3)
        ax.scatter(sl, view["low"].iloc[sl] - pad, marker="^", color="#e08a00", s=45, zorder=3)
        ax.set_title(f"{label}: {n} candle(s) each side | {len(sh)} swing highs (blue), {len(sl)} swing lows (orange)",
                     fontsize=13, loc="left")
        ticks = x[::tick_every]
        ax.set_xticks(ticks)
        ax.set_xticklabels(view["time"].iloc[ticks].dt.strftime(tick_fmt), rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_xlim(-1, len(view))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (server time)")
    args = parser.parse_args()

    m5 = load_m5(args.data_dir)
    m15 = resample(m5, "15min")

    if args.date:
        day = pd.Timestamp(args.date)
    else:
        counts = m5.groupby(m5["time"].dt.normalize()).size()
        day = counts[counts >= 200].index[-1]

    out_dir = os.path.join(HERE, "charts")
    os.makedirs(out_dir, exist_ok=True)

    print("\nAverage swings per trading day since 2025 (highs + lows):")
    for label, df in (("5M", m5), ("15M", m15)):
        for n, per_day in swings_per_day(df):
            print(f"  {label}, {n} candle(s) each side: {per_day:.1f} per day")

    make_chart(m5, "5M", day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=22),
               os.path.join(out_dir, f"swings_5M_{day.date()}.png"), 12, "%H:%M")
    make_chart(m15, "15M", day - pd.Timedelta(days=2) + pd.Timedelta(hours=1), day + pd.Timedelta(hours=23),
               os.path.join(out_dir, f"swings_15M_{day.date()}.png"), 8, "%d %H:%M")
    print("\nDone. Open the PNG files in the 'charts' folder.")


if __name__ == "__main__":
    main()
