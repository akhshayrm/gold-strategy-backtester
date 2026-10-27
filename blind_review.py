"""
Blind review. Draws setups from the development period on charts that STOP at the entry
candle, so you cannot see what happened next. You label each one take or skip; the labels
then get turned into a selection rule and tested on all 8,500 development trades.

    python blind_review.py                 20 days, up to 3 setups each (about 60 charts)
    python blind_review.py --days 10       fewer, if you want a quick first pass

Charts land in charts/blind/ as blind_001.png, blind_002.png, ...
DO NOT OPEN charts/blind/_answers.csv until you have finished labelling. It holds the
outcomes, and seeing them defeats the whole point.

Each chart shows 5M candles ending at your entry. Dashed lines are liquidity levels,
coloured by timeframe (5M grey, 15M blue, 30M purple, 1H orange, 4H black). The star is
the sweep, the box is the inversion gap, the diamond is the entry, the dark red dotted
line is the stop and the grey dotted lines are the targets.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from detector_charts import load_m1, price_range, style_axis, window
from detectors import build_timeframes, find_fvgs
from level_charts import TF_COLOR, TF_WIDTH
from levels import LOOKBACK_HOURS, build_levels
from setups import build_setups
from swings import draw_candles, load_m5

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_START, DEV_END = "2019-01-01", "2024-12-31"     # holdout starts 2025-01-01
HOURS_BEFORE = 10                                    # how much history each chart shows
GREEN, RED, DARKRED = "#2e9e6b", "#d9534f", "#8b0000"

LEGEND = [Line2D([], [], color=c, ls="--", lw=TF_WIDTH[tf], label=f"{tf} level")
          for tf, c in TF_COLOR.items()]


def trading_days(m5, n, seed):
    counts = m5.groupby(m5["time"].dt.normalize()).size()
    days = counts[(counts >= 250) & (counts.index.dayofweek < 5)].index
    days = days[(days >= pd.Timestamp(DEV_START)) & (days <= pd.Timestamp(DEV_END))]
    return sorted(pd.Series(days).sample(n=min(n, len(days)), random_state=seed))


def build_day(m5_all, m1_all, day):
    """Everything one day needs, computed once (run_day would repeat the level work)."""
    m5 = m5_all[(m5_all["time"] >= day - pd.Timedelta(days=45)) &
                (m5_all["time"] < day + pd.Timedelta(days=2))].reset_index(drop=True)
    m1 = m1_all[(m1_all["time"] >= day - pd.Timedelta(days=1)) &
                (m1_all["time"] < day + pd.Timedelta(days=2))].reset_index(drop=True)
    tfs = build_timeframes(m5, m1 if len(m1) > 500 else None)
    fvgs = {tf: find_fvgs(df, tf) for tf, df in tfs.items()}
    points, legs, piv = build_levels(tfs, m5)
    setups = build_setups(points, tfs, legs, piv, fvgs, day, day + pd.Timedelta(days=1))
    return tfs, setups, points, legs


def draw_blind(tfs, points, legs, s, n, path):
    """One setup, with every candle after the entry removed."""
    end = pd.Timestamp(s["entry_time"])
    view = window(tfs["5M"], end - pd.Timedelta(hours=HOURS_BEFORE), end)
    if len(view) < 20:
        return False
    vt = view["time"].to_numpy()

    def xc(ts):
        return max(int(np.searchsorted(vt, np.datetime64(ts), side="right")) - 1, 0)

    short = s["trade"] == "short"
    color = RED if short else GREEN
    fig, ax = plt.subplots(figsize=(17, 8))
    draw_candles(ax, view)
    y0, y1 = price_range(view)

    for _, p in points.iterrows():
        if p["known_time"] > end or p["time"] < end - pd.Timedelta(hours=LOOKBACK_HOURS[p["tf"]]):
            continue
        if not (y0 <= p["level"] <= y1):
            continue
        swept = pd.notna(p["sweep_time"]) and p["sweep_time"] <= end
        a = max(xc(p["time"]), 0)
        b = xc(p["sweep_time"]) if swept else len(view) - 1
        if b > a:
            ax.hlines(p["level"], a, b, colors=TF_COLOR[p["tf"]], linestyles="--",
                      linewidth=TF_WIDTH[p["tf"]], zorder=1)

    for lg in legs["15M"]:
        a, b = lg["start"]["time"], lg["extreme"]["time"]
        if b < view["time"].iat[0] or a > end:
            continue
        xa, xb = max(xc(a), 0), min(xc(b), len(view) - 1)
        if xb > xa:
            ax.plot([xa, xb], [lg["start"]["price"], lg["extreme"]["price"]],
                    color=RED if lg["dir"] == -1 else GREEN, linewidth=2.4, alpha=0.45, zorder=2)

    # mark holes in the data: price moved with no candles recorded, so the candle after a
    # hole can look like one huge bar that never really traded that way
    step = view["time"].diff().dt.total_seconds().div(60)
    for j in np.where(step.to_numpy() > 5.5)[0]:
        ax.axvline(j - 0.5, color="#b07d2b", linestyle="-", linewidth=1.4, alpha=0.65, zorder=0)
        ax.text(j - 0.5, y1, f" gap {step.iat[j]/60:.1f}h", fontsize=7, color="#b07d2b",
                va="top", ha="left", rotation=90, clip_on=True)

    x_sw, x_en = xc(s["sweep_time"]), len(view) - 1
    ax.scatter([x_sw], [s["level"]], marker="*", s=320, color=color, edgecolor="black", zorder=6)
    g0, g1 = xc(s["ifvg_formed"]), xc(s["ifvg_invert"])
    ax.add_patch(Rectangle((g0, s["ifvg_bottom"]), max(g1 - g0, 1), s["ifvg_top"] - s["ifvg_bottom"],
                           facecolor=color, alpha=0.35, edgecolor=color, zorder=3))
    ax.scatter([x_en], [s["entry_price"]], marker="D", s=95, color=color, edgecolor="black", zorder=6)
    pad = max(int(len(view) * 0.18), 12)
    ax.hlines(s["stop"], x_en, x_en + pad, colors=DARKRED, linestyles=":", linewidth=1.8, zorder=4)
    ax.text(x_en + pad + 0.5, s["stop"], "stop", fontsize=9, va="center", color=DARKRED, clip_on=False)
    for k in (1, 2, 3):
        tp = s.get(f"tp{k}")
        if pd.notna(tp) and y0 <= tp <= y1:
            ax.hlines(tp, x_en, x_en + pad, colors="#555555", linestyles=":", linewidth=1.2, zorder=4)
            ax.text(x_en + pad + 0.5, tp, f"TP{k}", fontsize=9, va="center", color="#555555", clip_on=False)

    ax.set_ylim(y0, y1)
    ax.set_title(f"#{n:03d}   {'SHORT' if short else 'LONG'}   "
                 f"(chart ends at the entry — nothing after it is shown)", fontsize=13, loc="left")
    ax.legend(handles=LEGEND, loc="upper left", fontsize=8, framealpha=0.9)
    style_axis(ax, view, max(len(view) // 14, 1), "%d %b %H:%M")
    ax.set_xlim(-1, len(view) + pad + 14)       # after style_axis, which resets the limits
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--per-day", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out = os.path.join(HERE, "charts", "blind")
    os.makedirs(out, exist_ok=True)

    m5_all = load_m5(args.data_dir)
    days = trading_days(m5_all, args.days, args.seed)
    print(f"Days chosen: {', '.join(str(d.date()) for d in days)}\n")
    m1_all = load_m1(args.data_dir, min(days) - pd.Timedelta(days=2))

    rng = np.random.default_rng(args.seed)
    rows, n = [], 0
    for day in days:
        tfs, setups, points, legs = build_day(m5_all, m1_all, day)
        if setups.empty:
            print(f"{day.date()}: no setups")
            continue
        take = setups.sample(n=min(args.per_day, len(setups)), random_state=int(rng.integers(1e6)))
        made = 0
        for _, s in take.sort_values("entry_time").iterrows():
            n += 1
            path = os.path.join(out, f"blind_{n:03d}.png")
            if draw_blind(tfs, points, legs, s, n, path):
                rows.append({"n": n, **s.to_dict()})
                made += 1
            else:
                n -= 1
        print(f"{day.date()}: {made} charts")

    truth = pd.DataFrame(rows)
    truth.to_csv(os.path.join(out, "_answers.csv"), index=False)
    sheet = truth[["n", "trade"]].copy()
    sheet["your_label"] = ""
    sheet.to_csv(os.path.join(out, "labels_blank.csv"), index=False)

    print(f"\n{len(truth)} charts in {out}")
    print("Go through blind_001.png onwards and decide take or skip for each.")
    print("Send the list back as: 1 skip, 2 take, 3 skip, ...")
    print("Do NOT open _answers.csv until you are done.")


if __name__ == "__main__":
    main()
