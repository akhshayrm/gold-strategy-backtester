"""
Two pre-registered tests on SMT-divergent sweeps. Reads only; places no trades.

Test 1 - does the 1-minute inversion-gap entry beat the reclaim entry?
         Same sweeps, same stop and target rules; only the entry moment differs.
         99% of trades have both available, so this is about entry price, not selection.

Test 2 - does requiring a 5-minute inversion gap select better setups?
         Both groups use the reclaim entry; the split is whether a 5M gap also inverted.

Costs: 0.86 index points, Topstep's real MNQ round trip (0.25 spread + $1.22 commission).

    python entry_test.py --nas <folder>
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from detectors import TF_MINUTES, build_timeframes, find_fvgs
from setups import find_ifvg
from smt_test import replay

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 0.86
CHUNK_MONTHS = 4


def ifvg_entries(div, m5, m1):
    """For each sweep, the 1M and 5M inversion-gap entry, if one formed after it."""
    out = {}
    start = pd.Timestamp(div["sweep_time"].min()).normalize()
    end = pd.Timestamp(div["sweep_time"].max()) + pd.Timedelta(days=1)
    while start < end:
        stop = start + pd.DateOffset(months=CHUNK_MONTHS)
        a5 = m5[(m5["time"] >= start - pd.Timedelta(days=5)) & (m5["time"] < stop)].reset_index(drop=True)
        a1 = m1[(m1["time"] >= start - pd.Timedelta(days=5)) & (m1["time"] < stop)].reset_index(drop=True)
        if len(a5) > 500:
            tfs = build_timeframes(a5, a1)
            fv = {tf: find_fvgs(tfs[tf], tf) for tf in ("5M", "1M")}
            sub = div[(div["sweep_time"] >= start) & (div["sweep_time"] < stop)]
            for i, r in sub.iterrows():
                rec = {}
                for etf in ("1M", "5M"):
                    g = find_ifvg(fv[etf], r["kind"], r["sweep_time"], TF_MINUTES[etf])
                    if g is None:
                        continue
                    t = g["invert_time"] + pd.Timedelta(minutes=TF_MINUTES[etf])
                    df = tfs[etf]
                    k = int(np.searchsorted(df["time"].to_numpy(), np.datetime64(g["invert_time"]), side="left"))
                    if k >= len(df):
                        continue
                    # The stop must sit behind the extreme price reached between the sweep
                    # and THIS entry, not the earlier reclaim entry. Keeping the earlier stop
                    # understates the risk, and on 36% of these the old stop had already been
                    # breached before the inversion even formed.
                    s = int(np.searchsorted(df["time"].to_numpy(), np.datetime64(r["sweep_time"]), side="left"))
                    seg_hi = df["high"].to_numpy()[s:k + 1]
                    seg_lo = df["low"].to_numpy()[s:k + 1]
                    if len(seg_hi) == 0:
                        continue
                    stop_px = float(seg_hi.max() if r["kind"] == "high" else seg_lo.min())
                    rec[etf] = (t, float(df["close"].iat[k]), stop_px)
                out[i] = rec
            print(f"  {start.date()}: {len(sub)} sweeps processed", flush=True)
        start = stop
    return out


def summarise(R, name):
    R = np.asarray([x for x in R if x is not None])
    if len(R) == 0:
        return f"{name:36s} no trades"
    pos, neg = R[R > 0].sum(), -R[R <= 0].sum()
    return (f"{name:36s} {len(R):4d} | win {100*(R>0).mean():5.1f}% | avg {R.mean():+.4f}R | "
            f"PF {pos/neg:.2f}" if neg > 0 else f"{name}: no losses")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nas", default="/tmp/nasdata")
    p.add_argument("--trades", default=os.path.join(HERE, "results", "smt_trades_nas_full.csv"))
    args = p.parse_args()

    d = pd.read_csv(args.trades, parse_dates=["entry_time", "sweep_time"])
    div = d[d["divergent"]].copy()
    m5 = pd.read_csv(os.path.join(args.nas, "NAS100_vx_M5.csv"), parse_dates=["time"])
    m1 = pd.read_csv(os.path.join(args.nas, "NAS100_vx_M1.csv"), parse_dates=["time"])
    m1 = m1[["time", "open", "high", "low", "close"]]

    print(f"{len(div)} SMT divergent sweeps. Finding inversion-gap entries...")
    entries = ifvg_entries(div, m5, m1)

    rows = []
    for i, r in div.iterrows():
        rec = entries.get(i, {})
        row = {"reclaim": r["R"], "has5M": "5M" in rec, "sweep_time": r["sweep_time"]}
        for etf in ("1M", "5M"):
            if etf not in rec:
                row[etf] = None
                continue
            t, px, stop = rec[etf]
            alt = dict(r)
            alt["entry_time"], alt["entry_price"], alt["stop"] = t, px, stop
            row[etf] = replay(alt, m1, COST)
            row[f"{etf}_risk"] = abs(px - stop)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results", "entry_test.csv"), index=False)

    print("\n" + "=" * 70)
    print("TEST 1 - 1M inversion-gap entry versus the reclaim entry")
    both = res[res["1M"].notna()]
    print(summarise(both["reclaim"], "  reclaim entry"))
    print(summarise(both["1M"], "  1M IFVG entry"))
    a, b = both["reclaim"].to_numpy(float), both["1M"].to_numpy(float)
    t, pv = stats.ttest_rel(b, a)
    print(f"  paired difference {b.mean()-a.mean():+.4f}R | p = {pv:.4f}")
    print(f"  PASS: {'YES' if pv < 0.05 and b.mean() > a.mean() and b.mean() > 0 else 'NO'}")

    print("\nTEST 2 - requiring a 5M inversion gap (reclaim entry for both)")
    print(summarise(res[res.has5M]["reclaim"], "  with a 5M IFVG"))
    print(summarise(res[~res.has5M]["reclaim"], "  without one"))
    x = res[res.has5M]["reclaim"].to_numpy(float)
    y = res[~res.has5M]["reclaim"].to_numpy(float)
    t2, p2 = stats.ttest_ind(x, y, equal_var=False)
    print(f"  difference {x.mean()-y.mean():+.4f}R | p = {p2:.4f}")
    print(f"  PASS: {'YES' if p2 < 0.05 and x.mean() > y.mean() and x.mean() > 0 else 'NO'}")

    print("\nstability by half-year (1M IFVG entry):")
    both = both.copy()
    both["half"] = both.sweep_time.dt.year.astype(str) + "H" + ((both.sweep_time.dt.month > 6).astype(int) + 1).astype(str)
    print(both.groupby("half")["1M"].agg(trades="size", avg="mean").round(4).to_string())


if __name__ == "__main__":
    main()
