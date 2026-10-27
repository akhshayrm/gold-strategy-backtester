"""
Delayed-entry test on SMT-divergent Nasdaq sweeps. Reads only; places no trades.

PRE-REGISTRATION - written before running it, not to be edited after:

  Why:      a random-wait control beat the 1M inversion-gap entry (+0.322R vs +0.120R),
            which suggests simply waiting after the reclaim may help. That was noticed
            after the fact, so it needs its own clean test.
  Universe: SMT-divergent sweeps from smt_test.py with the look-ahead FIXED
            (results/smt_trades_nas_causal.csv).
  Rule:     DELAY = 3. Enter at the close of the 3rd 5M candle after the reclaim candle
            (15 minutes later). One value, chosen now, no others will be tried.
  Stop:     beyond the extreme between the sweep and the delayed entry.
  Targets:  the same TP1-3 as the reclaim trade (all known before either entry); targets
            already passed by the delayed entry are dropped.
  Cost:     0.86 index points round trip (Topstep MNQ).
  Control:  the identical delay on the CONFIRMED sweeps. If waiting helps those just as
            much, the improvement is about waiting, not about SMT.
  Periods:  development 2024-09-01 to 2025-12-31, holdout 2026 (looked at once).
  PASS if, in development: delayed beats reclaim (paired p < 0.05) AND the delayed
            average is above zero after costs. Then in the holdout: the delayed average
            is above zero and still beats reclaim. Anything less is a fail.
  Extra:    "same size" result - the delayed trade's points divided by the RECLAIM trade's
            risk. If R improves only because stops got smaller, this exposes it.

    python delay_test.py --nas <folder with NAS100_vx_M5.csv and _M1.csv>
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from smt_test import replay

HERE = os.path.dirname(os.path.abspath(__file__))
DELAY = 3
COST = 0.86
DEV_END = pd.Timestamp("2026-01-01")


def delayed(tr, t5, h5, l5, c5):
    """The same trade entered DELAY candles later. Returns (trade dict, risk) or None."""
    reclaim = int(np.searchsorted(t5, np.datetime64(tr["entry_time"] - pd.Timedelta(minutes=5))))
    k = reclaim + DELAY
    s = int(np.searchsorted(t5, np.datetime64(tr["sweep_time"])))
    if k >= len(t5) or t5[reclaim] != np.datetime64(tr["entry_time"] - pd.Timedelta(minutes=5)):
        return None
    short = tr["trade"] == "short"
    stop = float(h5[s:k + 1].max() if short else l5[s:k + 1].min())
    alt = dict(tr)
    alt["entry_time"] = pd.Timestamp(t5[k]) + pd.Timedelta(minutes=5)
    alt["entry_price"] = float(c5[k])
    alt["stop"] = stop
    return alt


def paired(a, b):
    t, p = stats.ttest_rel(b, a)
    return b.mean() - a.mean(), p


def report(res, name):
    print(f"\n{name}: {len(res)} trades with both entries")
    for col, lab in (("reclaim", "reclaim entry"), ("delayed", f"delayed {DELAY} candles"),
                     ("same_size", "delayed, same size as reclaim")):
        R = res[col].to_numpy(float)
        pos, neg = R[R > 0].sum(), -R[R <= 0].sum()
        print(f"  {lab:32s} win {100*(R>0).mean():5.1f}% | avg {R.mean():+.4f}R | PF {pos/neg:.2f}")
    d, p = paired(res["reclaim"].to_numpy(float), res["delayed"].to_numpy(float))
    _, p0 = stats.ttest_1samp(res["delayed"].to_numpy(float), 0.0)
    print(f"  delayed minus reclaim {d:+.4f}R (paired p = {p:.4f}) | delayed vs zero p = {p0:.4f}")
    return d, p, res["delayed"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nas", required=True)
    ap.add_argument("--trades", default=os.path.join(HERE, "results", "smt_trades_nas_causal.csv"))
    ap.add_argument("--holdout", action="store_true", help="also run 2026 - do this ONCE")
    args = ap.parse_args()

    d = pd.read_csv(args.trades, parse_dates=["entry_time", "sweep_time"])
    m5 = pd.read_csv(os.path.join(args.nas, "NAS100_vx_M5.csv"), parse_dates=["time"])
    m1 = pd.read_csv(os.path.join(args.nas, "NAS100_vx_M1.csv"), parse_dates=["time"])[
        ["time", "open", "high", "low", "close"]]
    t5, h5, l5, c5 = (m5[k].to_numpy() for k in ("time", "high", "low", "close"))

    rows, skipped = [], 0
    for _, tr in d.iterrows():
        tr = tr.to_dict()
        alt = delayed(tr, t5, h5, l5, c5)
        R0 = replay(tr, m1, COST)
        R1 = replay(alt, m1, COST) if alt else None
        if R0 is None or R1 is None:
            skipped += 1
            continue
        reclaim_risk = abs(tr["entry_price"] + (COST if tr["trade"] == "long" else 0) - tr["stop"])
        delayed_risk = abs(alt["entry_price"] + (COST if tr["trade"] == "long" else 0) - alt["stop"])
        rows.append({"sweep_time": tr["sweep_time"], "divergent": tr["divergent"],
                     "reclaim": R0, "delayed": R1, "same_size": R1 * delayed_risk / reclaim_risk,
                     "reclaim_risk": reclaim_risk, "delayed_risk": delayed_risk})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results", "delay_test.csv"), index=False)
    print(f"{len(res)} trades, {skipped} skipped (no target left or ran out of data)")

    dev = res[res.sweep_time < DEV_END]
    print("\n" + "=" * 70 + "\nDEVELOPMENT 2024-09 to 2025-12")
    dd, dp, dm = report(dev[dev.divergent], "SMT divergent (the test)")
    report(dev[~dev.divergent], "S&P confirmed (control)")
    dev_pass = dp < 0.05 and dd > 0 and dm > 0
    print(f"\nDEVELOPMENT PASS: {'YES' if dev_pass else 'NO'}")

    if args.holdout:
        ho = res[res.sweep_time >= DEV_END]
        print("\n" + "=" * 70 + "\nHOLDOUT 2026 (one look)")
        hd, hp, hm = report(ho[ho.divergent], "SMT divergent")
        report(ho[~ho.divergent], "S&P confirmed (control)")
        print(f"\nHOLDOUT PASS: {'YES' if dev_pass and hd > 0 and hm > 0 else 'NO'}")


if __name__ == "__main__":
    main()
