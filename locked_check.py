"""
Re-scores the LOCKED gold strategy (opposing-magnet filter + rule B) on a trades file.
Used to compare results before and after the look-ahead fixes. Reads only.

    python locked_check.py results/trades_holdout.csv results/trades_holdout_fixed.csv --data-dir data
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from backtest import spread_at
from management_test import replay


def score(path, m5, m1, cost=None):
    d = pd.read_csv(path, parse_dates=["entry_time"])
    keep = d[d["opp_dist_30M"] < d["tp1_dist_R"]]
    R = []
    for _, row in keep.iterrows():
        r = replay(row, m1, cost if cost is not None else spread_at(m5, row["entry_time"]), "keep", "be")
        if r:
            R.append(r["R"])
    R = np.array(R)
    pos, neg = R[R > 0].sum(), -R[R <= 0].sum()
    p = stats.ttest_1samp(R, 0).pvalue
    print(f"{os.path.basename(path):28s} setups {len(d):5d} | kept {len(R):4d} | win {100*(R>0).mean():4.1f}% | "
          f"avg {R.mean():+.3f}R | PF {pos/neg:.2f} | p {p:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--stem", default="XAUUSD_vx")
    ap.add_argument("--cost", type=float, default=None, help="fixed round-trip cost (futures)")
    args = ap.parse_args()
    m5 = pd.read_csv(os.path.join(args.data_dir, f"{args.stem}_M5.csv"), parse_dates=["time"])
    m1 = pd.read_csv(os.path.join(args.data_dir, f"{args.stem}_M1.csv"), parse_dates=["time"],
                     usecols=["time", "open", "high", "low", "close"])
    for f in args.files:
        score(f, m5, m1, args.cost)


if __name__ == "__main__":
    main()
