"""
Prop firm evaluation simulator. Reads only; places no trades.

Takes the strategy's actual trade outcomes, replays them through a prop firm's rules
thousands of times in different orders, and reports how often you would pass.

Why shuffle the order? Because passing or failing an evaluation depends heavily on luck
of sequencing. The same set of trades can pass comfortably in one order and breach the
daily loss limit in week one in another. Running it once tells you almost nothing.

Trades are resampled a DAY at a time, so trades that happened on the same day stay
together and the daily loss limit means something.

    python prop_eval.py                                  default rules
    python prop_eval.py --target 8 --daily 5 --maxdd 10 --trailing
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_days(path, r_col="R"):
    """Group the trade outcomes into trading days: [[R, R], [R], ...]"""
    d = pd.read_csv(path, parse_dates=["entry_time"])
    d = d.sort_values("entry_time")
    return [g[r_col].to_numpy() for _, g in d.groupby(d["entry_time"].dt.normalize())]


def run_one(days, rng, risk, target, daily_cap, max_dd, trailing, max_days, min_days):
    """One evaluation attempt. Returns 'pass', 'daily loss', 'max drawdown' or 'ran out of time'."""
    equity, peak, traded_days = 1.0, 1.0, 0
    limit = max_days if max_days else 10_000
    for _ in range(limit):
        day = days[rng.integers(len(days))]
        start_of_day = equity
        traded_days += 1
        for r in day:
            equity *= (1 + risk * r)              # risk a fixed % of current equity per trade
            peak = max(peak, equity)
            floor = (peak - max_dd) if trailing else (1.0 - max_dd)
            if equity <= floor:
                return "max drawdown"
            if equity <= start_of_day - daily_cap:
                return "daily loss"
        if equity >= 1.0 + target and traded_days >= min_days:
            return "pass"
    return "ran out of time"


def simulate(days, runs=20000, risk=0.01, target=0.08, daily_cap=0.05, max_dd=0.10,
             trailing=False, max_days=0, min_days=0, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for _ in range(runs):
        r = run_one(days, rng, risk, target, daily_cap, max_dd, trailing, max_days, min_days)
        out[r] = out.get(r, 0) + 1
    return {k: v / runs for k, v in out.items()}


def show(name, res):
    p = res.get("pass", 0.0)
    rest = ", ".join(f"{k} {100*v:.0f}%" for k, v in sorted(res.items()) if k != "pass")
    print(f"  {name:38s} PASS {100*p:5.1f}%   ({rest})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default=os.path.join(HERE, "results", "holdout_ruleB.csv"))
    p.add_argument("--risk", type=float, default=1.0, help="percent of equity risked per trade")
    p.add_argument("--target", type=float, default=8.0, help="profit target, percent")
    p.add_argument("--daily", type=float, default=5.0, help="daily loss limit, percent")
    p.add_argument("--maxdd", type=float, default=10.0, help="max drawdown, percent")
    p.add_argument("--trailing", action="store_true", help="drawdown trails the equity peak")
    p.add_argument("--max-days", type=int, default=0, help="time limit in trading days (0 = none)")
    p.add_argument("--min-days", type=int, default=0)
    p.add_argument("--runs", type=int, default=20000)
    args = p.parse_args()

    days = load_days(args.trades)
    allR = np.concatenate(days)
    print(f"Trade data: {len(allR)} trades over {len(days)} trading days "
          f"({len(allR)/len(days):.1f} per day), average {allR.mean():+.3f}R\n")

    common = dict(runs=args.runs, target=args.target/100, daily_cap=args.daily/100,
                  max_dd=args.maxdd/100, trailing=args.trailing,
                  max_days=args.max_days, min_days=args.min_days)
    print(f"Rules: target {args.target}%, daily loss {args.daily}%, max drawdown {args.maxdd}%"
          f"{' (trailing)' if args.trailing else ' (static)'}"
          f"{f', {args.max_days} day limit' if args.max_days else ', no time limit'}\n")

    print("Your strategy, at different risk per trade:")
    for risk in (0.25, 0.5, 1.0, 2.0):
        show(f"risking {risk}% per trade", simulate(days, risk=risk/100, **common))

    print("\nFor comparison, the same rules with:")
    shifted = [d - allR.mean() for d in days]                      # exactly zero edge
    show("a zero-edge version of your trades", simulate(shifted, risk=args.risk/100, **common))
    good = [d - allR.mean() + 0.30 for d in days]                  # a genuinely good strategy
    show("a +0.30R edge (what good looks like)", simulate(good, risk=args.risk/100, **common))


if __name__ == "__main__":
    main()
