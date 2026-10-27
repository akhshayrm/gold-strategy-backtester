"""
Compares trade-management rules on setups that have already been found.

Finding setups is the slow part; re-running the trades is fast. So this loads a finished
backtest, throws away its exits, and replays every trade under several management rules.

    python management_test.py --trades results/trades_dev_v3.csv

Rules compared (all use one entry, all close at TP3):
    A  break-even at TP1, stop to TP1 at TP2     (what we have been testing)
    B  hold through TP1, break-even at TP2       (your proposal)
    C  no stop moves at all                      (the control)
    D  close the whole trade at TP1              (take the money)
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import LOT, MAX_HOURS, spread_at

HERE = os.path.dirname(os.path.abspath(__file__))

# for each rule: what the stop becomes after TP1, and after TP2.
# "keep" = leave it, "be" = break-even (the entry), "tp1" = the TP1 price
RULES = {
    "A break-even at TP1, TP1 at TP2": ("be", "tp1"),
    "B hold at TP1, break-even at TP2": ("keep", "be"),
    "C never move the stop": ("keep", "keep"),
    "D close it all at TP1": ("exit", "exit"),
}


def replay(trade, m1, spread, after_tp1, after_tp2):
    long_ = trade["trade"] == "long"
    t = m1["time"].to_numpy()
    start = int(np.searchsorted(t, np.datetime64(trade["entry_time"]), side="left"))
    if start >= len(m1):
        return None
    entry = trade["entry_price"] + spread if long_ else trade["entry_price"]
    stop = float(trade["stop"])
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    tps = [float(trade[f"tp{k}"]) for k in (1, 2, 3) if pd.notna(trade.get(f"tp{k}"))]
    tps = [x for x in tps if (x > entry if long_ else x < entry)]
    if not tps:
        return None

    hi, lo = m1["high"].to_numpy(), m1["low"].to_numpy()
    end = min(start + MAX_HOURS * 60, len(m1))
    hit, exit_price, exit_i = 0, None, end - 1

    for i in range(start, end):
        high = hi[i] + (0 if long_ else spread)
        low = lo[i] + (0 if long_ else spread)
        if (low <= stop) if long_ else (high >= stop):     # stop checked first, as before
            exit_price, exit_i = stop, i
            break
        while hit < len(tps) and (high >= tps[hit] if long_ else low <= tps[hit]):
            hit += 1
            move = after_tp1 if hit == 1 else after_tp2 if hit == 2 else "keep"
            if move == "exit" or hit >= 3 or hit >= len(tps):
                exit_price, exit_i = tps[hit - 1], i
                break
            if move == "be":
                stop = entry
            elif move == "tp1":
                stop = tps[0]
        if exit_price is not None:
            break
    if exit_price is None:
        if end >= len(m1):
            return None
        exit_price = float(m1["close"].iat[exit_i]) + (spread if not long_ else 0)
    pnl = (exit_price - entry) if long_ else (entry - exit_price)
    return {"R": pnl / risk, "tps_hit": hit, "usd": pnl * LOT * 100}


def summarise(name, rs):
    r = np.array([x["R"] for x in rs])
    pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
    peak = np.maximum.accumulate(np.cumsum(r))
    return {"rule": name, "trades": len(r), "win_rate": round(float((r > 0).mean()), 3),
            "scratch_rate": round(float((r == 0).mean()), 3),
            "avg_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
            "profit_factor": round(float(pos / neg), 2) if neg > 0 else np.nan,
            "max_dd_R": round(float((np.cumsum(r) - peak).min()), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default=os.path.join(HERE, "results", "trades_dev_v3.csv"))
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = p.parse_args()

    d = pd.read_csv(args.trades, parse_dates=["entry_time"])
    m5 = pd.read_csv(os.path.join(args.data_dir, "XAUUSD_vx_M5.csv"), parse_dates=["time"])
    m1 = pd.read_csv(os.path.join(args.data_dir, "XAUUSD_vx_M1.csv"), parse_dates=["time"],
                     usecols=["time", "open", "high", "low", "close"])
    spreads = [spread_at(m5, t) for t in d["entry_time"]]

    out = []
    for name, (a1, a2) in RULES.items():
        rs = []
        for (_, row), sp in zip(d.iterrows(), spreads):
            r = replay(row, m1, sp, a1, a2)
            if r:
                rs.append(r)
        out.append(summarise(name, rs))
        print("done:", name, flush=True)

    res = pd.DataFrame(out)
    print("\n" + res.to_string(index=False))
    path = os.path.join(HERE, "results", "management_comparison.csv")
    res.to_csv(path, index=False)
    print("\nSaved", path)


if __name__ == "__main__":
    main()
