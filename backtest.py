"""
Backtester. It only READS data and never places trades.

One entry per setup (your decision). The trade is simulated forward on 1-minute
candles, so we see the order things happened in rather than guessing.

Rules simulated:
  entry   at the close of the inversion candle (variant A)
  stop    behind the extreme of the sweep
  TP1-3   the nearest unswept 5M/15M swing points in the trade direction
  at TP1  stop moves to break-even
  at TP2  stop moves to the TP1 price
  at TP3  close
  timeout close at market after MAX_HOURS

Conservative assumptions:
  - if one candle reaches both the stop and the next target, the STOP is taken first
  - the spread is paid on entry; stops and targets are checked on the correct side
  - a trade still open at the end of the data is dropped, not counted as a winner

Results are reported in R (multiples of the money risked) and in dollars at 0.01 lot,
where a $1 move in gold is $1.

    python backtest.py --from 2019-01-01 --to 2024-12-31 --label dev
    python backtest.py --from 2025-01-01 --to 2026-10-21 --label holdout
"""
import argparse
import os

import numpy as np
import pandas as pd

from detectors import build_timeframes, find_fvgs
from levels import build_levels
from setups import build_setups
from swings import file_stem, load_m5

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_HOURS = 24           # close anything still open after this
WARMUP_DAYS = 45         # history each chunk needs before it can mark levels
CHUNK_DAYS = 30
DEFAULT_SPREAD = 0.33    # used when the data has no spread recorded (before 2025)
LOT = 0.01               # 0.01 lot of gold: a $1 move is $1


def spread_at(m5, t):
    """Broker spread in dollars, from the data when it is recorded, else the default."""
    i = int(np.searchsorted(m5["time"].to_numpy(), np.datetime64(t), side="right")) - 1
    if i < 0 or "spread" not in m5:
        return DEFAULT_SPREAD
    pts = float(m5["spread"].iat[i])
    return DEFAULT_SPREAD if pts <= 0 else pts * point_size(m5)


_POINT = {}


def point_size(m5):
    """One MT5 'point' in price terms, worked out from how many decimals the prices use.

    Gold quotes to 2 decimals so a point is 0.01; the Nasdaq CFD quotes to 1 so a point is
    0.1. Getting this wrong misstates every spread by a factor of ten.
    """
    key = id(m5)
    if key not in _POINT:
        # Find the coarsest precision the prices actually survive being rounded to.
        # Counting decimals in the text fails both ways: a 2-decimal instrument prints
        # 4133.6 for round numbers (understating), and float noise prints 30.30000000000001
        # (wildly overstating). Rounding is immune to both.
        px = m5["close"].tail(50000).to_numpy()
        digits = 5
        for d in range(6):
            if np.allclose(px, np.round(px, d), rtol=0, atol=1e-9):
                digits = d
                break
        _POINT[key] = 10.0 ** (-digits)
    return _POINT[key]


def simulate(trade, m1, spread):
    """Walk a single trade forward minute by minute. Returns the filled result or None."""
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
    tps = [trade.get(f"tp{k}") for k in (1, 2, 3)]
    tps = [float(x) for x in tps if pd.notna(x)]
    # drop targets on the wrong side of the entry
    tps = [x for x in tps if (x > entry if long_ else x < entry)]

    hi, lo = m1["high"].to_numpy(), m1["low"].to_numpy()
    end = min(start + MAX_HOURS * 60, len(m1))
    hit, exit_price, exit_reason, exit_i = 0, None, "timeout", end - 1

    for i in range(start, end):
        # ask = bid + spread; a short's stop and targets are reached on the ask
        high = hi[i] + (0 if long_ else spread)
        low = lo[i] + (0 if long_ else spread)

        stop_hit = low <= stop if long_ else high >= stop
        if stop_hit:                                    # checked first, on purpose
            exit_price, exit_reason, exit_i = stop, ("stop" if hit == 0 else f"trail_after_tp{hit}"), i
            break
        while hit < len(tps) and (high >= tps[hit] if long_ else low <= tps[hit]):
            hit += 1
            if hit == 1:
                stop = entry                            # break-even
            elif hit == 2:
                stop = tps[0]                           # stop to TP1
            if hit >= 3 or hit >= len(tps):
                exit_price, exit_reason, exit_i = tps[hit - 1], f"tp{hit}", i
                break
        if exit_price is not None:
            break
    if exit_price is None:
        if exit_i >= len(m1) - 1 and end >= len(m1):
            return None                                 # ran out of data: do not count it
        exit_price = float(m1["close"].iat[exit_i]) + (spread if not long_ else 0)

    pnl = (exit_price - entry) if long_ else (entry - exit_price)
    return {"fill": round(entry, 2), "risk_usd": round(risk, 2), "tps_hit": hit,
            "exit_time": pd.Timestamp(t[exit_i]), "exit_price": round(float(exit_price), 2),
            "exit_reason": exit_reason, "R": round(pnl / risk, 3), "usd": round(pnl * LOT * 100, 2),
            "minutes": int((pd.Timestamp(t[exit_i]) - pd.Timestamp(trade["entry_time"])).total_seconds() // 60)}


def generate(m5_all, m1_all, t0, t1):
    """Setup candidates between t0 and t1, built chunk by chunk with warm-up history."""
    out, start = [], pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    while start < t1:
        stop = min(start + pd.Timedelta(days=CHUNK_DAYS), t1)
        m5 = m5_all[(m5_all["time"] >= start - pd.Timedelta(days=WARMUP_DAYS)) &
                    (m5_all["time"] < stop + pd.Timedelta(days=2))].reset_index(drop=True)
        # 5 days back, so the "enough 1M data" check never depends on candles after the
        # signal (after a weekend or holiday, 2 days back held too few in short windows)
        m1 = m1_all[(m1_all["time"] >= start - pd.Timedelta(days=5)) &
                    (m1_all["time"] < stop + pd.Timedelta(days=2))].reset_index(drop=True)
        if len(m5) > 500:
            tfs = build_timeframes(m5, m1 if len(m1) > 500 else None)
            fvgs = {tf: find_fvgs(df, tf) for tf, df in tfs.items()}
            points, legs, piv = build_levels(tfs, m5)
            s = build_setups(points, tfs, legs, piv, fvgs, start, stop)
            if len(s):
                out.append(s)
        print(f"  {start.date()} -> {stop.date()}: {0 if not out else len(out[-1])} setups", flush=True)
        start = stop
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    return df.drop_duplicates(subset=["sweep_time", "trade", "entry_time"]).reset_index(drop=True)


def summarise(df, label):
    if df.empty:
        return pd.Series({"label": label, "trades": 0})
    r = df["R"]
    wins = r > 0
    gross_win, gross_loss = r[wins].sum(), -r[~wins].sum()
    peak = r.cumsum().cummax()
    return pd.Series({
        "label": label, "trades": len(r), "win_rate": round(wins.mean(), 3),
        "avg_R": round(r.mean(), 3), "median_R": round(r.median(), 3), "total_R": round(r.sum(), 1),
        "avg_win_R": round(r[wins].mean(), 2) if wins.any() else np.nan,
        "avg_loss_R": round(r[~wins].mean(), 2) if (~wins).any() else np.nan,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else np.nan,
        "max_drawdown_R": round((r.cumsum() - peak).min(), 1),
        "total_usd_0.01lot": round(df["usd"].sum(), 2),
        "avg_hours": round(df["minutes"].mean() / 60, 1),
    })


def by_group(df, col):
    rows = []
    for key, g in df.groupby(col):
        s = summarise(g, f"{col}={key}")
        rows.append(s)
    return pd.DataFrame(rows)[["label", "trades", "win_rate", "avg_R", "total_R", "profit_factor"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--from", dest="t0", required=True)
    p.add_argument("--to", dest="t1", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading data...", flush=True)
    m5_all = pd.read_csv(os.path.join(args.data_dir, f"{file_stem()}_M5.csv"), parse_dates=["time"])
    m1_all = pd.read_csv(os.path.join(args.data_dir, f"{file_stem()}_M1.csv"), parse_dates=["time"],
                         usecols=["time", "open", "high", "low", "close"])
    print(f"Generating setups {args.t0} to {args.t1}...", flush=True)
    setups = generate(m5_all, m1_all, args.t0, args.t1)
    if setups.empty:
        print("No setups found.")
        return

    print(f"\nSimulating {len(setups)} setups...", flush=True)
    results = []
    for _, s in setups.iterrows():
        res = simulate(s, m1_all, spread_at(m5_all, s["entry_time"]))
        if res:
            results.append({**s.to_dict(), **res})
    df = pd.DataFrame(results)
    path = os.path.join(args.out_dir, f"trades_{args.label}.csv")
    df.to_csv(path, index=False)

    print(f"\n===== {args.label}: all setups =====")
    print(summarise(df, "all").to_string())
    print("\nBy exit reason:")
    print(df["exit_reason"].value_counts().to_string())
    for col in ("trade", "top_tf", "window_extreme", "rank", "vs_1h", "vs_4h", "entry_tf"):
        if col in df:
            print(f"\nBy {col}:")
            print(by_group(df, col).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
