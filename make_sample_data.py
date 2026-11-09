"""Make a small SYNTHETIC gold-style dataset in sample/. No real broker data is used.

Writes sample/XAUUSD_vx_M1.csv and sample/XAUUSD_vx_M5.csv with the same columns as
download_history.py: time,open,high,low,close,tick_volume,spread,real_volume.

The prices are a random walk with changing volatility, so they look gold-like but mean
nothing. Do NOT read any strategy result from them. They are only for testing that the
scripts run (e.g. python backtest.py --data-dir sample ...).

    python make_sample_data.py
    python make_sample_data.py --days 20 --seed 7
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_m1(days, seed, start_date, start_price):
    rng = np.random.default_rng(seed)
    # Weekdays only, one candle per minute, 01:00-23:59 (a gap overnight, like a broker).
    day_list = pd.bdate_range(start_date, periods=days)
    times = pd.DatetimeIndex([])
    for d in day_list:
        times = times.append(pd.date_range(d + pd.Timedelta(hours=1),
                                           d + pd.Timedelta(hours=23, minutes=59), freq="1min"))
    n = len(times)

    # Volatility drifts slowly (regimes), so quiet and busy stretches both appear.
    vol = 0.12 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3)
    vol = np.clip(vol, 0.04, 0.6)
    steps = rng.normal(0, 1, n) * vol
    close = start_price + np.cumsum(steps)
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 1, (n, 2))) * vol[:, None] * 0.6
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]

    df = pd.DataFrame({
        "time": times,
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "tick_volume": rng.integers(5, 120, n).astype(float),
        "spread": 0,
        "real_volume": 0.0,
    })
    # Rounding can nudge a body outside the wicks; put the high/low back around it.
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def to_m5(m1):
    g = m1.set_index("time").resample("5min")
    m5 = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "tick_volume": g["tick_volume"].sum(),
        "spread": g["spread"].max(),
        "real_volume": g["real_volume"].sum(),
    }).dropna(subset=["open"]).reset_index()
    return m5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=10, help="trading days to make (default 10)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="2024-03-04", help="first day (default 2024-03-04)")
    ap.add_argument("--out", default="sample", help="output folder (default sample)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m1 = make_m1(args.days, args.seed, args.start, start_price=2000.0)
    m5 = to_m5(m1)
    m1.to_csv(os.path.join(args.out, "XAUUSD_vx_M1.csv"), index=False)
    m5.to_csv(os.path.join(args.out, "XAUUSD_vx_M5.csv"), index=False)
    print(f"Wrote {len(m1)} M1 and {len(m5)} M5 candles to {args.out}\\ (synthetic, not real prices)")


if __name__ == "__main__":
    main()
