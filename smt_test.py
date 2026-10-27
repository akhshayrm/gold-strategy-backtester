"""
SMT divergence test. Reads only; places no trades.

Hypothesis, fixed before looking at any result:
    Among sweeps of a prior swing on the Nasdaq, those where the S&P 500 does NOT confirm
    the move (no matching sweep of its own corresponding swing) outperform those where it does.

Definition used:
    - swings taken on the 15-minute chart, 2 candles either side, known only after confirmation
    - a sweep is the first time price trades beyond a confirmed swing
    - the S&P's "corresponding" swing is its most recent confirmed swing of the same kind,
      formed within SWING_TOL minutes of the Nasdaq's
    - confirmation means the S&P trades beyond that swing within +/- SWEEP_WINDOW minutes
    - no confirmation = SMT divergence

Trades (identical for both groups, so the only difference is the divergence label):
    entry  close of the first 5M candle that closes back INSIDE the swept level
    stop   beyond the extreme of the sweep
    TP1-3  nearest unswept 5M/15M swings in the trade direction
    rule B management: hold through TP1, break-even at TP2, close at TP3

    python smt_test.py --nas <dir> --sp <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import MAX_HOURS, point_size
from detectors import add_atr
from levels import build_levels, visible_pivots
from swings import find_swings, resample

HERE = os.path.dirname(os.path.abspath(__file__))
SWING_N = 2             # candles either side, on the 15M
SWING_TOL = 30          # minutes apart two swings may be and still count as the same point
SWEEP_WINDOW = 30       # minutes either side in which the other index may confirm
REENTRY_BARS = 12       # 5M candles allowed for price to close back inside the level
FUTURES_SPREAD = 0.25   # index points; the CFD spread makes intraday index CFDs untradeable


def confirmed_swings(df15):
    """[(time, known_time, kind, price)] - known_time respects the confirmation delay."""
    is_h, is_l = find_swings(df15, SWING_N)
    out = []
    for kind, flags in (("high", is_h), ("low", is_l)):
        for i in np.where(flags)[0]:
            k = i + SWING_N
            if k >= len(df15):
                continue
            out.append((df15["time"].iat[i], df15["time"].iat[k] + pd.Timedelta(minutes=15),
                        kind, float(df15["high"].iat[i] if kind == "high" else df15["low"].iat[i])))
    return sorted(out)


def find_sweeps(df5, swings):
    """First time price trades beyond each confirmed swing."""
    t, hi, lo = df5["time"].to_numpy(), df5["high"].to_numpy(), df5["low"].to_numpy()
    out = []
    for s_time, known, kind, level in swings:
        a = int(np.searchsorted(t, np.datetime64(known), side="left"))
        arr = hi[a:a + 5000] if kind == "high" else lo[a:a + 5000]
        if not len(arr):
            continue
        hit = arr > level if kind == "high" else arr < level
        if not hit.any():
            continue
        j = a + int(hit.argmax())
        out.append({"swing_time": s_time, "kind": kind, "level": level,
                    "sweep_time": pd.Timestamp(t[j]), "sweep_idx": j})
    return pd.DataFrame(out).sort_values("sweep_time").reset_index(drop=True)


class SwingIndex:
    """Swings of the other index, sorted by time so lookups are a binary search.

    The obvious version rescans every swing for every sweep, which is 13,000 x 13,000
    comparisons and takes eleven minutes. This takes seconds.
    """

    def __init__(self, swings):
        self.by_kind = {}
        for kind in ("high", "low"):
            rows = sorted((s for s in swings if s[2] == kind), key=lambda s: s[0])
            self.by_kind[kind] = (
                np.array([np.datetime64(s[0]) for s in rows]),
                np.array([np.datetime64(s[1]) for s in rows]),
                np.array([s[3] for s in rows], dtype=float))

    def corresponding(self, kind, swing_time, sweep_time, tol_min):
        times, known, prices = self.by_kind[kind]
        if not len(times):
            return None
        tol = np.timedelta64(tol_min, "m")
        a = int(np.searchsorted(times, np.datetime64(swing_time) - tol, side="left"))
        b = int(np.searchsorted(times, np.datetime64(swing_time) + tol, side="right"))
        if b <= a:
            return None
        sel = known[a:b] <= np.datetime64(sweep_time)
        if not sel.any():
            return None
        p = prices[a:b][sel]
        return float(p.max() if kind == "high" else p.min())


def confirmed_by_other(sw, other5, index, until=None):
    """Did the other index sweep its own corresponding swing? (False = SMT divergence)

    `until` is the open time of the last confirming-index candle that has CLOSED by the
    moment of entry. Without it the check looks SWEEP_WINDOW minutes past the sweep, which
    is usually past the entry: it reads prices you could not have seen yet (look-ahead).
    """
    t, hi, lo = other5["time"].to_numpy(), other5["high"].to_numpy(), other5["low"].to_numpy()
    level = index.corresponding(sw["kind"], sw["swing_time"], sw["sweep_time"], SWING_TOL)
    if level is None:
        return None                      # no comparable structure: unusable, not divergence
    a = int(np.searchsorted(t, np.datetime64(sw["sweep_time"] - pd.Timedelta(minutes=SWEEP_WINDOW))))
    if until is None:
        b = int(np.searchsorted(t, np.datetime64(sw["sweep_time"] + pd.Timedelta(minutes=SWEEP_WINDOW))))
    else:
        b = int(np.searchsorted(t, np.datetime64(until), side="right"))
    if b <= a:
        return None
    return bool((hi[a:b] > level).any()) if sw["kind"] == "high" else bool((lo[a:b] < level).any())


def build_trade(sw, df5, piv_by_tf):
    """Enter when price closes back inside the swept level; stop beyond the sweep extreme."""
    o, h, l, c = (df5[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = df5["time"].to_numpy()
    j = sw["sweep_idx"]
    end = min(j + REENTRY_BARS, len(df5) - 1)
    entry_i = None
    for i in range(j, end):
        inside = c[i] < sw["level"] if sw["kind"] == "high" else c[i] > sw["level"]
        if inside and i > j:
            entry_i = i
            break
    if entry_i is None:
        return None
    short = sw["kind"] == "high"
    entry = float(c[entry_i])
    stop = float(h[j:entry_i + 1].max() if short else l[j:entry_i + 1].min())
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    entry_time = pd.Timestamp(t[entry_i]) + pd.Timedelta(minutes=5)

    want = "low" if short else "high"
    tps = []
    for tf in ("5M", "15M"):
        for p in visible_pivots(piv_by_tf, tf, entry_time):
            if p["kind"] != want:
                continue
            if (want == "low" and p["price"] >= entry) or (want == "high" and p["price"] <= entry):
                continue
            tps.append(p["price"])
    tps = sorted(set(tps), reverse=short)[:3]
    if not tps:
        return None
    return {"trade": "short" if short else "long", "entry_time": entry_time, "entry_price": entry,
            "stop": stop, "tp1": tps[0],
            "tp2": tps[1] if len(tps) > 1 else np.nan,
            "tp3": tps[2] if len(tps) > 2 else np.nan}


def replay(tr, m1, spread):
    """Rule B: hold through TP1, break-even at TP2, close at TP3."""
    long_ = tr["trade"] == "long"
    t = m1["time"].to_numpy()
    s = int(np.searchsorted(t, np.datetime64(tr["entry_time"]), side="left"))
    if s >= len(m1):
        return None
    entry = tr["entry_price"] + spread if long_ else tr["entry_price"]
    stop = tr["stop"]
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    tps = [tr[f"tp{k}"] for k in (1, 2, 3)]
    tps = [float(x) for x in tps if pd.notna(x)]
    tps = [x for x in tps if (x > entry if long_ else x < entry)]
    if not tps:
        return None
    hi, lo, cl = m1["high"].to_numpy(), m1["low"].to_numpy(), m1["close"].to_numpy()
    end = min(s + MAX_HOURS * 60, len(m1))
    hit, ex, ei = 0, None, end - 1
    for i in range(s, end):
        H = hi[i] + (0 if long_ else spread)
        L = lo[i] + (0 if long_ else spread)
        if (L <= stop) if long_ else (H >= stop):
            ex, ei = stop, i
            break
        while hit < len(tps) and (H >= tps[hit] if long_ else L <= tps[hit]):
            hit += 1
            if hit >= 3 or hit >= len(tps):
                ex, ei = tps[hit - 1], i
                break
            if hit == 2:
                stop = entry
        if ex is not None:
            break
    if ex is None:
        if end >= len(m1):
            return None
        ex = cl[ei] + (spread if not long_ else 0)
    pnl = (ex - entry) if long_ else (entry - ex)
    return pnl / risk


def stats(R, name):
    R = np.asarray(R)
    if not len(R):
        return f"{name:34s} no trades"
    pos, neg = R[R > 0].sum(), -R[R <= 0].sum()
    pf = pos / neg if neg > 0 else np.nan
    return (f"{name:34s} {len(R):4d} | win {100*(R>0).mean():5.1f}% | "
            f"avg {R.mean():+.3f}R | total {R.sum():+7.1f}R | PF {pf:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traded", required=True, help="folder:stem of the instrument we trade")
    p.add_argument("--confirm", required=True, help="folder:stem of the instrument that confirms")
    p.add_argument("--cost", type=float, default=FUTURES_SPREAD, help="round-trip cost in price units")
    p.add_argument("--label", default="smt")
    p.add_argument("--lookahead", action="store_true",
                   help="reproduce the original, biased labelling (for comparison only)")
    p.add_argument("--from", dest="t0", default="2024-09-01")
    p.add_argument("--to", dest="t1", default="2026-10-23")
    args = p.parse_args()

    def load(spec, tf):
        folder, stem = spec.split(":")
        return pd.read_csv(os.path.join(folder, f"{stem}_{tf}.csv"), parse_dates=["time"])

    nas5 = load(args.traded, "M5")
    nas1 = load(args.traded, "M1")[["time", "open", "high", "low", "close"]]
    sp5 = load(args.confirm, "M5")
    for d in (nas5, nas1, sp5):
        d.query("time >= @args.t0 and time <= @args.t1", inplace=True)
        d.reset_index(drop=True, inplace=True)

    nas15, sp15 = resample(nas5, "15min"), resample(sp5, "15min")
    nas_sw, sp_sw = confirmed_swings(nas15), confirmed_swings(sp15)
    print(f"{args.label}: traded swings {len(nas_sw)} | confirming swings {len(sp_sw)}")

    sp_index = SwingIndex(sp_sw)
    sweeps = find_sweeps(nas5, nas_sw)
    print(f"sweeps: {len(sweeps)}")

    tfs = {"5M": add_atr(nas5), "15M": add_atr(nas15),
           "30M": add_atr(resample(nas5, "30min")), "1H": add_atr(resample(nas5, "1h")),
           "4H": add_atr(resample(nas5, "4h"))}
    points, _, piv = build_levels(tfs, nas5)
    sig = points[points["eligible"]]
    sig_levels = set(zip(sig["kind"], sig["level"].round(2)))

    rows = []
    for _, sw in sweeps.iterrows():
        tr = build_trade(sw, nas5, piv)
        if tr is None:
            continue
        # only S&P candles that have closed by the time we enter
        last_closed = tr["entry_time"] - pd.Timedelta(minutes=5)
        conf = confirmed_by_other(sw, sp5, sp_index, until=None if args.lookahead else last_closed)
        if conf is None:
            continue
        R = replay(tr, nas1, args.cost)
        if R is None:
            continue
        rows.append({**tr, "sweep_time": sw["sweep_time"], "kind": sw["kind"],
                     "level": sw["level"], "R": R, "divergent": not conf,
                     "significant": (sw["kind"], round(sw["level"], 2)) in sig_levels})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "results", f"smt_trades_{args.label}.csv"), index=False)

    print(f"\nTradeable sweeps: {len(d)}  (round-trip cost {args.cost} price units)")
    print("\nRUN 1 - any confirmed 15M swing")
    print(stats(d.R, "  all sweeps"))
    print(stats(d[d.divergent].R, "  WITH SMT divergence"))
    print(stats(d[~d.divergent].R, "  without (S&P confirmed)"))
    s = d[d.significant]
    print(f"\nRUN 2 - significant liquidity only ({len(s)} of {len(d)})")
    print(stats(s.R, "  all sweeps"))
    print(stats(s[s.divergent].R, "  WITH SMT divergence"))
    print(stats(s[~s.divergent].R, "  without (S&P confirmed)"))


if __name__ == "__main__":
    main()
