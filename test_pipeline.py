"""
Automated checks for the research pipeline. Reads only; places no trades.

    python test_pipeline.py                 fast unit tests (seconds)
    python test_pipeline.py --truncation    also the look-ahead test on real data (minutes)

The truncation test is the important one. For a sample of real setups it deletes every
candle that had not closed by the entry, rebuilds the setup from scratch, and checks the
result is identical. If any decision-relevant field changes, the pipeline was reading
prices from the future.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


def bars(rows, start="2026-01-05 10:00", freq="1min"):
    t = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    return df


# --------------------------------------------------------------------- units
def test_simulator():
    print("\nTrade simulator (backtest.simulate)")
    import backtest as bt
    base = dict(trade="long", entry_time=pd.Timestamp("2026-01-05 10:01"), entry_price=100.0,
                stop=98.0, tp1=103.0, tp2=106.0, tp3=110.0)
    flat = [(100, 100, 100, 100)]
    r = bt.simulate(base, bars(flat + [(100, 101, 99.5, 101), (101, 104, 101, 104), (104, 107, 104, 107),
                                       (107, 111, 107, 111)] + [(111,) * 4] * 5), 0.0)
    check("runs to TP3 = +5R", r and r["exit_reason"] == "tp3" and abs(r["R"] - 5.0) < 1e-9)
    r = bt.simulate(base, bars(flat + [(100, 100.2, 97, 97.5)] + [(97,) * 4] * 5), 0.0)
    check("straight to stop = -1R", r and r["R"] == -1.0)
    r = bt.simulate(base, bars(flat + [(100, 104, 97, 99)] + [(99,) * 4] * 5), 0.0)
    check("stop AND target in one candle -> stop wins", r and r["exit_reason"] == "stop")
    r = bt.simulate(base, bars(flat + [(100, 103.5, 100, 103), (103, 103, 99, 99.5)] + [(99,) * 4] * 5), 0.0)
    check("TP1 then reversal -> break-even", r and r["R"] == 0.0)
    r = bt.simulate(base, bars(flat + [(100, 100.2, 97, 97.5)] + [(97,) * 4] * 5), 0.40)
    check("long pays spread on entry", r and abs(r["fill"] - 100.4) < 1e-9 and r["R"] == -1.0)
    sh = dict(trade="short", entry_time=pd.Timestamp("2026-01-05 10:01"), entry_price=100.0,
              stop=102.0, tp1=97.0, tp2=94.0, tp3=90.0)
    r = bt.simulate(sh, bars(flat + [(100, 101.8, 100, 101.7)] + [(101,) * 4] * 5), 0.40)
    check("short stop is checked on the ask", r and r["exit_reason"] == "stop")
    r = bt.simulate(base, bars([(100,) * 4, (100, 100.5, 99.5, 100)]), 0.0)
    check("trade that runs out of data is dropped", r is None)


def test_point_size():
    print("\nSpread units (backtest.point_size)")
    import backtest as bt
    for name, px, want in [("2-decimal (gold)", [4133.02, 4133.6, 4134.11], 0.01),
                           ("1-decimal (index)", [28997.1, 28997.0, 29001.7], 0.1),
                           ("3-decimal (silver)", [53.797, 30.300000000000001, 53.8], 0.001)]:
        df = pd.DataFrame({"close": px})
        got = bt.point_size(df)
        check(f"{name}: point = {want}", abs(got - want) < 1e-15, f"got {got}")


def test_swings():
    print("\nSwing detection (swings.find_swings)")
    from swings import find_swings
    h = [4100, 4104, 4110, 4107, 4103, 4105, 4102, 4101, 4106, 4099]
    df = pd.DataFrame({"high": h, "low": [x - 2 for x in h]})
    hi, _ = find_swings(df, 2)
    check("2-each-side finds the one clear high", list(np.where(hi)[0]) == [2])
    df2 = pd.DataFrame({"high": [1, 3, 5, 5, 3, 1], "low": [0, 2, 4, 4, 2, 0]})
    hi, _ = find_swings(df2, 1)
    check("equal highs flag only the first", list(np.where(hi)[0]) == [2])


def test_fvg():
    print("\nFair value gaps (detectors.find_fvgs)")
    import detectors as d
    rows = [(100, 100.5, 99.5, 100)] * 30 + [(100, 101, 99.8, 100.8), (100.8, 106, 100.7, 105.5),
                                             (105.5, 107, 103, 106.5), (106.5, 107, 104, 104.5),
                                             (104.5, 105, 102.5, 103.5), (103.5, 103.8, 100.5, 100.8)]
    rows += [(100.8, 101, 100.4, 100.7)] * 5
    df = d.add_atr(bars(rows, freq="5min"))
    f = d.find_fvgs(df, "5M")
    g = f[(f.bottom == 101.0) & (f.top == 103.0)]
    check("bullish gap 101-103 found", len(g) == 1)
    if len(g):
        g = g.iloc[0]
        check("gap known only after candle 3 closes",
              g.formed_time == df.time.iloc[32] + pd.Timedelta(minutes=5))
        check("tapped on the right candle", g.tap_time == df.time.iloc[34])
        check("inverted by a body close below it", g.invert_time == df.time.iloc[35])


def test_smt_detection():
    print("\nSMT divergence labelling (smt_test)")
    import smt_test as st
    from swings import resample

    def m5(closes):
        """5M candles, each 1 point tall, from a list of closes."""
        return bars([(c, c + 0.5, c - 0.5, c) for c in closes], start="2026-01-05 10:00", freq="5min")

    def per15(prices):
        """Repeat each 15M price three times, so the 15M chart has the intended shape."""
        return [p for p in prices for _ in range(3)]

    # A clear 15M swing low at 90, then a sweep down to 88 at 15M bar 13.
    base = [100, 99, 98, 97, 96, 90, 96, 97, 98, 99, 100, 100, 100]
    nas = m5(per15(base + [88, 94, 95, 96, 96, 96]))

    def label(sp_closes, until=None):
        sp = m5(sp_closes)
        idx = st.SwingIndex(st.confirmed_swings(resample(sp, "15min")))
        sw = st.find_sweeps(nas, st.confirmed_swings(resample(nas, "15min"))).iloc[0]
        return st.confirmed_by_other(sw, sp, idx, until=until), sw

    hold = per15(base + [93, 94, 95, 96, 96, 96])
    got, sw = label(hold)
    check("Nasdaq sweep found at the right candle", sw["sweep_time"] == nas.time.iloc[39], str(sw["sweep_time"]))
    check("S&P holds its low -> SMT divergence", got is False)
    got, _ = label(per15(base + [88, 94, 95, 96, 96, 96]))
    check("S&P also sweeps -> confirmed", got is True)
    got, _ = label([100.0] * len(hold))
    check("S&P has no matching swing -> unusable, not divergence", got is None)

    # The look-ahead regression. S&P holds at the sweep, then breaks its low 25 minutes
    # later - after we would already be in the trade.
    late = hold.copy()
    late[39 + 5] = 88
    entry_bar_open = sw["sweep_time"] + pd.Timedelta(minutes=5)   # entry candle closes 10 min after the sweep
    got_old, _ = label(late)
    got_new, _ = label(late, until=entry_bar_open)
    check("old version counted the LATER S&P sweep (this was the bug)", got_old is True)
    check("fixed version ignores S&P moves after entry", got_new is False)


def test_prop_sim():
    print("\nProp evaluation simulator (prop_eval)")
    import prop_eval as pe
    kw = dict(runs=300, risk=0.01, target=0.08, daily_cap=0.05, max_dd=0.10, trailing=False, min_days=0)
    check("always win -> pass", pe.simulate([np.array([1.0])] * 20, max_days=0, **kw) == {"pass": 1.0})
    check("always lose -> drawdown", pe.simulate([np.array([-1.0])] * 20, max_days=0, **kw) == {"max drawdown": 1.0})
    kw2 = dict(kw, max_dd=0.50)
    check("6 losses in a day breaches 5% daily cap",
          pe.simulate([np.array([-1.0] * 6)], max_days=0, **kw2) == {"daily loss": 1.0})
    check("flat trades hit the time limit",
          pe.simulate([np.array([0.0])] * 20, max_days=20, **kw) == {"ran out of time": 1.0})


# --------------------------------------------------------------- truncation
def truncation_test(data_dir, days, per_day=4):
    print("\nLOOK-AHEAD (truncation) test on real gold data")
    from detectors import build_timeframes, find_fvgs
    from levels import build_levels
    from setups import build_setups
    from swings import load_m5
    from detector_charts import load_m1

    m5_all = load_m5(data_dir)
    m1_all = load_m1(data_dir, pd.Timestamp(min(days)) - pd.Timedelta(days=2))
    core = ["entry_price", "stop", "tp1", "tp2", "tp3", "opp_dist_30M", "tp1_dist_R", "trade", "level"]

    def run(m5, m1, day):
        tfs = build_timeframes(m5, m1)
        fv = {tf: find_fvgs(df, tf) for tf, df in tfs.items()}
        pts, legs, piv = build_levels(tfs, m5)
        return build_setups(pts, tfs, legs, piv, fv, day, day + pd.Timedelta(days=1))

    checked = mismatched = 0
    hidden = set()
    for d in days:
        day = pd.Timestamp(d)
        m5 = m5_all[(m5_all.time >= day - pd.Timedelta(days=45)) & (m5_all.time < day + pd.Timedelta(days=2))].reset_index(drop=True)
        m1 = m1_all[(m1_all.time >= day - pd.Timedelta(days=1)) & (m1_all.time < day + pd.Timedelta(days=2))].reset_index(drop=True)
        full = run(m5, m1, day)
        if full.empty:
            continue
        for _, s in full.head(per_day).iterrows():
            cut = s["entry_time"]
            # keep only candles that had CLOSED by the entry
            m5c = m5[m5.time + pd.Timedelta(minutes=5) <= cut].reset_index(drop=True)
            m1c = m1[m1.time + pd.Timedelta(minutes=1) <= cut].reset_index(drop=True)
            trunc = run(m5c, m1c, day)
            # the reverse direction: setups visible at the time that the full data hides
            if not trunc.empty:
                seen = set(zip(full.sweep_time, full.trade))
                ghosts = trunc[(trunc.entry_time <= cut) &
                               np.array([(a, b) not in seen for a, b in zip(trunc.sweep_time, trunc.trade)])]
                for _, g in ghosts.iterrows():
                    key = (g.sweep_time, g.trade)
                    if key not in hidden:
                        hidden.add(key)
                        print(f"    {d} {g.trade} sweep {g.sweep_time}: real-time setup MISSING from full-data run")
            match = trunc[(trunc.sweep_time == s.sweep_time) & (trunc.trade == s.trade)] if not trunc.empty else trunc
            checked += 1
            if match.empty:
                mismatched += 1
                print(f"    {d} {s.trade} sweep {s.sweep_time}: setup VANISHES when the future is removed")
                continue
            m = match.iloc[0]
            bad = [c for c in core if not (pd.isna(s[c]) and pd.isna(m[c])) and
                   not (isinstance(s[c], str) and s[c] == m[c]) and
                   not (not isinstance(s[c], str) and abs(float(s[c]) - float(m[c])) < 1e-6)]
            if bad:
                mismatched += 1
                print(f"    {d} {s.trade} sweep {s.sweep_time}: changed without the future: {bad}")
    check(f"{checked} real setups identical with all future candles deleted", mismatched == 0,
          f"{mismatched} differed")
    check("no real-time setups hidden by later data", not hidden, f"{len(hidden)} hidden")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truncation", action="store_true")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = p.parse_args()

    test_simulator()
    test_point_size()
    test_swings()
    test_fvg()
    test_smt_detection()
    test_prop_sim()
    if args.truncation:
        truncation_test(args.data_dir, ["2019-06-21", "2021-03-10", "2023-03-23", "2024-02-16", "2020-08-12", "2022-11-03",
                                        "2019-11-14", "2020-03-17", "2021-09-29", "2022-05-06", "2023-10-11", "2024-07-25"], per_day=6)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
