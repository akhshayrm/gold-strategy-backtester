"""
Trend legs and liquidity trigger levels. It only READS data.

Legs (5M, 15M, 30M and also used to know the "current trend" on 1H/4H):
  A down leg starts at a swing high and continues while price makes lower highs and
  lower lows; it ends when a swing high breaks above the last lower high (mirror for
  up legs). The start of a leg is the end of the previous leg.

Liquidity points (levels that price can sweep):
  - 5M/15M/30M: the turning points of legs that moved at least LEG_ATR_MULT x ATR(15M).
  - 1H/4H: swing points that are at least SWING_ATR_MULT x ATR(own timeframe) away from
    the neighbouring swing.
  - Near-equal levels (within EQUAL_TOL_ATR x ATR) are merged into one, at the extreme.

A sweep counts as a TRIGGER only if the level comes from before the current trend
(its time is not later than the start of the leg active when price sweeps it) and it is
inside the lookback window. Sweeps of lows made inside the current trend are just
continuation and are marked not eligible.

Everything is in broker SERVER time and uses only information known at that moment.
"""
import numpy as np
import pandas as pd

from detectors import SWING_N, TF_MINUTES
from swings import find_swings

LEG_TFS = ("5M", "15M", "30M")
SIG_TFS = ("1H", "4H")
LEG_ATR_MULT = 2.0
SWING_ATR_MULT = 2.5
EQUAL_TOL_ATR = 0.25
EQUAL_PIVOT_TOL = 0.60       # how close two swing points must be to count as "equal"
EQUAL_PIVOT_HOURS = 30       # how far back to look for equal highs / lows
SWEEP_PEN_ATR = 0.05
LOOKBACK_HOURS = {"5M": 24, "15M": 24, "30M": 24, "1H": 120, "4H": 120}
MAX_LOOK = 30000     # 5M candles followed after a level forms (about 100 days)


# ----------------------------------------------------------------- pivots
def raw_pivots(df, tf):
    n = SWING_N[tf]
    is_high, is_low = find_swings(df, n)
    minutes = TF_MINUTES[tf]
    rows = [(i, "high", df["high"].iat[i]) for i in np.where(is_high)[0]]
    rows += [(i, "low", df["low"].iat[i]) for i in np.where(is_low)[0]]
    rows.sort(key=lambda r: (r[0], r[1]))
    out = []
    for i, kind, price in rows:
        known = i + n
        if known >= len(df):
            continue
        out.append({"idx": i, "kind": kind, "price": float(price), "time": df["time"].iat[i],
                    "known_time": df["time"].iat[known] + pd.Timedelta(minutes=minutes),
                    "atr": float(df["atr"].iat[known]) if not np.isnan(df["atr"].iat[known]) else np.nan})
    return out


def alternate(pivots, replaced=None):
    """Make highs and lows alternate (keep the more extreme of consecutive same-type pivots).

    When a later pivot replaces an earlier one, the earlier one was still a real, visible
    swing until the later one was confirmed. `replaced` collects those, stamped with
    `replaced_at`, so target-finding can see the chart as it looked at the time.
    """
    out = []
    for p in pivots:
        if out and out[-1]["kind"] == p["kind"]:
            better = p["price"] > out[-1]["price"] if p["kind"] == "high" else p["price"] < out[-1]["price"]
            if better:
                if replaced is not None:
                    prev = out[-2]["price"] if len(out) >= 2 else np.nan
                    replaced.append({**out[-1], "replaced_at": p["known_time"], "prev_price": prev})
                out[-1] = p
        else:
            out.append(p)
    return out


# ------------------------------------------------------------------- legs
def build_legs(piv, thr_for):
    """thr_for(pivot) -> dollars a leg must reach to count as significant."""
    if len(piv) < 3:
        return []
    legs = []
    leg = {"dir": -1 if piv[0]["kind"] == "high" else 1, "start": piv[0], "extreme": piv[1],
           "flip_time": None, "reach_time": None}
    counter = piv[0]
    if abs(piv[1]["price"] - piv[0]["price"]) >= thr_for(piv[1]):
        leg["reach_time"] = piv[1]["known_time"]
    for k in range(2, len(piv)):
        p = piv[k]
        with_trend = (p["kind"] == "low") if leg["dir"] == -1 else (p["kind"] == "high")
        if with_trend:
            better = p["price"] < leg["extreme"]["price"] if leg["dir"] == -1 else p["price"] > leg["extreme"]["price"]
            if better:
                leg["extreme"] = p
                if leg["reach_time"] is None and abs(p["price"] - leg["start"]["price"]) >= thr_for(p):
                    leg["reach_time"] = p["known_time"]
        else:
            broke = p["price"] > counter["price"] if leg["dir"] == -1 else p["price"] < counter["price"]
            if broke:
                leg["flip_time"] = p["known_time"]
                legs.append(leg)
                leg = {"dir": -leg["dir"], "start": leg["extreme"], "extreme": p,
                       "flip_time": None, "reach_time": None}
                counter = piv[k - 1]
                if abs(p["price"] - leg["start"]["price"]) >= thr_for(p):
                    leg["reach_time"] = p["known_time"]
            else:
                counter = p
    legs.append(leg)
    return legs


def active_start_time(legs, t):
    """Start time of the leg that is active at time t (a leg flips when its end is confirmed)."""
    flips = [lg["flip_time"] for lg in legs[:-1]]
    k = int(np.searchsorted(np.array(flips, dtype="datetime64[ns]"), np.datetime64(t), side="right")) if flips else 0
    return legs[k]["start"]["time"]


# ----------------------------------------------------------------- points
def leg_points(legs, tf, thr_for):
    pts = []
    for k in range(len(legs) - 1):                      # completed legs only
        lg, nxt = legs[k], legs[k + 1]
        end = lg["extreme"]
        arrive = abs(lg["start"]["price"] - end["price"]) >= thr_for(end)
        times = [t for t in (lg["flip_time"] if arrive else None, nxt["reach_time"]) if t is not None]
        if not times:
            continue
        pts.append({"tf": tf, "kind": end["kind"], "level": end["price"], "time": end["time"],
                    "known_time": end["known_time"], "qual_time": max(min(times), lg["flip_time"], end["known_time"]),
                    "source": "leg", "merged": 1})
    return pts


def swing_points(piv, tf, thr_for, gone=()):
    """`gone`: pivots later replaced by a more extreme one. The swing that moved far enough
    AWAY from p may be one of those; the final list only keeps its replacement, which was
    confirmed later, so using only that delayed the level's qualification in hindsight."""
    pts = []
    for k in range(1, len(piv)):
        p = piv[k]
        arrive = abs(p["price"] - piv[k - 1]["price"]) >= thr_for(p)
        times = [p["known_time"]] if arrive else []
        if k + 1 < len(piv):
            nxt = piv[k + 1]
            away = [q for q in gone if q["kind"] == nxt["kind"] and p["known_time"] < q["known_time"] < nxt["known_time"]
                    and q["idx"] > p["idx"]]
            away.append(nxt)
            hit = [q["known_time"] for q in away if abs(q["price"] - p["price"]) >= thr_for(p)]
            if hit:
                times.append(min(hit))
        if times:
            pts.append({"tf": tf, "kind": p["kind"], "level": p["price"], "time": p["time"],
                        "known_time": p["known_time"], "qual_time": max(min(times), p["known_time"]),
                        "source": "swing", "merged": 1})
    return pts


# ----------------------------------------------------------------- sweeps
class Sweeper:
    def __init__(self, m5):
        self.t = m5["time"].to_numpy()
        self.h = m5["high"].to_numpy()
        self.l = m5["low"].to_numpy()

    def first_exceed(self, kind, level, pen, from_time):
        s = int(np.searchsorted(self.t, np.datetime64(from_time), side="left"))
        arr = self.h[s:s + MAX_LOOK] if kind == "high" else self.l[s:s + MAX_LOOK]
        if len(arr) == 0:
            return pd.NaT, np.nan
        hit = arr > level + pen if kind == "high" else arr < level - pen
        if not hit.any():
            return pd.NaT, np.nan
        j = int(hit.argmax())
        return pd.Timestamp(self.t[s + j]), float(arr[j])


def add_sweep(point, sweeper, atr_of):
    pen = SWEEP_PEN_ATR * atr_of(point["tf"], point["known_time"])
    start = max(point["known_time"], point["qual_time"])
    point["sweep_time"], point["extreme"] = sweeper.first_exceed(point["kind"], point["level"], pen, start)


def merge_equal(points, sweeper, atr_of, history=None):
    """Merge near-equal unswept levels of the same timeframe and type into one at the extreme."""
    # p only becomes a level once it has QUALIFIED, so that is the moment it can merge.
    # Merging at its earlier known_time rewrote q's level (and so its sweep) for a stretch
    # of time when p was not yet a level. (Found by the generic truncation test, 28 Oct.)
    def ready(r):
        return max(r["known_time"], r["qual_time"])

    out = []
    for p in sorted(points, key=ready):
        at = ready(p)
        tol = EQUAL_TOL_ATR * atr_of(p["tf"], p["known_time"])
        merged = False
        for q in reversed(out):
            if q["tf"] != p["tf"] or q["kind"] != p["kind"] or abs(q["level"] - p["level"]) > tol:
                continue
            if pd.notna(q["sweep_time"]) and q["sweep_time"] <= at:
                continue                                   # q was already swept before p existed
            # Before changing q, keep a copy of how it looked until p appeared. Without this,
            # a decision made in between sees q with p's later known_time and so cannot see
            # q at all - the future rewriting the past. (Found by the truncation test.)
            snap = dict(q)
            snap["valid_until"] = at
            if history is not None:
                history.append(snap)
            take_p = p["level"] > q["level"] if p["kind"] == "high" else p["level"] < q["level"]
            if take_p:
                q["level"], q["time"] = p["level"], p["time"]
            q["known_time"] = max(q["known_time"], at)
            q["qual_time"] = max(q["known_time"], min(q["qual_time"], p["qual_time"]))
            q["merged"] += p["merged"]
            add_sweep(q, sweeper, atr_of)
            merged = True
            break
        if not merged:
            out.append(p)
    return out


def count_equal_pivots(points, piv_by_tf, atr15_at):
    """How many swing points of the same type sit within EQUAL_PIVOT_TOL x ATR15 of the level.

    This looks at EVERY swing point on 5M/15M/30M, not only the ones big enough to be
    liquidity in their own right, because a cluster of small equal highs/lows is exactly
    the "equal highs" / "equal lows" pattern that makes a smaller sweep worth trading.
    """
    for p in points:
        if pd.isna(p["sweep_time"]):
            p["n_equal"] = 1
            continue
        tol = EQUAL_PIVOT_TOL * atr15_at(p["sweep_time"])
        window = p["sweep_time"] - pd.Timedelta(hours=EQUAL_PIVOT_HOURS)
        seen = []
        for tf in LEG_TFS:
            for q in piv_by_tf[tf]:
                if q["kind"] != p["kind"] or q["known_time"] > p["sweep_time"] or q["time"] < window:
                    continue
                if abs(q["price"] - p["level"]) > tol:
                    continue
                if not any(abs(q["time"] - s) < pd.Timedelta(minutes=20) for s in seen):
                    seen.append(q["time"])          # same pivot seen on 2 timeframes counts once
        p["n_equal"] = max(len(seen), 1)
    return points


def classify(points, legs_by_tf):
    for p in points:
        p["eligible"], p["rank"] = False, ""
        if pd.isna(p["sweep_time"]):
            continue
        start = active_start_time(legs_by_tf[p["tf"]], p["sweep_time"])
        age_h = (p["sweep_time"] - p["time"]).total_seconds() / 3600
        p["age_hours"] = round(age_h, 1)
        p["eligible"] = bool(p["time"] <= start and age_h <= LOOKBACK_HOURS[p["tf"]])
        p["rank"] = "origin" if p["time"] == start else "older"
    return points


def add_travel(points, m5, atr15_at):
    """How far price moved AWAY from the level before coming back to sweep it.

    A level price left behind and only returned to much later is "resting liquidity";
    one it never really left is just recent noise. Measured in multiples of ATR15.
    """
    t = m5["time"].to_numpy()
    hi, lo = m5["high"].to_numpy(), m5["low"].to_numpy()
    for p in points:
        p["travel_atr"] = np.nan
        if pd.isna(p["sweep_time"]):
            continue
        a = int(np.searchsorted(t, np.datetime64(p["known_time"]), side="left"))
        b = int(np.searchsorted(t, np.datetime64(p["sweep_time"]), side="left"))
        if b <= a:
            continue
        # distance price pulled back from the level before returning to sweep it
        away = p["level"] - float(lo[a:b].min()) if p["kind"] == "high" else float(hi[a:b].max()) - p["level"]
        atr = atr15_at(p["sweep_time"])
        if atr and not np.isnan(atr):
            p["travel_atr"] = round(max(away, 0.0) / atr, 2)
    return points


def visible_pivots(piv_by_tf, tf, t):
    """Pivots of `tf` as they appeared at time t: confirmed by t and not yet replaced."""
    now = [p for p in piv_by_tf.get(tf, []) if p["known_time"] <= t]
    old = [p for p in piv_by_tf.get(tf + "_replaced", []) if p["known_time"] <= t < p["replaced_at"]]
    return now + old


def replaced_swing_points(gone, tf, thr_for):
    """1H/4H swing levels that were on the chart for a while and then replaced by a more
    extreme swing of the same type. They are not trigger levels, but until replaced_at they
    were real levels a trader could see, so the opposing-level check must count them.
    Leaving them out let a later swing erase an earlier level from the past.
    (Found by the generic truncation test in backtest-engine, 28 Oct 2026.)"""
    pts = []
    for p in gone:
        if np.isnan(p["atr"]) or np.isnan(p["prev_price"]):
            continue
        if abs(p["price"] - p["prev_price"]) < thr_for(p):
            continue                        # never big enough to count as a level
        pts.append({"tf": tf, "kind": p["kind"], "level": p["price"], "time": p["time"],
                    "known_time": p["known_time"], "qual_time": p["known_time"],
                    "source": "replaced_swing", "valid_until": p["replaced_at"]})
    return pts


def build_levels(tfs, m5):
    """tfs: {name: candles with ATR}. Returns (points DataFrame, legs_by_tf, alternating pivots by tf)."""
    atr15_t = tfs["15M"]["time"].to_numpy()
    atr15_v = tfs["15M"]["atr"].to_numpy()

    def atr_of(tf, t):
        df = tfs[tf]
        k = int(np.searchsorted(df["time"].to_numpy(), np.datetime64(t), side="right")) - 1
        return float(df["atr"].iat[max(k, 0)]) if k >= 0 else np.nan

    def atr15_at(t):
        k = int(np.searchsorted(atr15_t, np.datetime64(t), side="right")) - 1
        return float(atr15_v[k]) if k >= 0 else np.nan

    sweeper = Sweeper(m5)
    legs_by_tf, piv_by_tf, points, ghosts = {}, {}, [], []
    for tf in LEG_TFS + SIG_TFS:
        gone = []
        piv = [p for p in alternate(raw_pivots(tfs[tf], tf), gone) if not np.isnan(p["atr"])]
        piv_by_tf[tf] = piv
        piv_by_tf[tf + "_replaced"] = gone
        if tf in LEG_TFS:
            thr_for = lambda p: LEG_ATR_MULT * atr15_at(p["known_time"])
            legs = build_legs(piv, thr_for)
            legs_by_tf[tf] = legs
            points += leg_points(legs, tf, thr_for)
        else:
            thr_for = lambda p, tf=tf: SWING_ATR_MULT * p["atr"]
            legs_by_tf[tf] = build_legs(piv, lambda p: 0.0)
            points += swing_points(piv, tf, thr_for, [q for q in gone if not np.isnan(q["atr"])])
            ghosts += replaced_swing_points(gone, tf, thr_for)
    for p in points + ghosts:
        add_sweep(p, sweeper, atr_of)
    history = list(ghosts)
    points = merge_equal(points, sweeper, atr_of, history)
    points = classify(points, legs_by_tf)
    points = count_equal_pivots(points, piv_by_tf, atr15_at)
    points = add_travel(points, m5, atr15_at)
    cols = ["tf", "kind", "level", "time", "known_time", "qual_time", "source", "merged",
            "sweep_time", "extreme", "eligible", "rank", "age_hours", "n_equal", "travel_atr"]
    df = pd.DataFrame(points)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df = df[cols].sort_values("known_time").reset_index(drop=True)
    # earlier states of merged levels, for checks that must see the past as it was
    df.attrs["history"] = pd.DataFrame(history, columns=["tf", "kind", "level", "known_time", "qual_time",
                                                         "sweep_time", "valid_until", "source"])
    return df, legs_by_tf, piv_by_tf
