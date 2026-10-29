"""
Setup candidates. It only READS data and never places trades.

A setup candidate = a trigger (eligible sweep of a liquidity level, see levels.py)
followed by a reversal confirmation on the 5M or 1M:
  a fair value gap that formed AFTER the sweep started is closed through by a candle
  body (an inversion gap, IFVG). For a swept HIGH the trade is a SHORT, for a swept LOW a LONG.

Tags (used later to test what matters; none of them removes a setup):
  window_extreme  the swept level was the highest high / lowest low of its lookback window
  htf_tap         highest timeframe (15M/30M/1H/4H) FVG tapped around the sweep, in the sweep direction
  vs_1h, vs_4h    "with" or "against" the running 1H / 4H trend (trade direction vs trend direction)
  rank            "origin" (start of the current leg) or "older" level

Entry variants:  A = at the close of the inversion candle, B = retrace into the inverted gap,
                 C = retrace into a new gap that forms after the inversion (near edge of the zone).
Stop anchor = the extreme of the sweep move up to the entry. Take profits = the nearest
unswept 5M/15M swing points in the trade direction (nearest first), your LRL rule.
"""
import numpy as np
import pandas as pd

import levels as lv
from detectors import TF_MINUTES

TF_RANK = {"5M": 0, "15M": 1, "30M": 2, "1H": 3, "4H": 4}
HTF_FVG_TFS = ("15M", "30M", "1H", "4H")
ENTRY_TFS = ("5M", "1M")
CLUSTER_MIN = 60        # sweeps of the same type this close together are one setup
MAX_WAIT_MIN = 120      # the inversion must come within this time of the sweep
TAP_BEFORE_MIN = 60     # an HTF gap tapped up to this long before the sweep still counts
RETRACE_WAIT_MIN = 60   # how long we wait for a retrace entry (variants B and C)
ENTRY_GAP_ATR = 0.25    # entry gaps must be at least this x ATR of their timeframe


def _idx(df, t):
    return int(np.searchsorted(df["time"].to_numpy(), np.datetime64(t), side="left"))


def window_extreme(points, p):
    look = pd.Timedelta(hours=lv.LOOKBACK_HOURS[p["tf"]])
    q = points[(points["kind"] == p["kind"]) & (points["qual_time"] <= p["sweep_time"]) &
               (points["time"] >= p["sweep_time"] - look) & (points.index != p.name)]
    q = q[q["sweep_time"].isna() | (q["sweep_time"] >= p["sweep_time"])]
    return not (q["level"] > p["level"]).any() if p["kind"] == "high" else not (q["level"] < p["level"]).any()


def make_event(points, kind, members):
    """One trigger event from a cluster of sweeps of the same type."""
    key = lambda p: (TF_RANK[p["tf"]], p["level"] if kind == "high" else -p["level"])
    primary = max(members, key=key)
    tfs = sorted({m["tf"] for m in members}, key=lambda x: TF_RANK[x])
    first = min(members, key=lambda m: m["sweep_time"])
    # sweep_time and level MUST come from the same member, or the setup claims a level
    # was swept at a moment when price never reached it.
    return {"sweep_time": primary["sweep_time"], "kind": kind,
            "first_sweep_time": first["sweep_time"], "first_level": float(first["level"]),
            "level": float(primary["level"]), "top_tf": primary["tf"], "tfs": "+".join(tfs),
            "n_levels": len(members), "equal_level": any(m["merged"] >= 2 for m in members),
            "window_extreme": any(window_extreme(points, m) for m in members),
            "rank": "origin" if any(m["rank"] == "origin" for m in members) else "older",
            "age_hours": float(primary["age_hours"])}


def sweep_groups(points, t0, t1):
    """Eligible sweeps between t0 and t1, by type, in time order."""
    el = points[points["eligible"] & points["sweep_time"].between(t0, t1)].copy()
    return [(kind, [p for _, p in g.sort_values("sweep_time").iterrows()]) for kind, g in el.groupby("kind")]


def trigger_events(points, t0, t1):
    """Clusters of sweeps as events, ignoring entries (used for charts and checks)."""
    out = []
    for kind, members in sweep_groups(points, t0, t1):
        cluster, last = [], None
        for p in members:
            if last is not None and (p["sweep_time"] - last) > pd.Timedelta(minutes=CLUSTER_MIN):
                out.append(make_event(points, kind, cluster))
                cluster = []
            cluster.append(p)
            last = p["sweep_time"]
        if cluster:
            out.append(make_event(points, kind, cluster))
    return sorted(out, key=lambda e: e["sweep_time"])


def refine_sweep(ev, tfs):
    """Find the exact minute inside the 5M sweep candle when the level was first exceeded."""
    if "1M" not in tfs:
        return ev["sweep_time"]
    m1 = tfs["1M"]
    a = _idx(m1, ev["sweep_time"])
    b = _idx(m1, ev["sweep_time"] + pd.Timedelta(minutes=5))
    seg = m1.iloc[a:b]
    if seg.empty:
        return ev["sweep_time"]
    hit = (seg["high"] > ev["level"]) if ev["kind"] == "high" else (seg["low"] < ev["level"])
    return pd.Timestamp(seg["time"].iat[int(hit.argmax())]) if hit.any() else ev["sweep_time"]


def active_dir(legs, t):
    flips = [lg["flip_time"] for lg in legs[:-1]]
    k = int(np.searchsorted(np.array(flips, dtype="datetime64[ns]"), np.datetime64(t), side="right")) if flips else 0
    return legs[k]["dir"]


def htf_tap(ev, tfs, fvgs):
    """Highest-timeframe FVG (in the sweep direction) tapped around the sweep."""
    want = "bear" if ev["kind"] == "high" else "bull"
    t0 = ev["sweep_time"] - pd.Timedelta(minutes=TAP_BEFORE_MIN)
    t1 = ev["sweep_time"] + pd.Timedelta(minutes=MAX_WAIT_MIN)
    m5 = tfs["5M"]
    s, e = _idx(m5, t0), _idx(m5, t1)
    hi, lo = m5["high"].to_numpy()[s:e + 1], m5["low"].to_numpy()[s:e + 1]
    tt = m5["time"].to_numpy()[s:e + 1]
    found = []
    for tf in HTF_FVG_TFS:
        g = fvgs[tf]
        g = g[(g["direction"] == want) & (g["formed_time"] <= ev["sweep_time"]) &
              (g["fill_time"].isna() | (g["fill_time"] >= t0))]
        for _, r in g.iterrows():
            if want == "bear" and r["top"] < ev["level"]:
                continue
            if want == "bull" and r["bottom"] > ev["level"]:
                continue
            hit = (hi >= r["bottom"]) if want == "bear" else (lo <= r["top"])
            if hit.any():
                found.append((tf, pd.Timestamp(tt[int(hit.argmax())])))
    if not found:
        return "", pd.NaT, ""
    top = max(found, key=lambda f: TF_RANK[f[0]])[0]
    return top, min(f[1] for f in found), "+".join(sorted({f[0] for f in found}, key=lambda x: TF_RANK[x]))


def find_ifvg(fv, kind, t_start, minutes):
    want = "bull" if kind == "high" else "bear"        # bullish gap closed below = bearish IFVG (short)
    first_candle = fv["formed_time"] - pd.Timedelta(minutes=3 * minutes)
    g = fv[(fv["direction"] == want) & (first_candle >= t_start) & fv["invert_time"].notna() &
           (fv["invert_time"] > t_start) &
           (fv["invert_time"] <= t_start + pd.Timedelta(minutes=MAX_WAIT_MIN)) &
           (fv["gap_size"] >= ENTRY_GAP_ATR * fv["atr"])]
    if g.empty:
        return None
    return g.sort_values(["invert_time", "gap_size"], ascending=[True, False]).iloc[0]


def retrace_entry(df, minutes, kind, zone_edge, start_i, anchor):
    """First touch of zone_edge after start_i within the wait; invalid if the anchor is broken first."""
    n = max(RETRACE_WAIT_MIN // minutes, 1)
    hi, lo = df["high"].to_numpy()[start_i:start_i + n], df["low"].to_numpy()[start_i:start_i + n]
    if len(hi) == 0:
        return pd.NaT, np.nan
    touch = (hi >= zone_edge) if kind == "high" else (lo <= zone_edge)
    broke = (hi > anchor) if kind == "high" else (lo < anchor)
    if not touch.any():
        return pd.NaT, np.nan
    j = int(touch.argmax())
    if broke[:j + 1].any():
        return pd.NaT, np.nan
    return pd.Timestamp(df["time"].iat[start_i + j]), float(zone_edge)


def opposing_liquidity(points, kind, entry_price, entry_time, risk):
    """Unswept liquidity sitting on the WRONG side of the trade: a reason for price to go
    against us before it reaches a target. For a long that is unswept lows below the entry;
    for a short, unswept highs above it.

    Returns the distance to the nearest one (in multiples of risk) for several minimum
    timeframes, so a filter can be chosen later without re-running everything. A higher
    timeframe level is treated as a stronger magnet by only counting levels at or above
    the chosen timeframe.
    """
    # a swept HIGH means we are short, so the danger is price continuing up: unswept highs
    want = "high" if kind == "high" else "low"
    # a level only exists once it has QUALIFIED (qual_time), which can be later than the
    # swing's confirmation (known_time); using known_time let later price action leak in
    live = points[(points["kind"] == want) & (points["qual_time"] <= entry_time) &
                  (points["sweep_time"].isna() | (points["sweep_time"] > entry_time))]
    hist = points.attrs.get("history")
    if hist is not None and len(hist):
        # levels that existed at entry but were later merged into a newer swing
        old = hist[(hist["kind"] == want) & (hist["qual_time"] <= entry_time) &
                   (hist["valid_until"] > entry_time) &
                   (hist["sweep_time"].isna() | (hist["sweep_time"] > entry_time))]
        live = pd.concat([live[["tf", "level"]], old[["tf", "level"]]], ignore_index=True)
    if kind == "high":                               # short: magnets above the entry
        live = live[live["level"] > entry_price]
        dist = live["level"] - entry_price
    else:                                            # long: magnets below the entry
        live = live[live["level"] < entry_price]
        dist = entry_price - live["level"]

    out = {}
    for floor in ("5M", "15M", "30M", "1H"):
        m = live["tf"].map(TF_RANK) >= TF_RANK[floor]
        d = dist[m]
        out[f"opp_dist_{floor}"] = round(float(d.min()) / risk, 2) if len(d) else np.nan
    nearest = dist.idxmin() if len(dist) else None
    out["opp_tf"] = live.loc[nearest, "tf"] if nearest is not None else ""
    return out


def opposing_fvg(fvgs, kind, entry_price, entry_time, risk):
    """Unfilled higher-timeframe fair value gaps on the WRONG side of the trade.

    Same idea as an opposing liquidity level: an unfilled gap is somewhere price may want
    to go, and if it sits against us it can drag price through the stop first.
    """
    found = []                                   # (distance in R, timeframe)
    for tf in HTF_FVG_TFS:
        g = fvgs.get(tf)
        if g is None or g.empty:
            continue
        g = g[(g["formed_time"] <= entry_time) & (g["fill_time"].isna() | (g["fill_time"] > entry_time))]
        if kind == "high":                       # short: danger is price continuing up
            g = g[g["bottom"] > entry_price]
            d = g["bottom"] - entry_price
        else:                                    # long: danger is price falling
            g = g[g["top"] < entry_price]
            d = entry_price - g["top"]
        if len(d):
            found.append((float(d.min()) / risk, tf))

    out = {}
    for floor in ("15M", "30M", "1H"):
        near = [f for f in found if TF_RANK[f[1]] >= TF_RANK[floor]]
        out[f"opp_fvg_{floor}"] = round(min(near)[0], 2) if near else np.nan
    out["opp_fvg_tf"] = min(found)[1] if found else ""
    return out


def take_profits(kind, entry_price, entry_time, tfs, piv_by_tf, atr15):
    """Nearest unswept 5M/15M swing points in the trade direction (LRL), nearest first."""
    m5 = tfs["5M"]
    hi, lo = m5["high"].to_numpy(), m5["low"].to_numpy()
    # number of 5M candles fully CLOSED by the entry; the candle in progress (and the
    # one after it) must not be used to decide whether a target was already swept
    e_i = int(np.searchsorted(m5["time"].to_numpy(), np.datetime64(entry_time - pd.Timedelta(minutes=5)),
                              side="right"))
    want = "low" if kind == "high" else "high"          # short targets lows, long targets highs
    cands = []
    for tf in ("5M", "15M"):
        for p in lv.visible_pivots(piv_by_tf, tf, entry_time):
            if p["kind"] != want:
                continue
            if (want == "low" and p["price"] >= entry_price) or (want == "high" and p["price"] <= entry_price):
                continue
            s = _idx(m5, p["time"])
            seg = lo[s + 1:e_i] if want == "low" else hi[s + 1:e_i]
            swept = len(seg) and ((seg < p["price"]).any() if want == "low" else (seg > p["price"]).any())
            if not swept:
                cands.append(p["price"])
    cands = sorted(set(cands), reverse=(want == "low"))     # nearest first
    tps = []
    for c in cands:
        if not tps or abs(c - tps[-1]) > 0.25 * atr15:
            tps.append(c)
    return (tps + [np.nan] * 3)[:3]


def build_setups(points, tfs, legs_by_tf, piv_by_tf, fvgs, t0, t1):
    """Setups from clusters of sweeps.

    A later sweep joins a cluster (and can become its primary level) only if it happened
    before the cluster's own entry. Otherwise the entry was already taken in real time, and
    letting the later sweep redefine the event erased a trade that a live trader would have
    seen. The later sweep then starts a new cluster. (Found by the generic truncation test
    in backtest-engine, 28 Oct 2026.)
    """
    rows = []
    for kind, members in sweep_groups(points, t0, t1):
        cluster, last, row = [], None, None
        for p in members:
            if last is not None:
                gap = (p["sweep_time"] - last) > pd.Timedelta(minutes=CLUSTER_MIN)
                # the 5M candle holding p's sweep must have CLOSED before the entry to count
                done = row is not None and row["entry_time"] < p["sweep_time"] + pd.Timedelta(minutes=5)
                if gap or done:
                    if row is not None:
                        rows.append(row)
                    cluster = []
            cluster.append(p)
            last = p["sweep_time"]
            row = setup_for(make_event(points, kind, cluster), points, tfs, legs_by_tf, piv_by_tf, fvgs)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r["sweep_time"])
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.insert(0, "id", range(1, len(df_out) + 1))
    return df_out


def setup_for(ev, points, tfs, legs_by_tf, piv_by_tf, fvgs):
    """The setup (one dict) for one trigger event, or None."""
    if True:                                  # (kept indented: the body is the old loop body)
        kind = ev["kind"]
        trade = "short" if kind == "high" else "long"
        trade_dir = -1 if trade == "short" else 1
        tap_tf, tap_time, tap_all = htf_tap(ev, tfs, fvgs)

        sweep_minute = refine_sweep(ev, tfs)
        best = None
        for etf in ENTRY_TFS:
            if etf not in fvgs:
                continue
            t_start = sweep_minute if etf == "1M" else ev["sweep_time"]
            g = find_ifvg(fvgs[etf], kind, t_start, TF_MINUTES[etf])
            if g is None:
                continue
            entry_time = g["invert_time"] + pd.Timedelta(minutes=TF_MINUTES[etf])
            if best is None or entry_time < best[0]:
                best = (entry_time, etf, g)
        if best is None:
            return None
        entry_time, etf, g = best
        df, minutes = tfs[etf], TF_MINUTES[etf]
        i = _idx(df, g["invert_time"])
        s = _idx(df, sweep_minute if etf == "1M" else ev["sweep_time"])
        entry = float(df["close"].iat[i])
        anchor = float(df["high"].to_numpy()[s:i + 1].max() if kind == "high" else df["low"].to_numpy()[s:i + 1].min())
        risk = abs(anchor - entry)
        if risk <= 0:
            return None
        closed15 = int(np.searchsorted(tfs["15M"]["time"].to_numpy(),
                                       np.datetime64(entry_time - pd.Timedelta(minutes=15)), side="right"))
        atr15 = float(tfs["15M"]["atr"].iat[max(closed15 - 1, 0)])
        tps = take_profits(kind, entry, entry_time, tfs, piv_by_tf, atr15)
        rr = [abs(entry - t) / risk if not np.isnan(t) else np.nan for t in tps]

        edge = g["bottom"] if kind == "high" else g["top"]
        b_time, b_price = retrace_entry(df, minutes, kind, edge, i + 1, anchor)
        c_time, c_price = pd.NaT, np.nan
        new = fvgs[etf]
        new = new[(new["direction"] == ("bear" if kind == "high" else "bull")) & (new["formed_time"] > entry_time) &
                  (new["formed_time"] <= entry_time + pd.Timedelta(minutes=RETRACE_WAIT_MIN)) &
                  (new["gap_size"] >= ENTRY_GAP_ATR * new["atr"])]
        for _, ng in new.iterrows():
            ng_edge = ng["bottom"] if kind == "high" else ng["top"]
            t, p = retrace_entry(df, minutes, kind, ng_edge, _idx(df, ng["formed_time"]), anchor)
            if pd.notna(t) and (pd.isna(c_time) or t < c_time):
                c_time, c_price = t, p

        opp = opposing_liquidity(points, kind, entry, entry_time, risk)
        opp.update(opposing_fvg(fvgs, kind, entry, entry_time, risk))
        tp1_dist = abs(tps[0] - entry) / risk if not np.isnan(tps[0]) else np.nan

        return {
            **opp, "tp1_dist_R": round(tp1_dist, 2) if not np.isnan(tp1_dist) else np.nan,
            "sweep_time": ev["sweep_time"], "sweep_minute": sweep_minute, "trade": trade,
            "equal_level": ev["equal_level"], "level": ev["level"], "top_tf": ev["top_tf"],
            "tfs": ev["tfs"], "n_levels": ev["n_levels"], "window_extreme": ev["window_extreme"],
            "rank": ev["rank"], "age_hours": ev["age_hours"],
            "wait_min": round((entry_time - sweep_minute).total_seconds() / 60),
            "gap_usd": round(float(g["top"] - g["bottom"]), 2),
            "gap_atr15": round(float(g["top"] - g["bottom"]) / atr15, 3),
            "gap_risk": round(float(g["top"] - g["bottom"]) / risk, 3),
            "htf_tap": tap_tf, "htf_taps": tap_all, "tap_time": tap_time,
            "tap_before_entry": bool(pd.notna(tap_time) and tap_time <= entry_time),
            "vs_1h": "with" if active_dir(legs_by_tf["1H"], ev["sweep_time"]) == trade_dir else "against",
            "vs_4h": "with" if active_dir(legs_by_tf["4H"], ev["sweep_time"]) == trade_dir else "against",
            "entry_tf": etf, "ifvg_formed": g["formed_time"], "ifvg_invert": g["invert_time"],
            "ifvg_bottom": float(g["bottom"]), "ifvg_top": float(g["top"]),
            "entry_time": entry_time, "entry_price": entry, "stop": anchor, "risk": round(risk, 2),
            "risk_atr15": round(risk / atr15, 2),
            "tp1": tps[0], "tp2": tps[1], "tp3": tps[2],
            "rr1": rr[0], "rr2": rr[1], "rr3": rr[2],
            "entryB_time": b_time, "entryB_price": b_price, "entryC_time": c_time, "entryC_price": c_price,
        }
