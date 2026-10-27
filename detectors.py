"""
Detectors for the strategy building blocks. They only READ data.

- Fair value gaps (FVG): bullish when candle 3's low is above candle 1's high;
  bearish when candle 3's high is below candle 1's low. Tracked as tapped /
  filled / inverted. Gaps smaller than MIN_GAP_ATR x ATR are ignored.
- Liquidity sweeps: price trades beyond a confirmed swing high or low for the
  first time (by at least MIN_SWEEP_ATR x ATR).
- Inversion FVG (IFVG): a candle BODY closes through the far side of a gap.

Everything is in broker SERVER time. A gap is only "known" once its third candle
has closed, and a swing only N candles after it forms. The backtester must respect that.
"""
import numpy as np
import pandas as pd

from swings import find_swings, resample

ATR_PERIOD = 14
MIN_GAP_ATR = 0.10
MIN_SWEEP_ATR = 0.05
LOOKAHEAD = 3000   # how many candles ahead we follow each gap / level

TF_MINUTES = {"1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240}
TF_RULES = {"15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
SWING_N = {"1M": 3, "5M": 3, "15M": 2, "30M": 2, "1H": 2, "4H": 2}


def add_atr(df, period=ATR_PERIOD):
    prev_close = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"],
                            (df["high"] - prev_close).abs(),
                            (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    out = df.copy()
    out["atr"] = true_range.rolling(period).mean()
    return out


def build_timeframes(m5, m1=None):
    """Return {timeframe name: candles with an ATR column}."""
    tfs = {"5M": add_atr(m5)}
    for name, rule in TF_RULES.items():
        tfs[name] = add_atr(resample(m5, rule))
    if m1 is not None:
        tfs["1M"] = add_atr(m1)
    return tfs


def _first(mask, offset):
    return offset + int(mask.argmax()) if mask.any() else None


def find_fvgs(df, tf, lookahead=LOOKAHEAD):
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    times = df["time"].to_numpy()
    minutes = TF_MINUTES[tf]

    def when(idx):
        return pd.NaT if idx is None else pd.Timestamp(times[idx])

    rows = []
    bull = np.where(l[2:] > h[:-2])[0] + 2
    bear = np.where(h[2:] < l[:-2])[0] + 2
    for direction, idxs in (("bull", bull), ("bear", bear)):
        for i in idxs:
            if np.isnan(atr[i]):
                continue
            if direction == "bull":
                bottom, top = h[i - 2], l[i]
            else:
                bottom, top = h[i], l[i - 2]
            gap = top - bottom
            if gap < MIN_GAP_ATR * atr[i]:
                continue
            sl = slice(i + 1, i + 1 + lookahead)
            if direction == "bull":
                tap, fill, inv = l[sl] <= top, l[sl] <= bottom, c[sl] < bottom
            else:
                tap, fill, inv = h[sl] >= bottom, h[sl] >= top, c[sl] > top
            rows.append({
                "tf": tf, "direction": direction, "top": top, "bottom": bottom,
                "gap_size": gap, "atr": atr[i], "formed_idx": i,
                "formed_time": pd.Timestamp(times[i]) + pd.Timedelta(minutes=minutes),
                "tap_time": when(_first(tap, i + 1)),
                "fill_time": when(_first(fill, i + 1)),
                "invert_time": when(_first(inv, i + 1)),
            })
    cols = ["tf", "direction", "top", "bottom", "gap_size", "atr", "formed_idx",
            "formed_time", "tap_time", "fill_time", "invert_time"]
    return pd.DataFrame(rows, columns=cols).sort_values("formed_time").reset_index(drop=True)


def find_sweeps(df, tf, lookahead=LOOKAHEAD):
    """One row per confirmed swing; sweep_time is NaT while the level is still unswept."""
    n = SWING_N[tf]
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    times = df["time"].to_numpy()
    is_high, is_low = find_swings(df, n)

    rows = []
    for kind, flags in (("high", is_high), ("low", is_low)):
        for s in np.where(flags)[0]:
            known = s + n                       # swing is only known after n more candles
            if known >= len(df) - 1 or np.isnan(atr[known]):
                continue
            level = h[s] if kind == "high" else l[s]
            pen = MIN_SWEEP_ATR * atr[known]
            start = known + 1
            sl = slice(start, start + lookahead)
            hit = (h[sl] > level + pen) if kind == "high" else (l[sl] < level - pen)
            j = _first(hit, start)
            row = {"tf": tf, "kind": kind, "level": level,
                   "swing_time": pd.Timestamp(times[s]),
                   "known_time": pd.Timestamp(times[known]) + pd.Timedelta(minutes=TF_MINUTES[tf]),
                   "sweep_time": pd.NaT, "extreme": np.nan, "closed_back_inside": np.nan}
            if j is not None:
                row["sweep_time"] = pd.Timestamp(times[j])
                row["extreme"] = h[j] if kind == "high" else l[j]
                row["closed_back_inside"] = bool(c[j] <= level) if kind == "high" else bool(c[j] >= level)
            rows.append(row)
    cols = ["tf", "kind", "level", "swing_time", "known_time", "sweep_time", "extreme", "closed_back_inside"]
    return pd.DataFrame(rows, columns=cols).sort_values("swing_time").reset_index(drop=True)
