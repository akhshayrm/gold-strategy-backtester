"""
Saves candle history for one symbol from the MT5 desktop terminal into CSV files.
It only READS data. It never places trades.

    python download_history.py            full download (do this first)
    python download_history.py --update   adds the newest candles to the files

Files are saved in a "data" folder next to this script.
Times are your broker's SERVER time (not converted).
"""
import os
import sys

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = os.environ.get("AGENT_SYMBOL", "XAUUSD.vx")
for _a in sys.argv[1:]:                      # or: python download_history.py --symbol NAS100
    if _a.startswith("--symbol="):
        SYMBOL = _a.split("=", 1)[1]
if "--symbol" in sys.argv:
    SYMBOL = sys.argv[sys.argv.index("--symbol") + 1]
TIMEFRAMES = {"M5": mt5.TIMEFRAME_M5, "M1": mt5.TIMEFRAME_M1}
CHUNK = 100000       # candles requested per call
MAX_CHUNKS = 30      # safety limit (30 x 100000 candles)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def to_frame(rates):
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def clean(df):
    """Sort, remove duplicates, and drop the newest candle (it is still forming)."""
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df.iloc[:-1]


def fetch_all(timeframe):
    parts = []
    for i in range(MAX_CHUNKS):
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, i * CHUNK, CHUNK)
        if rates is None or len(rates) == 0:
            if i == 0:
                print("  No data returned:", mt5.last_error())
            break
        parts.append(to_frame(rates))
        if len(rates) < CHUNK:
            break
    if not parts:
        return None
    return clean(pd.concat(parts))


def report_gaps(df):
    """Weekends (about 48h) and the daily break are normal. Flag gaps of 2 to 40 hours."""
    diffs = df["time"].diff()
    mask = (diffs > pd.Timedelta(hours=2)) & (diffs < pd.Timedelta(hours=40))
    gaps = df[mask].copy()
    gaps["gap_hours"] = diffs[mask].dt.total_seconds() / 3600
    print(f"  Unusual gaps (2 to 40 hours): {len(gaps)}")
    for _, row in gaps.sort_values("gap_hours", ascending=False).head(5).iterrows():
        print(f"    {row['gap_hours']:.1f} hours ending at {row['time']}")


def main():
    update = "--update" in sys.argv

    if not mt5.initialize():
        print("Could not connect to MT5:", mt5.last_error())
        print("Make sure the MT5 desktop terminal is open and logged in.")
        return
    if not mt5.symbol_select(SYMBOL, True):
        print("Symbol not found:", SYMBOL, mt5.last_error())
        mt5.shutdown()
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    for name, timeframe in TIMEFRAMES.items():
        path = os.path.join(DATA_DIR, f"{SYMBOL.replace('.', '_')}_{name}.csv")
        print(f"\n{name}:")

        if update and os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["time"])
            rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, 5000)
            if rates is None or len(rates) == 0:
                print("  No data returned:", mt5.last_error())
                continue
            df = clean(pd.concat([old, to_frame(rates)]))
            print(f"  Added {len(df) - len(old)} new candles.")
        else:
            df = fetch_all(timeframe)
            if df is None:
                continue

        df.to_csv(path, index=False)
        print(f"  {len(df)} candles saved to {path}")
        print(f"  From {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        if not update:
            report_gaps(df)

    mt5.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
