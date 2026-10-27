"""
Asks your MT5 terminal what it offers and how much history it holds. Reads only.

    python find_symbols.py                 list likely index symbols
    python find_symbols.py --all           list everything the broker offers
    python find_symbols.py --check NAS100  how much history that symbol has

MT5 must be open and logged in.
"""
import sys

import MetaTrader5 as mt5
import pandas as pd

HINTS = ("NAS", "NDX", "US100", "US500", "SPX", "US30", "DJ", "GER", "DAX", "UK100",
         "FTSE", "JP225", "NIK", "HK50", "AUS200", "EU50", "STOXX", "INDEX", "IDX")


def history_span(name):
    for tf, label in ((mt5.TIMEFRAME_M5, "M5"), (mt5.TIMEFRAME_M1, "M1")):
        rates = None
        for count in (500000, 200000, 100000, 50000, 20000, 5000):
            rates = mt5.copy_rates_from_pos(name, tf, 0, count)
            if rates is not None and len(rates):
                break
        if rates is None or not len(rates):
            print(f"   {label}: no data")
            continue
        t = pd.to_datetime(pd.DataFrame(rates)["time"], unit="s")
        years = (t.iloc[-1] - t.iloc[0]).days / 365.25
        print(f"   {label}: {len(t):>7,} candles, {t.iloc[0].date()} to {t.iloc[-1].date()} ({years:.1f} years)")


def main():
    if not mt5.initialize():
        print("Could not connect to MT5:", mt5.last_error())
        print("Open the MT5 desktop terminal and log in first.")
        return

    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        if not mt5.symbol_select(name, True):
            print("Symbol not found:", name, mt5.last_error())
        else:
            info = mt5.symbol_info(name)
            print(f"{name}: digits {info.digits}, spread {info.spread} points, "
                  f"contract size {info.trade_contract_size}, min lot {info.volume_min}")
            print("  pulling history (this may take a minute)...")
            history_span(name)
        mt5.shutdown()
        return

    symbols = mt5.symbols_get()
    print(f"Broker offers {len(symbols)} symbols in total.\n")
    show_all = "--all" in sys.argv
    picked = [s for s in symbols
              if show_all or any(h in s.name.upper() for h in HINTS)]
    if not picked:
        print("No obvious index symbols found. Run with --all to see the full list.")
    else:
        print(f"{'symbol':<18} {'description':<42} {'digits':>6} {'spread':>7}")
        print("-" * 78)
        for s in sorted(picked, key=lambda x: x.name):
            print(f"{s.name:<18} {s.description[:42]:<42} {s.digits:>6} {s.spread:>7}")
        print(f"\n{len(picked)} shown. Next:  python find_symbols.py --check <symbol>")
    mt5.shutdown()


if __name__ == "__main__":
    main()
