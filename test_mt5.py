"""
Connection test: checks that Python can talk to your MT5 desktop terminal,
reads the latest gold candles, and shows how much 5M and 1M history you have.
It only READS data. It never places trades.
"""
import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD.vx"
ACCOUNT_TYPES = {0: "DEMO", 1: "CONTEST", 2: "LIVE"}

if not mt5.initialize():
    print("Could not connect to MT5:", mt5.last_error())
    print("Make sure the MT5 desktop terminal is open and logged in.")
    quit()

terminal = mt5.terminal_info()
account = mt5.account_info()
print("Terminal:", terminal.name)
print("Server:", account.server)
print("Account type:", ACCOUNT_TYPES.get(account.trade_mode, "unknown"))

if not mt5.symbol_select(SYMBOL, True):
    print("Symbol not found:", SYMBOL, mt5.last_error())
    mt5.shutdown()
    quit()

info = mt5.symbol_info(SYMBOL)
tick = mt5.symbol_info_tick(SYMBOL)
print(f"Symbol: {SYMBOL} | digits: {info.digits} | spread now (points): {info.spread}")
print(f"Latest bid/ask: {tick.bid} / {tick.ask}")

for name, timeframe in [("M5", mt5.TIMEFRAME_M5), ("M1", mt5.TIMEFRAME_M1)]:
    rates = None
    # The terminal rejects requests bigger than its "max bars" setting,
    # so try large counts first and fall back to smaller ones.
    for count in (500000, 200000, 100000, 50000, 20000, 10000, 5000):
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count)
        if rates is not None and len(rates) > 0:
            break
        print(f"{name}: request for {count} candles failed:", mt5.last_error())
    if rates is None or len(rates) == 0:
        print(f"{name}: no data returned at any size")
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"\n{name}: {len(df)} candles")
    print(f"  earliest: {df['time'].iloc[0]}")
    print(f"  latest:   {df['time'].iloc[-1]}")
    if name == "M5":
        print("\nLast 5 five-minute candles:")
        print(df.tail(5)[["time", "open", "high", "low", "close", "tick_volume"]].to_string(index=False))

print("\nNote: times above are your broker's SERVER time, not your local time.")
mt5.shutdown()
