import csv
from datetime import datetime
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES

# =========================
# Configuration
# =========================
SYMBOL = "XAUUSDm"  # Gold
TIMEFRAME_KEY = "1m"  # 1 Minute timeframe
START_DATE = datetime(2024, 1, 1)  # Year, Month, Day
END_DATE = datetime(2026, 3, 30)
OUTPUT_FILENAME = "gold_historical_data.csv"


def save_range_to_csv(symbol, timeframe, start, end, filename):
    print(f"🚀 Fetching {symbol} ({timeframe}) from {start.date()} to {end.date()}...")

    # Initialize the provider
    provider = MT5MarketDataProvider()

    # 1. Use fetch_range as implemented in parabolic_sar_backtest1m.py
    series = provider.fetch_range(
        symbol,
        TIMEFRAMES[timeframe],
        start,
        end
    )

    candles = series.candles()

    if not candles:
        print("❌ No data found for the selected range.")
        return

    print(f"📝 Saving {len(candles)} candles to {filename}...")

    # 2. Define headers based on the Candle dataclass structure
    header = ["Time", "Open", "High", "Low", "Close", "Volume"]

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for candle in candles:
            # 3. Write each candle row
            writer.writerow([
                candle.time,
                f"{candle.open:.5f}",
                f"{candle.high:.5f}",
                f"{candle.low:.5f}",
                f"{candle.close:.5f}",
                candle.volume
            ])

    print(f"✅ Success! Data saved to {filename}")


if __name__ == "__main__":
    save_range_to_csv(SYMBOL, TIMEFRAME_KEY, START_DATE, END_DATE, OUTPUT_FILENAME)