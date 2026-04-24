import MetaTrader5 as mt5
from datetime import datetime

import pytz

from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries


class MT5MarketDataProvider:

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def fetch(self, symbol: str, timeframe, bars: int) -> MarketSeries:
        """Fetches the most recent 'n' bars."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars)
        if rates is None:
            raise ValueError(f"No data for {symbol}")

        return self._process_rates(rates)

    def fetch_range(self, symbol: str, timeframe, date_from: datetime, date_to: datetime) -> MarketSeries:
        # 1. Ensure symbol is visible/selected in MT5
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not found or not selectable. Check your Exness account type.")

        # 2. Convert local datetime to UTC (Exness requirement)
        utc_tz = pytz.utc
        # If dates don't have timezone info, assume they were meant to be UTC
        if date_from.tzinfo is None:
            date_from = utc_tz.localize(date_from)
        if date_to.tzinfo is None:
            date_to = utc_tz.localize(date_to)

        # 3. Fetch rates
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

        if rates is None or len(rates) == 0:
            # Troubleshooting: Try to fetch the last 100 bars to see if the connection is even alive
            fallback = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10)
            if fallback is None:
                error = mt5.last_error()
                raise ValueError(f"CRITICAL: Cannot reach {symbol}. MT5 Error: {error}")
            else:
                raise ValueError(
                    f"History Gap: {symbol} is alive, but no data exists between {date_from} and {date_to}. Try scrolling back the M1 chart in MT5.")

        return self._process_rates(rates)

    def _process_rates(self, rates) -> MarketSeries:
        """Internal helper to convert MT5 rates into the MarketSeries format."""
        candles = [
            Candle(
                time=datetime.fromtimestamp(r["time"]),
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["tick_volume"]
            )
            for r in rates
        ]
        return MarketSeries(candles)