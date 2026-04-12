# from collections import deque
# from typing import Deque, Optional, List
#
# from app.core.indicators.base_indicator import BaseIndicator
# from app.core.market.candle import Candle
# from app.core.market.market_series import MarketSeries
#
#
# class HeikinAshiIndicator(BaseIndicator):
#     def __init__(self, max_candles: int = 500):
#         super().__init__(period=1)
#         self.max_candles = max_candles
#         self._ha_candles: Deque[Candle] = deque(maxlen=max_candles)
#
#         self.current_value: Optional[Candle] = None
#         self.previous_value: Optional[Candle] = None
#
#     def calculate(self, series: MarketSeries) -> Candle:
#         if len(series) == 0:
#             raise ValueError("MarketSeries cannot be empty")
#
#         self.reset()
#         raw_candles = series.candles()
#
#         prev_ha_open = None
#         prev_ha_close = None
#
#         for raw in raw_candles:
#             ha = self._build_ha_candle(raw, prev_ha_open, prev_ha_close)
#             self._ha_candles.append(ha)
#             prev_ha_open = ha.open
#             prev_ha_close = ha.close
#
#         self._sync_state()
#         return self.current_value
#
#     def update(self, new_candle: Candle) -> Candle:
#         if not self._ha_candles:
#             ha = self._build_ha_candle(new_candle, None, None)
#             self._ha_candles.append(ha)
#             self._sync_state()
#             return ha
#
#         last_ha = self._ha_candles[-1]
#
#         # Prevent duplicate append if same candle time arrives again
#         if last_ha.time == new_candle.time:
#             self._ha_candles.pop()
#
#             prev_ha = self._ha_candles[-1] if self._ha_candles else None
#             ha = self._build_ha_candle(
#                 new_candle,
#                 prev_ha.open if prev_ha else None,
#                 prev_ha.close if prev_ha else None
#             )
#             self._ha_candles.append(ha)
#         else:
#             ha = self._build_ha_candle(
#                 new_candle,
#                 last_ha.open,
#                 last_ha.close
#             )
#             self._ha_candles.append(ha)
#
#         self._sync_state()
#         return ha
#
#     def _build_ha_candle(
#         self,
#         raw_candle: Candle,
#         prev_ha_open: Optional[float],
#         prev_ha_close: Optional[float]
#     ) -> Candle:
#         ha_close = (
#             raw_candle.open +
#             raw_candle.high +
#             raw_candle.low +
#             raw_candle.close
#         ) / 4
#
#         ha_open = (
#             (raw_candle.open + raw_candle.close) / 2
#             if prev_ha_open is None or prev_ha_close is None
#             else (prev_ha_open + prev_ha_close) / 2
#         )
#
#         ha_high = max(raw_candle.high, ha_open, ha_close)
#         ha_low = min(raw_candle.low, ha_open, ha_close)
#
#         return Candle(
#             time=raw_candle.time,
#             open=ha_open,
#             high=ha_high,
#             low=ha_low,
#             close=ha_close,
#             volume=getattr(raw_candle, "volume", 0)
#         )
#
#     def _sync_state(self):
#         self.current_value = self._ha_candles[-1] if self._ha_candles else None
#         self.previous_value = self._ha_candles[-2] if len(self._ha_candles) > 1 else None
#
#     def values(self) -> List[Candle]:
#         return list(self._ha_candles)
#
#     def last(self, n: int = 1):
#         if not self._ha_candles:
#             raise ValueError("No Heikin Ashi candles available")
#
#         if n == 1:
#             return self._ha_candles[-1]
#
#         if n > len(self._ha_candles):
#             raise ValueError(f"Requested {n} candles, but only {len(self._ha_candles)} available")
#
#         return list(self._ha_candles)[-n:]
#
#     def as_market_series(self) -> MarketSeries:
#         return MarketSeries(self.values())
#
#     def reset(self):
#         self._ha_candles.clear()
#         self.current_value = None
#         self.previous_value = None