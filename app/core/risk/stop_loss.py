from typing import Optional, Tuple

from app.core.strategy.trade_signal import StrategySignal


class StopLossCalculator:
    """
    Stop-loss selector compatible with the current backtest model.

    Supported modes:
    - WIDER   -> choose the farther valid stop
    - TIGHTER -> choose the closer valid stop
    - PSAR    -> use PSAR stop only
    - ATR     -> use ATR stop only
    """

    VALID_SL_MODES = {"WIDER", "TIGHTER", "PSAR", "ATR"}

    def __init__(self, atr_multiplier: float = 1.5, sl_mode: str = "WIDER"):
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be greater than 0")

        sl_mode = sl_mode.upper()
        if sl_mode not in self.VALID_SL_MODES:
            raise ValueError(f"Invalid sl_mode. Use one of {self.VALID_SL_MODES}")

        self.atr_multiplier = atr_multiplier
        self.sl_mode = sl_mode

    def calculate(self, signal: StrategySignal, entry: Optional[float] = None) -> Tuple[Optional[float], str, Optional[float], Optional[float]]:
        direction = signal.signal.upper()
        candle = signal.candle
        entry = candle.close if entry is None else entry
        atr = signal.atr
        psar_sl = signal.sl

        if direction not in ("BUY", "SELL"):
            return None, "NONE", None, None

        if atr is None or atr <= 0:
            atr_sl = None
        else:
            atr_sl = self._atr_stop(direction, candle, atr)

        final_sl, sl_source = self._select_stop(direction, entry, psar_sl, atr_sl)
        return final_sl, sl_source, psar_sl, atr_sl

    def _atr_stop(self, direction: str, candle, atr: float) -> float:
        if direction == "BUY":
            return candle.low - (self.atr_multiplier * atr)
        return candle.high + (self.atr_multiplier * atr)

    def _select_stop(
        self,
        direction: str,
        entry: float,
        psar_sl: Optional[float],
        atr_sl: Optional[float],
    ) -> Tuple[Optional[float], str]:
        candidates = []

        if psar_sl is not None and self._valid_stop(direction, entry, psar_sl):
            candidates.append(("PSAR", psar_sl))

        if atr_sl is not None and self._valid_stop(direction, entry, atr_sl):
            candidates.append(("ATR", atr_sl))

        if not candidates:
            return None, "NONE"

        if self.sl_mode == "PSAR":
            for source, value in candidates:
                if source == "PSAR":
                    return value, source
            return None, "NONE"

        if self.sl_mode == "ATR":
            for source, value in candidates:
                if source == "ATR":
                    return value, source
            return None, "NONE"

        if direction == "BUY":
            if self.sl_mode == "WIDER":
                source, value = min(candidates, key=lambda item: item[1])
            else:
                source, value = max(candidates, key=lambda item: item[1])
        else:
            if self.sl_mode == "WIDER":
                source, value = max(candidates, key=lambda item: item[1])
            else:
                source, value = min(candidates, key=lambda item: item[1])

        return value, source

    @staticmethod
    def _valid_stop(direction: str, entry: float, stop: float) -> bool:
        if direction == "BUY":
            return stop < entry
        return stop > entry
