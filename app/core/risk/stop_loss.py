from app.core.strategy.trade_signal import StrategySignal


class StopLossCalculator:

    def calculate(self, signal: StrategySignal) -> float:
        if signal.signal == "BUY":
            return min(signal.candle.low - (1.0 * signal.atr), signal.sl) if signal.sl else signal.candle.low - (1.0 * signal.atr)
        else:
            return max(signal.candle.high + (1.0 * signal.atr), signal.sl) if signal.sl else signal.candle.high + (1.0 * signal.atr)
