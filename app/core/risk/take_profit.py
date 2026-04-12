from app.core.strategy.trade_signal import StrategySignal


class TakeProfitCalculator:

    def calculate(self, entry: float, stop: float, rr: float, signal: StrategySignal) -> float:
        risk = abs(entry - stop)
        if signal.signal == "BUY":
            return min(entry + risk * rr, signal.tp) if signal.tp else entry + risk * rr
        else:
            return max(entry - risk * rr, signal.tp) if signal.tp else entry - risk * rr