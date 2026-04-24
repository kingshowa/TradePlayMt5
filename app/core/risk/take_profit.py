from app.core.strategy.trade_signal import StrategySignal


class TakeProfitCalculator:

    def calculate(self, entry: float, stop: float, rr: float, signal: StrategySignal) -> float:
        if rr <= 0:
            raise ValueError("rr must be greater than 0")

        risk = abs(entry - stop)
        if risk <= 0:
            raise ValueError("risk distance must be greater than 0")

        if signal.signal.upper() == "BUY":
            return entry + (risk * rr)
        return entry - (risk * rr)
