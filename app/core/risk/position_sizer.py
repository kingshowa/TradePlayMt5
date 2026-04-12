class PositionSizer:

    def calculate(self, balance: float, risk_pct: float, entry: float, stop: float) -> float:
        risk_amount = balance * risk_pct
        stop_distance = abs(entry - stop)
        return round(risk_amount / stop_distance, 2)
