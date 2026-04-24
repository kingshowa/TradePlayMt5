class PositionSizer:
    """
    Converts balance risk into position size.
    """

    def calculate(self, balance: float, risk_pct: float, entry: float, stop: float) -> float:
        if balance <= 0:
            raise ValueError("balance must be greater than 0")

        if risk_pct <= 0:
            raise ValueError("risk_pct must be greater than 0")

        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            raise ValueError("stop distance must be greater than 0")

        risk_amount = balance * risk_pct
        return round(risk_amount / stop_distance, 2)
