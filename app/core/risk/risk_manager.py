from app.core.risk.position_sizer import PositionSizer
from app.core.risk.stop_loss import StopLossCalculator
from app.core.risk.take_profit import TakeProfitCalculator
from app.core.strategy.trade_object import Trade
from app.core.strategy.trade_signal import StrategySignal


class RiskManager:

    def __init__(self, risk_pct: float = 0.01, rr: float = 2):
        self.risk_pct = risk_pct
        self.rr = rr

        self.sizer = PositionSizer()
        self.sl_calc = StopLossCalculator()
        self.tp_calc = TakeProfitCalculator()

    def build_trade(self, signal: StrategySignal, balance: float):
        entry = signal.candle.close
        stop = self.sl_calc.calculate(signal)
        size = self.sizer.calculate(balance, self.risk_pct, entry, stop)
        tp = self.tp_calc.calculate(entry, stop, self.rr, signal)

        data = {
            "direction": signal.signal,
            "entry": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "position_size": size,
            "candle": signal.candle,
            "pattern_name": signal.pattern_name,
            "reason": signal.reason,
        }

        trade = Trade(**data)

        return trade
