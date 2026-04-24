from typing import Optional

from app.core.risk.position_sizer import PositionSizer
from app.core.risk.stop_loss import StopLossCalculator
from app.core.risk.take_profit import TakeProfitCalculator
from app.core.strategy.trade_object import Trade
from app.core.strategy.trade_signal import StrategySignal


class RiskManager:
    """
    Live-trading risk manager aligned with the current backtest flow.

    Key behavior:
    - sl_mode is fixed when the manager is initialized.
    - balance is supplied when each trade is created.
    - stop-loss can be selected from PSAR / ATR according to the mode.
    """

    def __init__(
        self,
        risk_pct: float = 0.01,
        rr: float = 2.0,
        atr_multiplier: float = 1.5,
        sl_mode: str = "WIDER",
    ):
        if risk_pct <= 0:
            raise ValueError("risk_pct must be greater than 0")

        if rr <= 0:
            raise ValueError("rr must be greater than 0")

        self.risk_pct = risk_pct
        self.rr = rr
        self.atr_multiplier = atr_multiplier
        self.sl_mode = sl_mode.upper()

        self.sizer = PositionSizer()
        self.sl_calc = StopLossCalculator(
            atr_multiplier=self.atr_multiplier,
            sl_mode=self.sl_mode,
        )
        self.tp_calc = TakeProfitCalculator()

    def build_trade(self, signal: StrategySignal, balance: float) -> Optional[Trade]:
        direction = signal.signal.upper()
        if direction not in ("BUY", "SELL"):
            return None

        entry = signal.candle.close
        stop, sl_source, psar_sl, atr_sl = self.sl_calc.calculate(signal, entry=entry)
        if stop is None:
            return None

        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return None

        risk_amount = balance * self.risk_pct
        position_size = self.sizer.calculate(balance, self.risk_pct, entry, stop)
        take_profit = self.tp_calc.calculate(entry, stop, self.rr, signal)

        trade = Trade(
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=take_profit,
            position_size=position_size,
            candle=signal.candle,
            pattern_name=signal.pattern_name,
            reason=signal.reason,
        )

        # Optional metadata: preserved when Trade supports dynamic attributes.
        extra_fields = {
            "risk_amount": risk_amount,
            "stop_distance": stop_distance,
            "rr": self.rr,
            "sl_source": sl_source,
            "psar_sl": psar_sl,
            "atr_sl": atr_sl,
        }

        for field_name, value in extra_fields.items():
            try:
                setattr(trade, field_name, value)
            except Exception:
                pass

        return trade
