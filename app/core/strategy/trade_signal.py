# app/core/strategy/models.py

from dataclasses import dataclass
from typing import Literal, Optional

from app.core.market.candle import Candle

SignalType = Literal["BUY", "SELL", "CLOSE"]
StrategyType = Literal["TRENDING", "RANGING"]

@dataclass
class StrategySignal:
    signal: SignalType
    strategy_type: StrategyType
    reason: str
    pattern_name: str
    atr: Optional[float]
    tp: Optional[float]
    sl: Optional[float]
    candle: Optional[Candle]