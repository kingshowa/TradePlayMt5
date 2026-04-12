from app.core.market.candle import Candle

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Trade:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    position_size: float
    candle: Candle
    pattern_name: str
    reason: str

    open: bool = True
    max_favorable_price: float = None
    bars_open: int = 0
    time: datetime = field(default_factory=datetime.now)
    ticket: int = None

    def __post_init__(self):
        self.max_favorable_price = self.entry

    def set_ticket(self, ticket):
        self.ticket = ticket


# @dataclass
# class Trade:
#
#     def __init__(self, direction, entry, stop_loss, take_profit, position_size, candle: Candle, pattern_name):
#         self.direction = direction
#         self.entry = entry
#         self.stop_loss = stop_loss
#         self.take_profit = take_profit
#         self.position_size = position_size
#         self.candle = candle
#         self.open = True
#         self.max_favorable_price = entry
#         self.bars_open = 0
#         self.time = datetime.now()
#         self.pattern_name = pattern_name
#         self.ticket: int
#
#     def set_ticket(self, ticket):
#         self.ticket = ticket