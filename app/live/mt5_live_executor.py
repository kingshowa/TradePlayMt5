import MetaTrader5 as mt5
from typing import Optional


class MT5LiveExecutor:
    """
    MT5 execution adapter for live trading.

    Responsibilities:
    - connect / shutdown
    - inspect open positions
    - send trades with optional TP
    - modify SL/TP
    - trail SL using PSAR only
    """

    def __init__(
        self,
        symbol: str,
        lot_precision: int = 2,
        deviation: int = 20,
        magic: int = 123456,
        comment: str = "LiveBotPSAR",
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        auto_connect: bool = True,
    ):
        self.symbol = symbol
        self.lot_precision = lot_precision
        self.deviation = deviation
        self.magic = magic
        self.comment = comment
        self.login = login
        self.password = password
        self.server = server

        if auto_connect:
            self.connect()

    # ----------------------------------------
    # CONNECTION
    # ----------------------------------------

    def connect(self) -> None:
        kwargs = {}
        if self.login is not None:
            kwargs["login"] = self.login
        if self.password is not None:
            kwargs["password"] = self.password
        if self.server is not None:
            kwargs["server"] = self.server

        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

        print("✅ Connected to MT5")

    def shutdown(self) -> None:
        mt5.shutdown()
        print("🛑 MT5 shutdown")

    # ----------------------------------------
    # ACCOUNT / SYMBOL HELPERS
    # ----------------------------------------

    def get_account_info(self):
        return mt5.account_info()

    def get_symbol_info(self):
        return mt5.symbol_info(self.symbol)

    def get_symbol_tick(self):
        return mt5.symbol_info_tick(self.symbol)

    # ----------------------------------------
    # POSITION CHECK
    # ----------------------------------------

    def has_open_position(self) -> bool:
        positions = mt5.positions_get(symbol=self.symbol)
        return positions is not None and len(positions) > 0

    def get_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            return positions[0]
        return None

    def get_position_by_ticket(self, ticket: int):
        positions = mt5.positions_get(ticket=int(ticket))
        if positions:
            return positions[0]
        return None

    def get_position_from_result(self, result):
        if result is None or getattr(result, "deal", None) is None:
            return None

        deals = mt5.history_deals_get(ticket=result.deal)
        if not deals:
            return None

        pos_id = deals[0].position_id
        positions = mt5.positions_get(ticket=pos_id)
        return positions[0] if positions else None

    # ----------------------------------------
    # OPEN TRADE
    # ----------------------------------------

    def send_trade(self, trade, use_tp: bool = True) -> Optional[int]:
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            print(f"❌ Symbol {self.symbol} not found")
            return None

        tick = self.get_symbol_tick()
        if tick is None:
            print("❌ No tick data")
            return None

        point = symbol_info.point
        digits = symbol_info.digits
        min_stop_distance = symbol_info.trade_stops_level * point

        direction = str(trade.direction).upper()
        sl = round(float(trade.stop_loss), digits)

        tp = None
        trade_tp = getattr(trade, "take_profit", None)
        if use_tp and trade_tp is not None:
            tp = round(float(trade_tp), digits)

        if direction == "BUY":
            price = float(tick.ask)
            if (price - sl) < min_stop_distance:
                print(f"❌ BUY SL too close. price={price} sl={sl} min={min_stop_distance}")
                return None
            if tp is not None and (tp - price) < min_stop_distance:
                print(f"❌ BUY TP too close. price={price} tp={tp} min={min_stop_distance}")
                return None
            order_type = mt5.ORDER_TYPE_BUY

        elif direction == "SELL":
            price = float(tick.bid)
            if (sl - price) < min_stop_distance:
                print(f"❌ SELL SL too close. price={price} sl={sl} min={min_stop_distance}")
                return None
            if tp is not None and (price - tp) < min_stop_distance:
                print(f"❌ SELL TP too close. price={price} tp={tp} min={min_stop_distance}")
                return None
            order_type = mt5.ORDER_TYPE_SELL

        else:
            print(f"❌ Unsupported trade direction: {trade.direction}")
            return None

        volume = round(float(trade.position_size), self.lot_precision)
        if volume <= 0:
            print("❌ Computed volume is not positive")
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)
        if result is None:
            print(f"❌ order_send returned None: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: retcode={result.retcode} comment={result.comment}")
            return None

        position = self.get_position_from_result(result)
        if position is None:
            print("⚠ Order placed, but failed to resolve live position")
            return None

        print(f"🚀 Trade opened successfully @ {price} volume={volume}")
        return int(position.ticket)

    # ----------------------------------------
    # MODIFY SL/TP
    # ----------------------------------------

    def modify_position(self, position_ticket: int, new_sl: Optional[float] = None, new_tp: Optional[float] = None):
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            print(f"❌ Symbol {self.symbol} not found")
            return None

        current_position = self.get_position_by_ticket(position_ticket)
        if current_position is None:
            print(f"⚠ Position {position_ticket} not found")
            return None

        digits = symbol_info.digits
        sl = current_position.sl if new_sl is None else round(float(new_sl), digits)

        if new_tp is None:
            tp = current_position.tp
        elif float(new_tp) == 0:
            tp = 0.0
        else:
            tp = round(float(new_tp), digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": int(position_ticket),
            "sl": sl,
            "tp": tp,
        }

        result = mt5.order_send(request)
        if result is None:
            print(f"❌ Modify returned None: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Modify failed: retcode={result.retcode} comment={result.comment}")
            return None

        print(f"✅ Position updated: SL={sl} TP={tp}")
        return result

    # ----------------------------------------
    # PSAR TRAILING
    # ----------------------------------------

    def trail_position_with_psar(self, position_ticket: int, direction: str, psar_sl: Optional[float], use_tp: bool = True):
        if psar_sl is None:
            return None

        position = self.get_position_by_ticket(position_ticket)
        if position is None:
            return None

        symbol_info = self.get_symbol_info()
        tick = self.get_symbol_tick()
        if symbol_info is None or tick is None:
            return None

        digits = symbol_info.digits
        min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
        current_sl = float(position.sl)
        psar_sl = round(float(psar_sl), digits)
        direction = str(direction).upper()

        if direction == "BUY":
            if psar_sl <= current_sl:
                return None

            market_price = float(tick.bid)
            if (market_price - psar_sl) < min_stop_distance:
                return None

        elif direction == "SELL":
            if current_sl != 0 and psar_sl >= current_sl:
                return None

            market_price = float(tick.ask)
            if (psar_sl - market_price) < min_stop_distance:
                return None

        else:
            print(f"❌ Unsupported direction for PSAR trailing: {direction}")
            return None

        tp_value = position.tp if use_tp else 0
        return self.modify_position(
            position_ticket=int(position.ticket),
            new_sl=psar_sl,
            new_tp=tp_value,
        )
