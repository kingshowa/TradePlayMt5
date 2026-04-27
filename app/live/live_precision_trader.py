import csv
import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5

from app.core.market.candle import Candle
from app.core.strategy.trade_signal import StrategySignal
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.live.mt5_live_executor import MT5LiveExecutor


class LiveTraderPrecisionPsar:
    """
    Tick-driven live trader for PrecisionPsarStrategy.

    Execution contract:

    1. Warm up strategy with historical closed candles.
    2. On every tick:
         - If flat: call strategy.on_tick(price, forming_candle)
           to detect precision PSAR-dot crossing.
         - If a BUY/SELL signal is returned, build trade from the latest
           account balance/equity and send it to MT5.
    3. On every newly closed candle:
         - Append candle to local series.
         - Call strategy.update(closed_candle) to refresh PSAR/EMA/ATR.
         - If a trade is open, trail SL with the confirmed PSAR value.
    4. No strategy CLOSE signal is used. Exits are broker SL/TP/manual close,
       plus PSAR SL trailing.

    Important:
    - For live trading, strategy should be initialized with:
        entry_on_update=False
        use_close_signal=False
    - strategy.on_tick() must use only previously confirmed indicator state.
    """

    def __init__(
        self,
        symbol: str,
        strategy_cls,
        risk_manager,
        timeframe: str = "1m",
        bars: int = 500,
        strategy_kwargs: Optional[dict] = None,
        provider: Optional[MT5MarketDataProvider] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        lot_precision: int = 2,
        deviation: int = 20,
        magic: int = 123456,
        comment: str = "PrecisionPSARLive",
        poll_seconds: float = 0.25,
        closed_candle_poll_seconds: float = 1.0,
        use_tp: bool = True,
        use_equity: bool = False,
        log_file: str = "live_precision_psar_trades_log.csv",
        tick_price_mode: str = "SIDE",  # SIDE, MID, BID, ASK
    ):
        self.symbol = symbol
        self.timeframe_name = timeframe
        self.timeframe = TIMEFRAMES[timeframe]
        self.initial_bars = bars
        self.strategy_cls = strategy_cls
        self.strategy_kwargs = dict(strategy_kwargs or {})
        self.risk_manager = risk_manager
        self.provider = provider or MT5MarketDataProvider()

        # Force live-safe strategy settings unless user explicitly set them.
        self.strategy_kwargs.setdefault("entry_on_update", False)
        self.strategy_kwargs.setdefault("use_close_signal", False)

        self.use_tp = use_tp
        self.use_equity = use_equity
        self.poll_seconds = poll_seconds
        self.closed_candle_poll_seconds = closed_candle_poll_seconds
        self.log_file = log_file
        self.tick_price_mode = tick_price_mode.upper()

        if self.tick_price_mode not in {"SIDE", "MID", "BID", "ASK"}:
            raise ValueError("tick_price_mode must be one of: SIDE, MID, BID, ASK")

        self.series = None
        self.strategy = None
        self.current_trade = None
        self.last_closed_candle_time = None
        self._last_closed_candle_check_monotonic = 0.0

        self.executor = MT5LiveExecutor(
            symbol=symbol,
            lot_precision=lot_precision,
            deviation=deviation,
            magic=magic,
            comment=comment,
            login=login,
            password=password,
            server=server,
            auto_connect=True,
        )

        self._ensure_symbol_selected()
        self._load_initial_history()
        self._bootstrap_trade_state()

    # ---------------------------------------------------
    # Startup / shutdown
    # ---------------------------------------------------

    def shutdown(self) -> None:
        self.executor.shutdown()

    def _ensure_symbol_selected(self) -> None:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found in MT5: {self.symbol}")
        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise RuntimeError(f"Failed to select symbol in MT5 Market Watch: {self.symbol}")

    def _load_initial_history(self) -> None:
        self.series = self.provider.fetch(
            self.symbol,
            self.timeframe,
            self.initial_bars,
        )
        self.strategy = self.strategy_cls(self.series, **self.strategy_kwargs)
        print(f"✅ Loaded {len(self.series._candles)} candles for warm-up")
        print(f"✅ Strategy initialized with live-safe kwargs: {self.strategy_kwargs}")

    def _bootstrap_trade_state(self) -> None:
        position = self.executor.get_open_position()
        if position is None:
            self.current_trade = None
            return

        trade = self._runtime_trade_from_position(position)

        if not self._direction_aligns_with_current_trend(trade.direction):
            print(
                f"⚠ Existing MT5 position found but not synced because it is not aligned: "
                f"ticket={trade.ticket} direction={trade.direction} "
                f"psar_trend={self.strategy.state().get('psar_trend')}"
            )
            self.current_trade = None
            return

        self.current_trade = trade
        self.strategy.sync_trade(trade)
        print(
            f"ℹ Synced existing aligned MT5 position: "
            f"ticket={trade.ticket} direction={trade.direction} entry={trade.entry}"
        )

    # ---------------------------------------------------
    # Market data helpers
    # ---------------------------------------------------

    def _get_tick(self):
        return self.executor.get_symbol_tick()

    def _get_tick_entry_price(self, tick, expected_direction: Optional[str] = None) -> Optional[float]:
        if tick is None:
            return None

        bid = float(tick.bid)
        ask = float(tick.ask)

        if self.tick_price_mode == "BID":
            return bid
        if self.tick_price_mode == "ASK":
            return ask
        if self.tick_price_mode == "MID":
            return (bid + ask) / 2.0

        # SIDE mode: use the executable side when known, otherwise mid.
        if expected_direction == "BUY":
            return ask
        if expected_direction == "SELL":
            return bid
        return bid

    def _get_latest_closed_candle(self) -> Optional[Candle]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, 1)
        if rates is None or len(rates) == 0:
            return None

        rate = rates[0]
        return Candle(
            time=datetime.fromtimestamp(rate["time"]),
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            volume=float(rate["tick_volume"]),
        )

    def _get_current_forming_candle(self) -> Optional[Candle]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
        if rates is None or len(rates) == 0:
            return None

        rate = rates[0]
        return Candle(
            time=datetime.fromtimestamp(rate["time"]),
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            volume=float(rate["tick_volume"]),
        )

    # ---------------------------------------------------
    # Account / risk helpers
    # ---------------------------------------------------

    def _get_balance_for_sizing(self) -> float:
        account = self.executor.get_account_info()
        if account is None:
            raise RuntimeError("Unable to read MT5 account info")
        return float(account.equity if self.use_equity else account.balance)

    # ---------------------------------------------------
    # Trade lifecycle
    # ---------------------------------------------------

    def _open_trade(self, signal) -> None:
        balance = self._get_balance_for_sizing()
        trade = self.risk_manager.build_trade(signal=signal, balance=balance)

        if trade is None:
            print("⚠ Risk manager rejected the signal; no trade opened")
            return

        if not self.use_tp:
            trade.take_profit = None

        ticket = self.executor.send_trade(trade, use_tp=self.use_tp)
        if ticket is None:
            print("⚠ MT5 executor did not open the trade")
            return

        if hasattr(trade, "set_ticket"):
            trade.set_ticket(ticket)
        else:
            trade.ticket = ticket

        if not self.use_tp:
            trade.take_profit = 0.0

        self.current_trade = trade
        self.strategy.sync_trade(trade)

        self._log_trade_event(trade, event="OPEN", extra=getattr(signal, "reason", ""))

        print(
            f"🚀 Opened {trade.direction}: ticket={ticket} entry={trade.entry} "
            f"SL={trade.stop_loss} TP={getattr(trade, 'take_profit', None)} "
            f"size={trade.position_size}"
        )

    def _check_trade_closed(self) -> bool:
        return self.current_trade is not None and self.executor.get_position_by_ticket(int(self.current_trade.ticket)) is None

    def _handle_trade_closed(self) -> None:
        if self.current_trade is None:
            return

        print("✅ Trade closed on broker side")
        self._log_trade_event(self.current_trade, event="CLOSE", extra="BrokerClose")

        # Accurate SL/TP classification can be added later from deal history.
        self.strategy.on_trade_closed(exit_type="BrokerClose")
        self.current_trade = None

    def _trail_stop_with_confirmed_psar(self) -> None:
        if self.current_trade is None:
            return

        psar_sl = self.strategy.get_psar_sl(self.current_trade.direction)
        if psar_sl is None:
            return

        result = self.executor.trail_position_with_psar(
            position_ticket=int(self.current_trade.ticket),
            direction=self.current_trade.direction,
            psar_sl=psar_sl,
            use_tp=self.use_tp,
        )

        if result is not None:
            self.current_trade.stop_loss = float(psar_sl)
            if not self.use_tp:
                self.current_trade.take_profit = 0.0
            print(f"🔄 Confirmed PSAR trailing applied: SL -> {psar_sl:.5f}")

    def _runtime_trade_from_position(self, position):
        """Build a lightweight trade object from an existing MT5 position."""
        class RuntimeTrade:
            pass

        trade = RuntimeTrade()
        trade.ticket = int(position.ticket)
        trade.direction = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
        trade.entry = float(position.price_open)
        trade.stop_loss = float(position.sl)
        trade.take_profit = float(position.tp)
        trade.position_size = float(position.volume)
        return trade

    def _direction_aligns_with_current_trend(self, direction: str) -> bool:
        """
        Only adopt/trail broker positions that agree with the current confirmed PSAR trend.

        BUY  aligns with PSAR trend UP.
        SELL aligns with PSAR trend DOWN.
        """
        state = self.strategy.state() if self.strategy is not None else {}
        psar_trend = str(state.get("psar_trend", "")).upper()
        direction = str(direction).upper()

        if direction == "BUY":
            return psar_trend == "UP"
        if direction == "SELL":
            return psar_trend == "DOWN"
        return False

    def _adopt_external_position_if_aligned(self) -> bool:
        """
        Detect an MT5 position that exists while the app has no current_trade.

        This can happen when:
        - the trade was opened manually,
        - another script opened it,
        - app state was cleared without restarting MT5,
        - the trader missed an internal state sync.

        We adopt it only when it aligns with the current confirmed PSAR trend.
        """
        if self.current_trade is not None:
            return True

        position = self.executor.get_open_position()
        if position is None:
            return False

        trade = self._runtime_trade_from_position(position)

        if not self._direction_aligns_with_current_trend(trade.direction):
            print(
                f"⚠ External position detected but not adopted: "
                f"ticket={trade.ticket} direction={trade.direction} "
                f"psar_trend={self.strategy.state().get('psar_trend')}"
            )
            return False

        self.current_trade = trade
        self.strategy.sync_trade(trade)
        self._log_trade_event(trade, event="SYNC", extra="Adopted aligned broker position")

        print(
            f"ℹ Adopted aligned broker position: "
            f"ticket={trade.ticket} direction={trade.direction} entry={trade.entry}"
        )
        return True

    def _copy_signal_with_entry_price(self, signal, entry_price: float):
        """Return a new StrategySignal with a rebuilt Candle using executable price."""
        base = signal.candle
        updated_candle = Candle(
            time=base.time,
            open=base.open,
            high=max(base.high, entry_price),
            low=min(base.low, entry_price),
            close=entry_price,
            volume=base.volume,
        )

        return StrategySignal(
            signal=signal.signal,
            strategy_type=signal.strategy_type,
            reason=signal.reason,
            pattern_name=signal.pattern_name,
            atr=signal.atr,
            sl=signal.sl,
            tp=signal.tp,
            candle=updated_candle,
        )

    # ---------------------------------------------------
    # Strategy event processing
    # ---------------------------------------------------

    def _process_tick_entry(self) -> None:
        if self.current_trade is not None:
            return
        if self.executor.has_open_position():
            return

        tick = self._get_tick()
        if tick is None:
            return

        forming_candle = self._get_current_forming_candle()

        trend = self.strategy.state().get("psar_trend")
        # print(trend)
        # First use a neutral price to allow the strategy to detect direction.
        price = self._get_tick_entry_price(tick)
        signal = self.strategy.on_tick(price=price, candle=forming_candle)

        if signal is None:
            return

        signal_name = str(getattr(signal, "signal", "")).upper()
        if signal_name not in {"BUY", "SELL"}:
            return

        # Rebuild the signal with executable side price if SIDE mode is enabled.
        # Candle is frozen in this project, so never mutate signal.candle in place.
        if self.tick_price_mode == "SIDE":
            executable_price = self._get_tick_entry_price(tick, expected_direction=signal_name)
            if executable_price is not None:
                signal = self._copy_signal_with_entry_price(signal, executable_price)

        print(f"🎯 Precision signal detected: {signal_name} | {getattr(signal, 'reason', '')}")
        self._open_trade(signal)

    def _process_closed_candle_if_new(self) -> None:
        candle = self._get_latest_closed_candle()
        if candle is None:
            return

        if self.last_closed_candle_time == candle.time:
            return

        self.last_closed_candle_time = candle.time
        self.series.append_from_mt5_rate(candle)

        # update() refreshes confirmed PSAR/EMA/ATR only. It must not open trades live.
        signal = self.strategy.update(candle)
        if signal is not None and getattr(signal, "signal", None) in {"BUY", "SELL"}:
            print("⚠ Ignored update() entry signal. Live trader only enters via on_tick().")

        if self.current_trade is not None:
            if self._check_trade_closed():
                self._handle_trade_closed()
            else:
                self._trail_stop_with_confirmed_psar()

    # ---------------------------------------------------
    # Logging
    # ---------------------------------------------------

    def _log_trade_event(self, trade, event: str, extra: str = "") -> None:
        file_exists = False
        try:
            with open(self.log_file, "r", newline=""):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "LoggedAt",
                    "Event",
                    "Symbol",
                    "Direction",
                    "Entry",
                    "StopLoss",
                    "TakeProfit",
                    "PositionSize",
                    "Ticket",
                    "Extra",
                ])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                event,
                self.symbol,
                getattr(trade, "direction", None),
                getattr(trade, "entry", None),
                getattr(trade, "stop_loss", None),
                getattr(trade, "take_profit", None),
                getattr(trade, "position_size", None),
                getattr(trade, "ticket", None),
                extra,
            ])

    # ---------------------------------------------------
    # Main loop
    # ---------------------------------------------------

    def run(self) -> None:
        print(
            f"🔥 Precision PSAR live trading started: {self.symbol} {self.timeframe_name} "
            f"use_tp={self.use_tp} tick_price_mode={self.tick_price_mode}"
        )

        try:
            while True:
                # Broker close can happen between candles. Detect it quickly.
                if self.current_trade is not None and self._check_trade_closed():
                    self._handle_trade_closed()

                # If MT5 has a position but app state is empty, adopt it only if
                # it agrees with the current confirmed PSAR trend, then trail it.
                self._adopt_external_position_if_aligned()

                # Tick-level precision entries happen continuously while flat.
                self._process_tick_entry()

                # Closed-candle indicator refresh + confirmed PSAR trailing.
                self._process_closed_candle_if_new()

                # Also attempt trailing after adoption even before a new candle closes.
                if self.current_trade is not None:
                    self._trail_stop_with_confirmed_psar()

                time.sleep(self.poll_seconds)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.shutdown()


if __name__ == "__main__":
    from app.core.strategy.precision_psar_strategy import PrecisionPsarStrategy
    from app.core.risk.risk_manager import RiskManager

    trader = LiveTraderPrecisionPsar(
        symbol="BTCUSDm",
        timeframe="1m",
        bars=500,
        strategy_cls=PrecisionPsarStrategy,
        strategy_kwargs={
            "psar_step": 0.025,
            "psar_max_step": 0.05,
            "atr_period": 14,
            "use_ema_trend": True,
            "ema_trend_period": 50,
            "ema_offset": 3,
            "ema_slope_threshold": 0.0,
            "require_trigger_ema_side": True,
            "use_ema_distance_filter": False,
            "entry_on_update": False,
            "use_close_signal": False,
        },
        risk_manager=RiskManager(
            risk_pct=0.01,
            rr=2.0,
            atr_multiplier=1.5,
            sl_mode="WIDER",
        ),
        use_tp=True,
        use_equity=False,
        poll_seconds=0.25,
        tick_price_mode="SIDE",
    )
    trader.run()
