import csv
import time
from datetime import datetime
from typing import Optional

from app.core.market.candle import Candle
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.live.mt5_live_executor import MT5LiveExecutor


class LiveTrader:
    """
    Live trader aligned with the backtest flow.

    Candle-by-candle execution contract (mirrors the backtest loop):

      1. strategy.update(candle)          -> signal or None
      2. If trade open:
           a. Check if broker closed it   -> on_trade_closed + clear state
           b. Trail SL via get_psar_sl()  -> unconditional, every candle
      3. If flat and signal is BUY/SELL:
           a. Open trade via executor
           b. strategy.sync_trade(trade)  -> keep strategy state in sync

    Key design decisions:
    - Trailing is driven by strategy.get_psar_sl(), NOT by the signal
      object. Signal objects only exist on entry/exit candles; PSAR
      moves every candle and must be applied every candle.
    - strategy.sync_trade() is called in both live and backtest after
      every trade open, so direction state and close-signal logic are
      always consistent.
    - strategy.on_trade_closed() is called when the broker confirms the
      position is gone, mirroring the backtest calling it on SL/TP/close.
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
        comment: str = "LiveBotPSAR",
        poll_seconds: int = 1,
        use_tp: bool = True,
        use_equity: bool = False,
        log_file: str = "live_trades_log.csv",
    ):
        self.symbol = symbol
        self.timeframe_name = timeframe
        self.timeframe = TIMEFRAMES[timeframe]
        self.initial_bars = bars
        self.strategy_cls = strategy_cls
        self.strategy_kwargs = dict(strategy_kwargs or {})
        self.risk_manager = risk_manager
        self.provider = provider or MT5MarketDataProvider()

        self.use_tp = use_tp
        self.use_equity = use_equity
        self.poll_seconds = poll_seconds
        self.log_file = log_file

        self.series = None
        self.strategy = None
        self.current_trade = None
        self.last_candle_time = None

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

        self._load_initial_history()
        self._bootstrap_trade_state()

    # ---------------------------------------------------
    # Startup
    # ---------------------------------------------------

    def shutdown(self) -> None:
        self.executor.shutdown()

    def _load_initial_history(self) -> None:
        self.series = self.provider.fetch(
            self.symbol,
            self.timeframe,
            self.initial_bars,
        )
        self.strategy = self.strategy_cls(self.series, **self.strategy_kwargs)
        print(f"✅ Loaded {len(self.series._candles)} candles for warm-up")

    def _bootstrap_trade_state(self) -> None:
        """
        On startup, sync any position that is already open on the broker
        side (e.g. bot restarted mid-trade). Reconstructs a minimal trade
        object so trailing and close detection work immediately.
        """
        position = self.executor.get_open_position()
        if position is None:
            self.current_trade = None
            return

        class RuntimeTrade:
            pass

        trade = RuntimeTrade()
        trade.ticket = int(position.ticket)
        trade.direction = "BUY" if position.type == 0 else "SELL"
        trade.entry = float(position.price_open)
        trade.stop_loss = float(position.sl)
        trade.take_profit = float(position.tp)
        trade.position_size = float(position.volume)

        self.current_trade = trade

        # Keep strategy direction state consistent so close-signal logic works
        self.strategy.sync_trade(trade)

        print(f"ℹ Synced existing MT5 position: ticket={trade.ticket} dir={trade.direction}")

    # ---------------------------------------------------
    # Account helpers
    # ---------------------------------------------------

    def _get_balance_for_sizing(self) -> float:
        account = self.executor.get_account_info()
        if account is None:
            raise RuntimeError("Unable to read MT5 account info")
        return float(account.equity if self.use_equity else account.balance)

    def _get_latest_closed_candle(self) -> Optional[Candle]:
        import MetaTrader5 as mt5

        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, 1)

        if rates is None or len(rates) == 0:
            return None

        rate = rates[0]
        return Candle(
            time=datetime.fromtimestamp(rate["time"]),
            open=rate["open"],
            high=rate["high"],
            low=rate["low"],
            close=rate["close"],
            volume=rate["tick_volume"],
        )

    # ---------------------------------------------------
    # Trade lifecycle
    # ---------------------------------------------------

    def _open_trade(self, signal) -> None:
        balance = self._get_balance_for_sizing()
        trade = self.risk_manager.build_trade(signal=signal, balance=balance)

        if trade is None:
            print("⚠ Risk manager did not return a trade")
            return

        if not self.use_tp:
            trade.take_profit = None

        ticket = self.executor.send_trade(trade, use_tp=self.use_tp)
        if ticket is None:
            return

        if hasattr(trade, "set_ticket"):
            trade.set_ticket(ticket)
        else:
            trade.ticket = ticket

        if not self.use_tp:
            trade.take_profit = 0.0

        self.current_trade = trade

        # Mirror the backtest: always sync strategy after a trade opens
        self.strategy.sync_trade(trade)

        print(
            f"🚀 Trade opened: {trade.direction} entry={trade.entry} "
            f"sl={trade.stop_loss} tp={getattr(trade, 'take_profit', None)} "
            f"ticket={ticket}"
        )

    def _trail_stop(self) -> None:
        """
        Trail the open position's SL using the strategy's current PSAR dot.

        Called unconditionally every candle while a trade is open.
        Does NOT depend on a signal being emitted — PSAR moves every candle
        and this must be evaluated every candle.

        The executor's trail_position_with_psar() silently does nothing if:
          - The new SL is not an improvement over the current one.
          - The new SL is too close to market price (broker min distance).
        So repeated calls are always safe.
        """
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
            print(f"🔄 PSAR trail applied: SL -> {psar_sl:.5f}")

    def _check_trade_closed(self) -> bool:
        return not self.executor.has_open_position()

    def _handle_trade_closed(self) -> None:
        """
        Called when the broker confirms the position is gone.
        Notifies the strategy so cooldown and direction state reset correctly,
        mirroring the backtest's on_trade_closed() call.
        """
        print("✅ Trade closed on broker side")
        self._log_trade_close(self.current_trade, close_reason="BrokerClose")

        # We can't know if it was SL or TP from the broker side without
        # inspecting the deal history. Treat as a non-SL close so the
        # strategy doesn't apply the stop cooldown unnecessarily.
        # If you want accurate SL detection, query mt5.history_deals_get()
        # and pass "SL" or "TP" accordingly.
        self.strategy.on_trade_closed(exit_type="BrokerClose")
        self.current_trade = None

    def _log_trade_close(self, trade, close_reason: str = "Unknown") -> None:
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
                    "Symbol",
                    "Direction",
                    "Entry",
                    "StopLoss",
                    "TakeProfit",
                    "PositionSize",
                    "Ticket",
                    "CloseReason",
                ])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self.symbol,
                getattr(trade, "direction", None),
                getattr(trade, "entry", None),
                getattr(trade, "stop_loss", None),
                getattr(trade, "take_profit", None),
                getattr(trade, "position_size", None),
                getattr(trade, "ticket", None),
                close_reason,
            ])

    # ---------------------------------------------------
    # Main loop
    # ---------------------------------------------------

    def run(self) -> None:
        print(
            f"🔥 Live trading started: {self.symbol} {self.timeframe_name} "
            f"use_tp={self.use_tp}"
        )

        while True:
            candle = self._get_latest_closed_candle()
            if candle is None:
                time.sleep(self.poll_seconds)
                continue

            # Only process each closed candle once
            if self.last_candle_time == candle.time:
                time.sleep(self.poll_seconds)
                continue

            self.last_candle_time = candle.time
            # self.series.append_from_mt5_rate(candle)

            # Step 1: update strategy indicators and get any signal
            signal = self.strategy.update(candle)
            signal_name = getattr(signal, "signal", None) if signal is not None else None

            # Step 2: manage open trade
            if self.current_trade is not None:
                if self._check_trade_closed():
                    # Broker closed the position (SL/TP hit or manual close)
                    self._handle_trade_closed()
                else:
                    # Trail unconditionally — not gated on signal emission
                    self._trail_stop()

            # Step 3: open new trade if flat and signal warrants it
            if self.current_trade is None and not self.executor.has_open_position():
                if signal_name in ("BUY", "SELL"):
                    self._open_trade(signal)

            time.sleep(self.poll_seconds)


if __name__ == "__main__":
    from app.core.strategy.parabolic_sar_strategy_v1 import ParabolicSarStrategy
    from app.core.risk.risk_manager import RiskManager

    trader = LiveTrader(
        symbol="BTCUSDm",
        timeframe="1m",
        bars=300,
        strategy_cls=ParabolicSarStrategy,
        strategy_kwargs={
            "use_close_signal": False,
            "psar_step": 0.025,
            "psar_max_step": 0.05,
            "atr_period": 14,
            "use_ema_trend": True,
            "ema_trend_period": 50,
            "ema_offset": 3,
            "ema_slope_threshold": 0.0,
            "use_adx": False,
        },
        risk_manager=RiskManager(
            risk_pct=0.01,
            rr=2.0,
            atr_multiplier=1.5,
            sl_mode="WIDER",
        ),
        use_tp=True,
    )
    trader.run()