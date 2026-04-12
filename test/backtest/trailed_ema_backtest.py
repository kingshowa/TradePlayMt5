from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES

from app.core.risk.risk_manager import RiskManager
from app.core.strategy.base_strategy import BaseStrategy
from app.core.indicators.atr import ATRIndicator
import csv
import pandas as pd
import matplotlib.pyplot as plt

from app.core.strategy.ema_strategy import EmaStrategy


# =========================
# Trade Management Engine
# =========================
# class ManagedTradeSimulation:
#     @staticmethod
#     def simulate(trade, series_after_entry, atr, max_bars=30, trail_mult=1.0):
#         """
#         Managed trade simulation:
#         - ATR trailing stop
#         - Time-based exit
#         """
#         direction = trade.direction.upper()
#         entry = trade.entry
#         sl = trade.stop_loss
#         tp = trade.take_profit
#         size = trade.position_size
#
#         best_price = entry
#         bars_held = 0
#
#         for candle in series_after_entry._candles:
#             bars_held += 1
#
#             # ---- Update trailing stop
#             if direction == "BUY":
#                 best_price = max(best_price, candle.high)
#                 trailing_sl = best_price - atr * trail_mult
#                 sl = max(sl, trailing_sl)
#
#                 if candle.low <= sl:
#                     pnl = (sl - entry) * size
#                     return sl, "TRAIL_SL", pnl
#
#                 if candle.high >= tp:
#                     pnl = (tp - entry) * size
#                     return tp, "TP", pnl
#
#             else: # SELL
#                 best_price = min(best_price, candle.low)
#                 trailing_sl = best_price + atr * trail_mult
#                 sl = min(sl, trailing_sl)
#
#                 if candle.high >= sl:
#                     pnl = (entry - sl) * size
#                     return sl, "TRAIL_SL", pnl
#
#                 if candle.low <= tp:
#                     pnl = (entry - tp) * size
#                     return tp, "TP", pnl
#
#             # ---- Time stop
#             if bars_held >= max_bars:
#                 exit_price = candle.close
#                 pnl = (
#                     (exit_price - entry) * size
#                     if direction == "BUY"
#                     else (entry - exit_price) * size
#                 )
#                 return exit_price, "TIME_EXIT", pnl

class ManagedTradeSimulation:
    @staticmethod
    def simulate(trade, series_after_entry, atr, max_bars=30, trail_mult=2.0, be_trigger=2.8):
        """
        Optimized Simulation:
        - Breakeven Trigger at 1.5 * ATR
        - Closing-basis Trailing (Prevents wick-outs)
        - Wider trail_mult (Recommended 2.0+ for ATR)
        """
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        at_breakeven = False
        bars_held = 0

        for candle in series_after_entry._candles:
            bars_held += 1
            curr_close = candle.close

            if direction == "BUY":
                # 1. Breakeven logic: Move SL to entry + tiny buffer once target hit
                if not at_breakeven and (candle.high - entry) >= (atr * be_trigger):
                    sl = entry + (atr * 0.1)
                    at_breakeven = True

                # 2. Optimized Trailing: Trail based on CLOSE, not HIGH
                potential_sl = curr_close - (atr * trail_mult)
                sl = max(sl, potential_sl)

                # 3. Check Exits
                if candle.low <= sl:
                    return sl, "TRAIL_SL", (sl - entry) * size
                if candle.high >= tp:
                    return tp, "TP", (tp - entry) * size

            else: # SELL
                if not at_breakeven and (entry - candle.low) >= (atr * be_trigger):
                    sl = entry - (atr * 0.1)
                    at_breakeven = True

                potential_sl = curr_close + (atr * trail_mult)
                sl = min(sl, potential_sl)

                if candle.high >= sl:
                    return sl, "TRAIL_SL", (entry - sl) * size
                if candle.low <= tp:
                    return tp, "TP", (entry - tp) * size

        # ---- Force exit
        exit_price = series_after_entry.last().close
        pnl = (
            (exit_price - entry) * size
            if direction == "BUY"
            else (entry - exit_price) * size
        )
        return exit_price, "FORCED_EXIT", pnl


# =========================
# Symbols
# =========================
comodities = {
    "Gold": "XAUUSDm",
    "Silver": "XAGUSDm",
    "Platinum": "XPTUSDm",
    "Oil": "BNO",
    "Euro": "EURUSDm",
    "EuroJpy": "EURJPYm",
    "BTC": "BTC",
    "ETH": "ETH",
}


# =========================
# Backtest
# =========================
def backtest_real_data_managed(
    symbol=comodities["Gold"],
    timeframe="5m",
    bars=500,
    cut_off=100,
    blc=200
):
    print(f"\n🚀 Trailed EMA Backtest Started | {symbol}")

    provider = MT5MarketDataProvider()
    series = provider.fetch(symbol, TIMEFRAMES[timeframe], bars)
    strategy = EmaStrategy(series.subseries(0,cut_off))
    risk_engine = RiskManager()

    trades_count = 0
    balance = blc

    # CSV output
    with open("trades_outcome_trailed_ema.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Direction", "Entry", "SL", "TP",
            "PositionSize", "ExitPrice", "Pattern", "Reason", "ExitType", "PnL"
        ])

    # =========================
    # Main Loop
    # =========================
    for i in range(cut_off, len(series._candles) - 1):
        new_candle = series._candles[i]

        signal = strategy.update(new_candle)
        if not signal:
            continue


        trade = risk_engine.build_trade(signal, balance)
        trades_count += 1

        # Simulate managed exit
        series_after_entry = series.subseries(i + 1, len(series._candles))

        exit_price, exit_type, pnl = ManagedTradeSimulation.simulate(
                trade=trade,
                series_after_entry=series_after_entry,
                atr=signal.atr,
                max_bars=10,
                trail_mult=2.0
            )

        balance = balance + pnl

        with open("trades_outcome_trailed_ema.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.candle.time,
                trade.direction,
                trade.entry,
                trade.stop_loss,
                trade.take_profit,
                trade.position_size,
                exit_price,
                trade.pattern_name,
                trade.reason,
                exit_type,
                pnl
            ])

    print(f"✅ Trailed ema backtest complete. {trades_count} trades executed.")

    # =========================
    # Results & Equity Curve
    # =========================
    df = pd.read_csv("trades_outcome_trailed_ema.csv")
    df["CumulativePnL"] = df["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["CumulativePnL"], label="Trailed EMA Equity Curve")
    plt.title(f"Trailed EMA Backtest Equity Curve - {symbol}")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.legend()
    plt.show()

    win_rate = (df["PnL"] > 0).mean()
    avg_win = df[df["PnL"] > 0]["PnL"].mean()
    avg_loss = df[df["PnL"] < 0]["PnL"].mean()

    print(f"Win rate: {win_rate * 100:.2f}%")
    print(f"Avg win: {avg_win:.2f}")
    print(f"Avg loss: {avg_loss:.2f}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    backtest_real_data_managed()