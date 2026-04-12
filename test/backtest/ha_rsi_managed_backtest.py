from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES

from app.core.risk.risk_manager import RiskManager
from app.core.strategy.base_strategy import BaseStrategy
from app.core.indicators.atr import ATRIndicator
import csv
import pandas as pd
import matplotlib.pyplot as plt

from app.core.strategy.ema_rsi_strategy import EmaRsiChannelStrategy
from app.core.strategy.ema_strategy import EmaStrategy
from app.core.strategy.ha_rsi_strategy import HeikinAshiRsiStrategy


class TradeSimulation:
    @staticmethod
    def simulate(trade, series_after_entry):
        """
        Simulate the trade until it hits SL or TP using subsequent candles.
        Returns exit price, exit type ('SL', 'TP', 'Close'), and PnL.
        """
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        for candle in series_after_entry._candles:
            if direction == "BUY":
                if candle.low <= sl:
                    pnl = (sl - entry) * size
                    return sl, "SL", pnl
                elif candle.high >= tp:
                    pnl = (tp - entry) * size
                    return tp, "TP", pnl
            else:  # SELL
                if candle.high >= sl:
                    pnl = (entry - sl) * size
                    return sl, "SL", pnl
                elif candle.low <= tp:
                    pnl = (entry - tp) * size
                    return tp, "TP", pnl

        # If neither SL nor TP hit, exit at last candle close
        exit_price = series_after_entry.last().close
        pnl = (exit_price - entry) * size if direction == "BUY" else (entry - exit_price) * size
        return exit_price, "Close", pnl



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
    "BTC": "BTCUSDm",
    "ETH": "ETHUSDm",
}


# =========================
# Backtest
# =========================

# ... (Keep Imports and TradeSimulation class as they are)

# =========================
# Backtest
# =========================
def backtest_real_data_managed(
        symbol=comodities["Gold"],
        timeframe="5m",
        bars=10000,
        cut_off=200,
        blc=200,
        use_strategy_close=True  # New optional toggle
):
    print(f"\n🚀 HA-RSI Backtest Started | {symbol} | Use Strategy Close: {use_strategy_close}")

    provider = MT5MarketDataProvider()
    series = provider.fetch(symbol, TIMEFRAMES[timeframe], bars)
    strategy = HeikinAshiRsiStrategy(series.subseries(0, cut_off))
    risk_engine = RiskManager()

    trades_count = 0
    balance = blc
    active_trade = None  # Tracks if we are currently in a position

    # CSV output
    with open("trades_outcome_ha_rsi_m.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Direction", "Entry", "SL", "TP",
            "PositionSize", "ExitPrice", "ExitType", "PnL", "Pattern", "Reason"
        ])

    # =========================
    # Main Loop (Candle-by-Candle)
    # =========================
    for i in range(cut_off, len(series.candles())):
        new_candle = series.candles()[i]
        signal = strategy.update(new_candle)

        # --- 1. HANDLE ACTIVE TRADE ---
        if active_trade:
            exit_price = None
            exit_type = None

            # Check SL/TP First (Standard Exit)
            if active_trade.direction == "BUY":
                if new_candle.low <= active_trade.stop_loss:
                    exit_price, exit_type = active_trade.stop_loss, "SL"
                # elif new_candle.high >= active_trade.take_profit:
                #     exit_price, exit_type = active_trade.take_profit, "TP"
            else:  # SELL
                if new_candle.high >= active_trade.stop_loss:
                    exit_price, exit_type = active_trade.stop_loss, "SL"
                # elif new_candle.low <= active_trade.take_profit:
                #     exit_price, exit_type = active_trade.take_profit, "TP"

            # Check for Strategy "CLOSE" signal (Optional Exit)
            if exit_type is None and use_strategy_close and signal and signal.signal == "CLOSE":
                if active_trade.direction == "BUY":
                    exit_price, exit_type = new_candle.close, "TP" if new_candle.close >= active_trade.take_profit else "Strategy_Close"
                else:
                    exit_price, exit_type = new_candle.close, "TP" if new_candle.close <= active_trade.take_profit else "Strategy_Close"
                print(f"Close Signal: {signal.reason} at {exit_price}")

            # If an exit condition was met, record it
            if exit_type:
                pnl = (exit_price - active_trade.entry) * active_trade.position_size if active_trade.direction == "BUY" else (
                                                                                                                                                    active_trade.entry - exit_price) * active_trade.position_size
                balance += pnl

                # Write to CSV
                with open("trades_outcome_ha_rsi_m.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        active_trade.candle.time, active_trade.direction, f"{active_trade.entry:.2f}",
                        f"{active_trade.stop_loss:.2f}", f"{active_trade.take_profit:.2f}",
                        f"{active_trade.position_size:.2f}", f"{exit_price:.2f}", exit_type,
                        f"{pnl:.2f}", active_trade.pattern_name, active_trade.reason
                    ])

                active_trade = None  # Reset for next trade

            # Continue to next candle (don't open new trade while one is active)
            continue

        # --- 2. HANDLE NEW ENTRIES ---
        if signal and signal.signal in ("BUY", "SELL"):
            print(f"🟢 Open {signal.signal} at {new_candle.close} SL: {signal.sl:.2f}")
            active_trade = risk_engine.build_trade(signal, balance)
            trades_count += 1

    print(f"✅ HA-RSI backtest complete. {trades_count} trades executed.")
    print(f"Final Balance: {balance:.2f}")

    # ... (Rest of the results & plotting code)


    # =========================
    # Results & Equity Curve
    # =========================
    df = pd.read_csv("trades_outcome_ha_rsi_m.csv")
    df["CumulativePnL"] = df["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["CumulativePnL"], label="HA-RSI Equity Curve")
    plt.title(f"HA-RSI Backtest Equity Curve - {symbol}")
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