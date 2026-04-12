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
    "BTC": "BTC",
    "ETH": "ETH",
}


# =========================
# Backtest
# =========================
def backtest_real_data_managed(
    symbol=comodities["Silver"],
    timeframe="1m",
    bars=10000,
    cut_off=300,
    blc=200
):
    print(f"\n🚀 EMA Backtest Started | {symbol}")

    provider = MT5MarketDataProvider()
    series = provider.fetch(symbol, TIMEFRAMES[timeframe], bars)
    strategy = EmaStrategy(series.subseries(0,cut_off))
    risk_engine = RiskManager()

    trades_count = 0
    balance = blc

    # CSV output
    with open("trades_outcome_ema.csv", "w", newline="") as f:
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

        exit_price, exit_type, pnl = TradeSimulation.simulate(
                trade=trade,
                series_after_entry=series_after_entry
            )

        balance = balance + pnl

        with open("trades_outcome_ema.csv", "a", newline="") as f:
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

    print(f"✅ EMA backtest complete. {trades_count} trades executed.")
    print(f"Balance: {balance}")

    # =========================
    # Results & Equity Curve
    # =========================
    df = pd.read_csv("trades_outcome_ema.csv")
    df["CumulativePnL"] = df["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["CumulativePnL"], label="EMA Equity Curve")
    plt.title(f"EMA Backtest Equity Curve - {symbol}")
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