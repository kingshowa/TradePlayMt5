from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES

from app.core.risk.risk_manager import RiskManager
import csv
import pandas as pd
import matplotlib.pyplot as plt

from app.core.strategy.ema_rsi_strategy import EmaRsiChannelStrategy


class TradeSimulation:
    @staticmethod
    def check_exit(trade, candle):
        """
        Check whether an already open trade exits on the current candle.

        Returns:
            (exit_price, exit_type, pnl) if closed
            None if still open
        """
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        if direction == "BUY":
            # Conservative assumption:
            # if both SL and TP are touched in same candle, SL is assumed first
            if candle.low <= sl:
                pnl = (sl - entry) * size
                return sl, "SL", pnl

            if candle.high >= tp:
                pnl = (tp - entry) * size
                return tp, "TP", pnl

        else:  # SELL
            if candle.high >= sl:
                pnl = (entry - sl) * size
                return sl, "SL", pnl

            if candle.low <= tp:
                pnl = (entry - tp) * size
                return tp, "TP", pnl

        return None


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
    timeframe="1m",
    bars=100000,
    cut_off=300,
    blc=200
):
    print(f"\n🚀 EMA-RSI Backtest Started | {symbol}")

    provider = MT5MarketDataProvider()
    series = provider.fetch(symbol, TIMEFRAMES[timeframe], bars)

    strategy = EmaRsiChannelStrategy(series.subseries(0, cut_off))
    risk_engine = RiskManager()

    trades_count = 0
    balance = blc
    active_trade = None
    active_trade_entry_index = None

    output_file = "trades_outcome_rsi_non_ov.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Direction", "Entry", "SL", "TP",
            "PositionSize", "ExitPrice", "Pattern", "Reason", "ExitType", "PnL"
        ])

    # =========================
    # Main Loop
    # =========================
    for i in range(cut_off, len(series._candles)):
        new_candle = series._candles[i]

        # 1. Always update indicators/strategy state
        signal = strategy.update(new_candle)

        # 2. If trade is already open, check exit on this candle
        #    Do not check exit on the exact same candle where the trade was opened
        if active_trade is not None and i > active_trade_entry_index:
            exit_result = TradeSimulation.check_exit(active_trade, new_candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl

                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        active_trade.candle.time,
                        active_trade.direction,
                        f"{active_trade.entry:.2f}",
                        f"{active_trade.stop_loss:.2f}",
                        f"{active_trade.take_profit:.2f}",
                        f"{active_trade.position_size:.2f}",
                        f"{exit_price:.2f}",
                        active_trade.pattern_name,
                        active_trade.reason,
                        exit_type,
                        f"{pnl:.2f}"
                    ])

                active_trade = None
                active_trade_entry_index = None

            # While a trade is open, do not open another one
            continue

        # 3. If no active trade, allow signal to open one
        if active_trade is None and signal is not None:
            active_trade = risk_engine.build_trade(signal, balance)
            active_trade_entry_index = i
            trades_count += 1

    # =========================
    # Force-close last open trade at end of backtest
    # =========================
    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close

        if active_trade.direction.upper() == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size

        balance += pnl

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                active_trade.candle.time,
                active_trade.direction,
                f"{active_trade.entry:.2f}",
                f"{active_trade.stop_loss:.2f}",
                f"{active_trade.take_profit:.2f}",
                f"{active_trade.position_size:.2f}",
                f"{exit_price:.2f}",
                active_trade.pattern_name,
                active_trade.reason,
                "Close",
                f"{pnl:.2f}"
            ])

    print(f"✅ EMA-RSI backtest complete. {trades_count} trades executed.")
    print(f"Balance: {balance:.2f}")

    # =========================
    # Results & Equity Curve
    # =========================
    df = pd.read_csv(output_file)

    if df.empty:
        print("No trades were executed.")
        return

    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce")
    df["CumulativePnL"] = df["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["CumulativePnL"], label="EMA-RSI Equity Curve")
    plt.title(f"EMA-RSI Backtest Equity Curve - {symbol}")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.legend()
    plt.show()

    win_rate = (df["PnL"] > 0).mean()
    avg_win = df[df["PnL"] > 0]["PnL"].mean()
    avg_loss = df[df["PnL"] < 0]["PnL"].mean()

    print(f"Win rate: {win_rate * 100:.2f}%")
    print(f"Avg win: {avg_win:.2f}" if pd.notna(avg_win) else "Avg win: 0.00")
    print(f"Avg loss: {avg_loss:.2f}" if pd.notna(avg_loss) else "Avg loss: 0.00")


# =========================
# Run
# =========================
if __name__ == "__main__":
    backtest_real_data_managed()