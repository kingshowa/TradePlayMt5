import csv
import json
import math
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.risk.risk_manager import RiskManager
from app.core.strategy.parabolic_sar_strategy import ParabolicSarStrategy


SYMBOLS = {
    "Gold": "XAUUSDm",
    "Silver": "XAGUSDm",
    "Platinum": "XPTUSDm",
    "Oil": "BNO",
    "Euro": "EURUSDm",
    "EuroJpy": "EURJPYm",
    "BTC": "BTC",
    "ETH": "ETH",
}


class TradeSimulation:
    """
    Candle-level SL / TP exit checker.

    Conservative rule:
    If SL and TP are touched in the same candle, SL is assumed first.
    """

    @staticmethod
    def check_exit(trade, candle):
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        if direction == "BUY":
            if sl is not None and candle.low <= sl:
                return sl, "SL", (sl - entry) * size
            if tp is not None and candle.high >= tp:
                return tp, "TP", (tp - entry) * size

        elif direction == "SELL":
            if sl is not None and candle.high >= sl:
                return sl, "SL", (entry - sl) * size
            if tp is not None and candle.low <= tp:
                return tp, "TP", (entry - tp) * size

        return None


_CSV_HEADER = [
    "EntryTime", "Direction",
    "Entry", "SL", "TP", "PositionSize",
    "ExitPrice", "ExitType",
    "PnL", "BalanceAfter",
    "Pattern", "Reason",
]


_SUMMARY_HEADER = [
    "RunTimestamp", "Comment", "Symbol", "Timeframe", "StartDate", "EndDate",
    "Candles", "CutOff", "InitialBalance", "FinalBalance", "UseCloseSignal",
    "TotalTrades", "Wins", "Losses", "Breakeven", "WinRate", "ProfitFactor",
    "AvgWin", "AvgLoss", "BestTrade", "WorstTrade", "TotalPnL",
    "MaxConsecWins", "MaxConsecLoss", "MaxDrawdown", "MaxDrawdownPct",
    "TotalReturnPct", "SharpeRatio",
    "ExitSL", "ExitTP", "ExitStrategyClose", "ExitForceClose",
    "BuyTrades", "SellTrades",
    "StrategyParams", "TradeLogFile",
]


def _write_csv_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)


def _append_trade_row(path: str, trade, exit_price, exit_type, pnl, balance_after) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            trade.candle.time,
            trade.direction,
            f"{trade.entry:.5f}",
            f"{trade.stop_loss:.5f}" if trade.stop_loss is not None else "",
            f"{trade.take_profit:.5f}" if trade.take_profit is not None else "",
            f"{trade.position_size:.4f}",
            f"{exit_price:.5f}",
            exit_type,
            f"{pnl:.4f}",
            f"{balance_after:.4f}",
            trade.pattern_name,
            trade.reason,
        ])


def _compute_stats(df: pd.DataFrame, initial_balance: float) -> dict:
    pnl = df["PnL"]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_cw = max_cl = cur_cw = cur_cl = 0
    for val in pnl:
        if val > 0:
            cur_cw += 1
            cur_cl = 0
        elif val < 0:
            cur_cl += 1
            cur_cw = 0

        max_cw = max(max_cw, cur_cw)
        max_cl = max(max_cl, cur_cl)

    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    max_dd = drawdown.max()

    peak_at_max_dd = running_peak[drawdown.idxmax()] if not drawdown.empty else 0
    max_dd_pct = (
        max_dd / (initial_balance + peak_at_max_dd) * 100
        if initial_balance + peak_at_max_dd > 0
        else 0
    )

    std_pnl = pnl.std()
    sharpe = pnl.mean() / std_pnl * math.sqrt(len(pnl)) if std_pnl and std_pnl > 0 else 0.0

    return {
        "total_trades": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(pnl[pnl == 0]),
        "win_rate": len(wins) / len(df) if len(df) else 0,
        "profit_factor": profit_factor,
        "avg_win": wins.mean() if not wins.empty else 0.0,
        "avg_loss": losses.mean() if not losses.empty else 0.0,
        "best_trade": pnl.max(),
        "worst_trade": pnl.min(),
        "total_pnl": pnl.sum(),
        "max_consec_wins": max_cw,
        "max_consec_loss": max_cl,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_return_pct": pnl.sum() / initial_balance * 100 if initial_balance else 0,
        "sharpe_ratio": sharpe,
        "exits_by_type": df["ExitType"].value_counts().to_dict(),
        "exits_by_dir": df["Direction"].value_counts().to_dict(),
    }


def _print_stats(stats, symbol, timeframe, initial_balance, final_balance, start_date, end_date):
    sep = "─" * 52

    print(f"\n{sep}")
    print(f"  Parabolic SAR Backtest — {symbol} {timeframe.upper()}")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(sep)
    print(f"  {'Initial balance':<28} {initial_balance:>10.2f}")
    print(f"  {'Final balance':<28} {final_balance:>10.2f}")
    print(f"  {'Net PnL':<28} {stats['total_pnl']:>+10.2f}")
    print(f"  {'Return':<28} {stats['total_return_pct']:>+9.2f}%")
    print(sep)
    print(f"  {'Total trades':<28} {stats['total_trades']:>10}")
    print(f"  {'Wins / Losses / BE':<28} {stats['wins']:>4} / {stats['losses']:>4} / {stats['breakeven']:>4}")
    print(f"  {'Win rate':<28} {stats['win_rate'] * 100:>9.2f}%")
    print(f"  {'Profit factor':<28} {stats['profit_factor']:>10.2f}")
    print(sep)
    print(f"  {'Avg win':<28} {stats['avg_win']:>+10.2f}")
    print(f"  {'Avg loss':<28} {stats['avg_loss']:>+10.2f}")
    print(f"  {'Best trade':<28} {stats['best_trade']:>+10.2f}")
    print(f"  {'Worst trade':<28} {stats['worst_trade']:>+10.2f}")
    print(sep)
    print(f"  {'Max drawdown':<28} {stats['max_drawdown']:>+10.2f}")
    print(f"  {'Max drawdown %':<28} {stats['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Sharpe ratio':<28} {stats['sharpe_ratio']:>10.3f}")
    print(sep)
    print("  Exit breakdown:")
    for exit_type, count in sorted(stats["exits_by_type"].items()):
        print(f"    {exit_type:<26} {count:>10}")
    print("  Direction breakdown:")
    for direction, count in sorted(stats["exits_by_dir"].items()):
        print(f"    {direction:<26} {count:>10}")
    print(sep)


def _plot_results(df, stats, symbol, timeframe, start_date, end_date):
    pnl = df["PnL"]
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Parabolic SAR Backtest — {symbol} {timeframe.upper()}   "
        f"{start_date.date()} → {end_date.date()}   "
        f"Win rate: {stats['win_rate'] * 100:.1f}%   "
        f"PF: {stats['profit_factor']:.2f}   "
        f"Return: {stats['total_return_pct']:+.1f}%",
        fontsize=12,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1.5], hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative.values, linewidth=1.4, label="Cumulative PnL")
    ax1.axhline(0, linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_xlabel("Trade #")
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(range(len(drawdown)), -drawdown.values, 0, alpha=0.35, label="Drawdown")
    ax2.plot(-drawdown.values, linewidth=0.8)
    ax2.axhline(0, linewidth=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Trade #")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    ax3 = fig.add_subplot(gs[2])
    ax3.hist(pnl[pnl > 0].values, bins=30, alpha=0.65, label="Wins")
    ax3.hist(pnl[pnl <= 0].values, bins=30, alpha=0.65, label="Losses")
    ax3.axvline(0, linewidth=0.8, linestyle="--")
    ax3.set_xlabel("PnL per trade")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)
    ax3.grid(True, linewidth=0.4, alpha=0.5)

    plt.show()


def _append_summary_row(
    path,
    comment_name,
    stats,
    strategy_params,
    symbol,
    timeframe,
    start_date,
    end_date,
    candles_count,
    cut_off,
    initial_balance,
    final_balance,
    use_close_signal,
    trade_log_file,
):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    exits = stats.get("exits_by_type", {})
    dirs = stats.get("exits_by_dir", {})

    row = {
        "RunTimestamp": datetime.now().isoformat(timespec="seconds"),
        "Comment": comment_name,
        "Symbol": symbol,
        "Timeframe": timeframe,
        "StartDate": start_date.date().isoformat(),
        "EndDate": end_date.date().isoformat(),
        "Candles": candles_count,
        "CutOff": cut_off,
        "InitialBalance": f"{initial_balance:.2f}",
        "FinalBalance": f"{final_balance:.4f}",
        "UseCloseSignal": use_close_signal,
        "TotalTrades": stats["total_trades"],
        "Wins": stats["wins"],
        "Losses": stats["losses"],
        "Breakeven": stats["breakeven"],
        "WinRate": f"{stats['win_rate']:.6f}",
        "ProfitFactor": f"{stats['profit_factor']:.6f}",
        "AvgWin": f"{stats['avg_win']:.6f}",
        "AvgLoss": f"{stats['avg_loss']:.6f}",
        "BestTrade": f"{stats['best_trade']:.6f}",
        "WorstTrade": f"{stats['worst_trade']:.6f}",
        "TotalPnL": f"{stats['total_pnl']:.6f}",
        "MaxConsecWins": stats["max_consec_wins"],
        "MaxConsecLoss": stats["max_consec_loss"],
        "MaxDrawdown": f"{stats['max_drawdown']:.6f}",
        "MaxDrawdownPct": f"{stats['max_drawdown_pct']:.6f}",
        "TotalReturnPct": f"{stats['total_return_pct']:.6f}",
        "SharpeRatio": f"{stats['sharpe_ratio']:.6f}",
        "ExitSL": exits.get("SL", 0),
        "ExitTP": exits.get("TP", 0),
        "ExitStrategyClose": exits.get("StrategyClose", 0),
        "ExitForceClose": exits.get("ForceClose", 0),
        "BuyTrades": dirs.get("BUY", 0),
        "SellTrades": dirs.get("SELL", 0),
        "StrategyParams": json.dumps(strategy_params, sort_keys=True),
        "TradeLogFile": trade_log_file,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def backtest_parabolic_sar(
    symbol: str = SYMBOLS["Gold"],
    timeframe: str = "5m",
    start_date: datetime = datetime(2024, 1, 1),
    end_date: datetime = datetime(2026, 12, 30),
    cut_off: int = 300,
    initial_balance: float = 200.0,
    use_close_signal: bool = True,
    output_file: str = "trades_parabolic_sar.csv",
    strategy_params: Optional[dict] = None,
    comment_name: str = "manual_run",
    summary_file: str = "parabolic_sar_experiment_results.csv",
    plot: bool = True,
) -> Optional[dict]:

    print("\n  Parabolic SAR Backtest")
    print(f"  Symbol: {symbol}  |  Timeframe: {timeframe}")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(f"  Balance: {initial_balance:.2f}  |  Close signal: {use_close_signal}")
    print(f"  Warming up on first {cut_off} candles…")
    print(f"  Comment: {comment_name}")

    strategy_params = dict(strategy_params or {})
    print(f"  Strategy params: {strategy_params if strategy_params else 'DEFAULTS'}")

    provider = MT5MarketDataProvider()
    series = provider.fetch_range(
        symbol,
        TIMEFRAMES[timeframe],
        start_date,
        end_date,
    )

    total_candles = len(series._candles)
    print(f"  Loaded {total_candles:,} candles ({total_candles - cut_off:,} tradeable after warm-up)")

    if total_candles <= cut_off:
        print("  ERROR: not enough candles for the selected cut_off.")
        return None

    warmup_series = series.subseries(0, cut_off)

    strategy = ParabolicSarStrategy(
        warmup_series,
        use_close_signal=use_close_signal,
        **strategy_params,
    )
    risk_engine = RiskManager()

    balance = initial_balance
    active_trade = None
    active_trade_entry_idx = None

    _write_csv_header(output_file)

    for i in range(cut_off, total_candles):
        candle = series._candles[i]

        signal = strategy.update(candle)

        if active_trade is not None and i > active_trade_entry_idx:
            exit_result = TradeSimulation.check_exit(active_trade, candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl

                strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                _append_trade_row(output_file, active_trade, exit_price, exit_type, pnl, balance)

                active_trade = None
                active_trade_entry_idx = None
                continue

            if use_close_signal and signal is not None and signal.signal == "CLOSE":
                exit_price = candle.close
                direction = active_trade.direction.upper()

                if direction == "BUY":
                    pnl = (exit_price - active_trade.entry) * active_trade.position_size
                else:
                    pnl = (active_trade.entry - exit_price) * active_trade.position_size

                balance += pnl

                strategy.on_trade_closed(exit_type="StrategyClose", exit_price=exit_price)
                _append_trade_row(output_file, active_trade, exit_price, "StrategyClose", pnl, balance)

                active_trade = None
                active_trade_entry_idx = None
                continue

            continue

        if (
            active_trade is None
            and signal is not None
            and signal.signal in ("BUY", "SELL")
        ):
            active_trade = risk_engine.build_trade(signal, balance)
            active_trade_entry_idx = i
            strategy.sync_trade(active_trade)

    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close
        direction = active_trade.direction.upper()

        if direction == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size

        balance += pnl
        strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)
        _append_trade_row(output_file, active_trade, exit_price, "ForceClose", pnl, balance)

    print(f"\n  Backtest complete. Reading results from {output_file}…")

    df = pd.read_csv(output_file)

    if df.empty:
        print("  No trades were executed. Check strategy filters and parameters.")
        return None

    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)

    stats = _compute_stats(df, initial_balance)
    _print_stats(stats, symbol, timeframe, initial_balance, balance, start_date, end_date)

    _append_summary_row(
        path=summary_file,
        comment_name=comment_name,
        stats=stats,
        strategy_params=strategy_params,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        candles_count=total_candles,
        cut_off=cut_off,
        initial_balance=initial_balance,
        final_balance=balance,
        use_close_signal=use_close_signal,
        trade_log_file=output_file,
    )

    print(f"  Summary appended to {summary_file}")

    if plot:
        _plot_results(df, stats, symbol, timeframe, start_date, end_date)

    return {
        "comment_name": comment_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "candles_count": total_candles,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "use_close_signal": use_close_signal,
        "strategy_params": strategy_params,
        **stats,
    }


if __name__ == "__main__":
    strategy_params = {
        "psar_step": 0.015,
        "psar_max_step": 0.15,

        "use_ema_trend": True,
        "ema_trend_period": 200,
        "ema_offset": 3,
        "ema_slope_threshold": 0.0,

        "use_adx": True,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "require_adx_bias": False,

        "close_signal_adx_limit": 20.0,

        "atr_period": 14,
    }

    backtest_parabolic_sar(
        symbol=SYMBOLS["Gold"],
        timeframe="5m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 12, 30),
        cut_off=300,
        initial_balance=200.0,
        use_close_signal=False,
        output_file="trades_parabolic_sar.csv",
        strategy_params=strategy_params,
        comment_name="psar_first_test",
        summary_file="parabolic_sar_experiment_results.csv",
        plot=True,
    )