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
from app.core.strategy.golden_ribon_strategy import GoldenRibbonStrategy


# =============================================================================
# Symbols
# =============================================================================

SYMBOLS = {
    "Gold":     "XAUUSDm",
    "Silver":   "XAGUSDm",
    "Platinum": "XPTUSDm",
    "Oil":      "BNO",
    "Euro":     "EURUSDm",
    "EuroJpy":  "EURJPYm",
    "BTC":      "BTC",
    "ETH":      "ETH",
}


# =============================================================================
# Exit simulation
# =============================================================================

class TradeSimulation:
    """
    Candle-level SL / TP exit checker.

    Conservative rule: if both SL and TP are touched in the same candle,
    SL is assumed to have been hit first. This avoids over-optimistic
    backtest results caused by wide-candle ambiguity.
    """

    @staticmethod
    def check_exit(trade, candle):
        """
        Evaluate whether an open trade closes on this candle.

        Returns:
            (exit_price, exit_type, pnl)  when the trade is closed.
            None                           when the trade remains open.
        """
        direction = trade.direction.upper()
        entry     = trade.entry
        sl        = trade.stop_loss
        tp        = trade.take_profit
        size      = trade.position_size

        if direction == "BUY":
            if candle.low <= sl:                         # SL hit (intrabar)
                return sl, "SL", (sl - entry) * size
            if candle.high >= tp:                        # TP hit (intrabar)
                return tp, "TP", (tp - entry) * size

        else:  # SELL
            if candle.high >= sl:                        # SL hit (intrabar)
                return sl, "SL", (entry - sl) * size
            if candle.low <= tp:                         # TP hit (intrabar)
                return tp, "TP", (entry - tp) * size

        return None


# =============================================================================
# Statistics
# =============================================================================

def _compute_stats(df: pd.DataFrame, initial_balance: float) -> dict:
    """
    Derive all performance metrics from the completed trade DataFrame.

    Metrics
    -------
    win_rate         Fraction of trades with PnL > 0.
    profit_factor    Gross profit / gross loss (∞ if no losses).
    avg_win          Mean PnL of winning trades.
    avg_loss         Mean PnL of losing trades (negative).
    best_trade       Highest single-trade PnL.
    worst_trade      Lowest single-trade PnL.
    max_consec_wins  Longest run of consecutive winning trades.
    max_consec_loss  Longest run of consecutive losing trades.
    max_drawdown     Peak-to-trough drawdown on the cumulative equity curve.
    max_drawdown_pct max_drawdown as a percentage of the peak balance.
    total_return_pct Net PnL as a percentage of initial balance.
    sharpe_ratio     Mean trade PnL / std trade PnL (trade-count normalised).
    exits_by_type    Count of trades broken down by exit type.
    exits_by_dir     Count of trades broken down by direction (BUY/SELL).
    """
    pnl = df["PnL"]

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max consecutive wins / losses
    max_cw = max_cl = cur_cw = cur_cl = 0
    for val in pnl:
        if val > 0:
            cur_cw += 1
            cur_cl  = 0
        elif val < 0:
            cur_cl += 1
            cur_cw  = 0
        max_cw = max(max_cw, cur_cw)
        max_cl = max(max_cl, cur_cl)

    # Max drawdown on cumulative equity curve
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown_series = running_peak - cumulative
    max_dd = drawdown_series.max()
    peak_at_max_dd = running_peak[drawdown_series.idxmax()] if not drawdown_series.empty else 0
    max_dd_pct = (max_dd / (initial_balance + peak_at_max_dd) * 100) if (initial_balance + peak_at_max_dd) > 0 else 0

    # Sharpe (simplified: per-trade, not annualised — no risk-free rate)
    std_pnl = pnl.std()
    sharpe = (pnl.mean() / std_pnl * math.sqrt(len(pnl))) if std_pnl and std_pnl > 0 else 0.0

    return {
        "total_trades":     len(df),
        "wins":             len(wins),
        "losses":           len(losses),
        "breakeven":        len(pnl[pnl == 0]),
        "win_rate":         len(wins) / len(df) if len(df) else 0,
        "profit_factor":    profit_factor,
        "avg_win":          wins.mean()   if not wins.empty   else 0.0,
        "avg_loss":         losses.mean() if not losses.empty else 0.0,
        "best_trade":       pnl.max(),
        "worst_trade":      pnl.min(),
        "total_pnl":        pnl.sum(),
        "max_consec_wins":  max_cw,
        "max_consec_loss":  max_cl,
        "max_drawdown":     max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_return_pct": (pnl.sum() / initial_balance * 100) if initial_balance else 0,
        "sharpe_ratio":     sharpe,
        "exits_by_type":    df["ExitType"].value_counts().to_dict(),
        "exits_by_dir":     df["Direction"].value_counts().to_dict(),
    }


def _print_stats(
    stats: dict,
    symbol: str,
    timeframe: str,
    initial_balance: float,
    final_balance: float,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Print a formatted performance summary to stdout."""
    sep = "─" * 52

    print(f"\n{sep}")
    print(f"  Golden Ribbon Backtest — {symbol} {timeframe.upper()}")
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
    print(f"  {'Max consec. wins':<28} {stats['max_consec_wins']:>10}")
    print(f"  {'Max consec. losses':<28} {stats['max_consec_loss']:>10}")
    print(f"  {'Max drawdown':<28} {stats['max_drawdown']:>+10.2f}")
    print(f"  {'Max drawdown %':<28} {stats['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Sharpe ratio':<28} {stats['sharpe_ratio']:>10.3f}")
    print(sep)
    print(f"  Exit breakdown:")
    for exit_type, count in sorted(stats["exits_by_type"].items()):
        print(f"    {exit_type:<26} {count:>10}")
    print(f"  Direction breakdown:")
    for direction, count in sorted(stats["exits_by_dir"].items()):
        print(f"    {direction:<26} {count:>10}")
    print(sep)


# =============================================================================
# Chart
# =============================================================================

def _plot_results(
    df: pd.DataFrame,
    stats: dict,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """
    Three-panel chart:
      1. Equity curve with per-trade PnL markers.
      2. Drawdown depth (filled area).
      3. PnL distribution histogram.
    """
    pnl       = df["PnL"]
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown  = running_peak - cumulative

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Golden Ribbon Backtest — {symbol} {timeframe.upper()}   "
        f"{start_date.date()} → {end_date.date()}   "
        f"Win rate: {stats['win_rate']*100:.1f}%   "
        f"PF: {stats['profit_factor']:.2f}   "
        f"Return: {stats['total_return_pct']:+.1f}%",
        fontsize=12,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1.5], hspace=0.45)

    # ── Panel 1: equity curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative.values, color="#2563EB", linewidth=1.4, label="Cumulative PnL")
    ax1.axhline(0, color="#6B7280", linewidth=0.6, linestyle="--")

    wins_idx   = df.index[pnl > 0].tolist()
    losses_idx = df.index[pnl < 0].tolist()
    ax1.scatter(wins_idx,   cumulative.iloc[wins_idx].values,   color="#16A34A", s=18, zorder=3, label="Win")
    ax1.scatter(losses_idx, cumulative.iloc[losses_idx].values, color="#DC2626", s=18, zorder=3, label="Loss")

    ax1.set_ylabel("Cumulative PnL")
    ax1.set_xlabel("Trade #")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel 2: drawdown ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(range(len(drawdown)), -drawdown.values, 0, color="#DC2626", alpha=0.35, label="Drawdown")
    ax2.plot(-drawdown.values, color="#DC2626", linewidth=0.8)
    ax2.axhline(0, color="#6B7280", linewidth=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Trade #")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel 3: PnL distribution ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    wins_pnl   = pnl[pnl > 0].values
    losses_pnl = pnl[pnl <= 0].values

    if len(wins_pnl):
        ax3.hist(wins_pnl,   bins=30, color="#16A34A", alpha=0.65, label=f"Wins ({len(wins_pnl)})")
    if len(losses_pnl):
        ax3.hist(losses_pnl, bins=30, color="#DC2626", alpha=0.65, label=f"Losses ({len(losses_pnl)})")

    ax3.axvline(0, color="#6B7280", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("PnL per trade")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)
    ax3.grid(True, linewidth=0.4, alpha=0.5)

    plt.show()


# =============================================================================
# CSV helpers
# =============================================================================

_CSV_HEADER = [
    "EntryTime", "Direction",
    "Entry", "SL", "TP", "PositionSize",
    "ExitPrice", "ExitType",
    "PnL", "BalanceAfter",
    "Pattern", "Reason",
]


def _write_csv_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)


def _append_trade_row(
    path: str,
    trade,
    exit_price: float,
    exit_type: str,
    pnl: float,
    balance_after: float,
) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            trade.candle.time,
            trade.direction,
            f"{trade.entry:.5f}",
            f"{trade.stop_loss:.5f}"  if trade.stop_loss  is not None else "",
            f"{trade.take_profit:.5f}" if trade.take_profit is not None else "",
            f"{trade.position_size:.4f}",
            f"{exit_price:.5f}",
            exit_type,
            f"{pnl:.4f}",
            f"{balance_after:.4f}",
            trade.pattern_name,
            trade.reason,
        ])

# =============================================================================
# Experiment summary log helpers
# =============================================================================

_SUMMARY_HEADER = [
    "RunTimestamp", "Comment", "Symbol", "Timeframe", "StartDate", "EndDate", "Candles", "CutOff",
    "InitialBalance", "FinalBalance", "UseCloseSignal",
    "TotalTrades", "Wins", "Losses", "Breakeven", "WinRate", "ProfitFactor",
    "AvgWin", "AvgLoss", "BestTrade", "WorstTrade", "TotalPnL",
    "MaxConsecWins", "MaxConsecLoss", "MaxDrawdown", "MaxDrawdownPct",
    "TotalReturnPct", "SharpeRatio",
    "ExitSL", "ExitTP", "ExitStrategyClose", "ExitForceClose",
    "BuyTrades", "SellTrades",
    "StrategyParams", "TradeLogFile",
]


def _append_summary_row(
    path: str,
    comment_name: str,
    stats: dict,
    strategy_params: dict,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    candles_count: int,
    cut_off: int,
    initial_balance: float,
    final_balance: float,
    use_close_signal: bool,
    trade_log_file: str,
) -> None:
    """Append one backtest result row to a persistent experiment results CSV."""
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


# =============================================================================
# Backtest
# =============================================================================

def backtest_golden_ribbon(
    symbol: str         = SYMBOLS["Gold"],
    timeframe: str      = "5m",
    start_date: datetime = datetime(2022, 1, 1),
    end_date: datetime   = datetime(2025, 12, 31),
    cut_off: int        = 300,
    initial_balance: float = 200.0,
    use_close_signal: bool = True,
    output_file: str    = "trades_golden_ribbon.csv",
    strategy_params: Optional[dict] = None,
    comment_name: str   = "manual_run",
    summary_file: str   = "golden_ribbon_experiment_results.csv",
    plot: bool          = True,
) -> Optional[dict]:
    """
    Run a candle-by-candle backtest of the GoldenRibbonStrategy.

    Parameters
    ----------
    symbol            MT5 symbol string.
    timeframe         Key into TIMEFRAMES dict (e.g. "5m", "1m", "15m").
    start_date        First candle date/time to include in the dataset.
    end_date          Last candle date/time to include in the dataset.
    cut_off           Candles used to warm up the strategy before trading starts.
                      Must be large enough for all indicators to initialise
                      (recommended: ≥ 200 for EMA 50 + ATR 20-bar average).
    initial_balance   Starting account balance for position sizing.
    use_close_signal  Whether the strategy emits CLOSE signals.
                      When True:  trades close on ribbon reversal or RSI 50 cross.
                      When False: trades only close via SL / TP.
    output_file       Path to the per-trade CSV results file.
    strategy_params   Dictionary of GoldenRibbonStrategy constructor inputs.
                      Paste trained parameters here to reproduce a candidate.
    comment_name      Manual label saved into the experiment summary CSV so
                      each run is identifiable later.
    summary_file      Append-only CSV where one summary row is saved per run.
    plot              Whether to show the equity/drawdown/PnL chart.

    Exit priority per candle (when a trade is open)
    -----------------------------------------------
    1. SL / TP intrabar check  — based on candle high/low (can be hit at any
       point during the candle). Stops are always respected first.
       Calls strategy.on_stop_hit() on SL so the cooldown resets.
    2. Strategy CLOSE signal   — indicator-based, fires at candle close.
       Close price = candle.close of the signal candle.
    No new entry is opened on the same candle that closes a trade.
    """
    print(f"\n  Golden Ribbon Backtest")
    print(f"  Symbol: {symbol}  |  Timeframe: {timeframe}")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(f"  Balance: {initial_balance:.2f}  |  Close signal: {use_close_signal}")
    print(f"  Warming up on first {cut_off} candles…")
    print(f"  Comment: {comment_name}")

    # Accept trained inputs in this format:
    # strategy_params = {"ema_fast_period": 13, ...}
    # Any missing parameter falls back to GoldenRibbonStrategy defaults.
    if strategy_params is None:
        strategy_params = {}
    else:
        strategy_params = dict(strategy_params)

    print(f"  Strategy params: {strategy_params if strategy_params else 'DEFAULTS'}")

    # ── Data ─────────────────────────────────────────────────────────────
    provider = MT5MarketDataProvider()
    series   = provider.fetch_range(
        symbol,
        TIMEFRAMES[timeframe],
        start_date,
        end_date,
    )

    total_candles = len(series._candles)
    print(f"  Loaded {total_candles:,} candles ({total_candles - cut_off:,} tradeable after warm-up)")

    if total_candles <= cut_off:
        print(
            f"  ERROR: dataset has {total_candles} candles but cut_off is {cut_off}. "
            "Increase the date range or reduce cut_off."
        )
        return None

    warmup_series = series.subseries(0, cut_off)

    # ── Strategy + risk engine ────────────────────────────────────────────
    strategy     = GoldenRibbonStrategy(
        warmup_series,
        use_close_signal=use_close_signal,
        **strategy_params,
    )
    risk_engine  = RiskManager()

    # ── State ─────────────────────────────────────────────────────────────
    balance                 = initial_balance
    active_trade            = None
    active_trade_entry_idx  = None
    trades_count            = 0

    _write_csv_header(output_file)

    # ── Main loop ─────────────────────────────────────────────────────────
    for i in range(cut_off, total_candles):
        candle = series._candles[i]

        # Step 1 — advance all indicators, collect any signal
        signal = strategy.update(candle)

        # Step 2 — trade management (skip the candle the trade opened on)
        if active_trade is not None and i > active_trade_entry_idx:

            # 2a. SL / TP — intrabar price check takes priority
            exit_result = TradeSimulation.check_exit(active_trade, candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl
                trades_count += 1

                # Notify strategy so cooldown/state resets correctly.
                if hasattr(strategy, "on_trade_closed"):
                    strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                elif exit_type == "SL":
                    strategy.on_stop_hit()

                _append_trade_row(output_file, active_trade, exit_price, exit_type, pnl, balance)
                active_trade           = None
                active_trade_entry_idx = None
                continue  # no new entry on this candle

            # 2b. Strategy CLOSE signal — fires at candle close if SL/TP not hit
            if use_close_signal and signal is not None and signal.signal == "CLOSE":
                exit_price = candle.close
                direction  = active_trade.direction.upper()

                if direction == "BUY":
                    pnl = (exit_price - active_trade.entry) * active_trade.position_size
                else:
                    pnl = (active_trade.entry - exit_price) * active_trade.position_size

                balance += pnl
                trades_count += 1

                if hasattr(strategy, "on_trade_closed"):
                    strategy.on_trade_closed(exit_type="StrategyClose", exit_price=exit_price)

                _append_trade_row(output_file, active_trade, exit_price, "StrategyClose", pnl, balance)
                active_trade           = None
                active_trade_entry_idx = None
                continue  # no new entry on this candle

            # Trade still open, nothing triggered — skip entry check
            continue

        # Step 3 — open a new trade when flat and signal is BUY or SELL
        if (
            active_trade is None
            and signal is not None
            and signal.signal in ("BUY", "SELL")
        ):
            active_trade           = risk_engine.build_trade(signal, balance)
            active_trade_entry_idx = i

            # Keep strategy state aligned with the risk-managed trade object
            # when the strategy implementation supports sync_trade().
            if hasattr(strategy, "sync_trade"):
                strategy.sync_trade(active_trade)

    # ── Force-close any trade still open at end of data ──────────────────
    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price  = last_candle.close
        direction   = active_trade.direction.upper()

        if direction == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size

        balance += pnl
        trades_count += 1

        if hasattr(strategy, "on_trade_closed"):
            strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)

        _append_trade_row(output_file, active_trade, exit_price, "ForceClose", pnl, balance)

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n  Backtest complete. Reading results from {output_file}…")

    df = pd.read_csv(output_file)

    if df.empty:
        print("  No trades were executed. Check cut_off and strategy parameters.")
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


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Paste the best params from training here.
    # strategy_params = {
    #     "ema_fast_period": 13,
    #     "ema_slow_period": 21,
    #     "ema_trend_period": 100,
    #     "rsi_period": 14,
    #     "rsi_slope_offset": 2,
    #     "rsi_buy_level": 52.0,
    #     "rsi_sell_level": 48.0,
    #     "atr_period": 10,
    #     "ema_offset": 1,
    #     "ema_fast_slope_threshold": 0.04
    # }

    strategy_params = {
        "ema_fast_period": 13,
        "ema_slow_period": 21,
        "ema_trend_period": 100,
        "rsi_period": 14,
        "rsi_slope_offset": 1,
        "rsi_buy_level": 50.0, #48
        "rsi_sell_level": 50.0,
        "atr_period": 10,
        "ema_fast_slope_threshold": 0.04,
    }

    # Change this manually before each run so the summary row is identifiable.
    comment_name = "1st test on training data"

    backtest_golden_ribbon(
        symbol           = SYMBOLS["Gold"],
        timeframe        = "5m",
        start_date       = datetime(2024, 1, 1),
        end_date         = datetime(2025, 12, 30),
        cut_off          = 300,
        initial_balance  = 200.0,
        use_close_signal = False,
        output_file      = "trades_golden_ribbon.csv",
        strategy_params  = strategy_params,
        comment_name     = comment_name,
        summary_file     = "golden_ribbon_experiment_results.csv",
        plot             = True,
    )
