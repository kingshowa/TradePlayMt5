from __future__ import annotations

import csv
import itertools
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

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
    SL is assumed to have been hit first.
    """

    @staticmethod
    def check_exit(trade, candle) -> Optional[Tuple]:
        direction = trade.direction.upper()
        entry     = trade.entry
        sl        = trade.stop_loss
        tp        = trade.take_profit
        size      = trade.position_size

        if direction == "BUY":
            if sl is not None and candle.low <= sl:
                return sl, "SL", (sl - entry) * size
            if tp is not None and candle.high >= tp:
                return tp, "TP", (tp - entry) * size
        else:
            if sl is not None and candle.high >= sl:
                return sl, "SL", (entry - sl) * size
            if tp is not None and candle.low <= tp:
                return tp, "TP", (entry - tp) * size

        return None


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
            f"{trade.stop_loss:.5f}"   if trade.stop_loss   is not None else "",
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
# Statistics
# =============================================================================

def _empty_stats() -> Dict[str, Any]:
    return {
        "total_trades":     0,
        "wins":             0,
        "losses":           0,
        "breakeven":        0,
        "win_rate":         0.0,
        "profit_factor":    0.0,
        "avg_win":          0.0,
        "avg_loss":         0.0,
        "best_trade":       0.0,
        "worst_trade":      0.0,
        "total_pnl":        0.0,
        "max_consec_wins":  0,
        "max_consec_loss":  0,
        "max_drawdown":     0.0,
        "max_drawdown_pct": 0.0,
        "total_return_pct": 0.0,
        "sharpe_ratio":     0.0,
        "calmar_ratio":     0.0,
        "exits_by_type":    {},
        "exits_by_dir":     {},
    }


def _compute_stats(df: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    pnl    = df["PnL"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit  = wins.sum()
    gross_loss    = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    max_cw = max_cl = cur_cw = cur_cl = 0
    for val in pnl:
        if val > 0:
            cur_cw += 1; cur_cl = 0
        elif val < 0:
            cur_cl += 1; cur_cw = 0
        max_cw = max(max_cw, cur_cw)
        max_cl = max(max_cl, cur_cl)

    cumulative   = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown     = running_peak - cumulative
    max_dd       = drawdown.max() if not drawdown.empty else 0.0
    peak_at_dd   = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else 0.0
    denom        = initial_balance + peak_at_dd
    max_dd_pct   = (max_dd / denom * 100.0) if denom > 0 else 0.0

    std_pnl = pnl.std()
    sharpe  = (
        pnl.mean() / std_pnl * math.sqrt(len(pnl))
        if std_pnl and std_pnl > 0 else 0.0
    )

    total_return_pct = (pnl.sum() / initial_balance * 100.0) if initial_balance else 0.0
    calmar_ratio     = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    return {
        "total_trades":     len(df),
        "wins":             len(wins),
        "losses":           len(losses),
        "breakeven":        len(pnl[pnl == 0]),
        "win_rate":         len(wins) / len(df) if len(df) else 0.0,
        "profit_factor":    profit_factor,
        "avg_win":          wins.mean()   if not wins.empty   else 0.0,
        "avg_loss":         losses.mean() if not losses.empty else 0.0,
        "best_trade":       pnl.max()     if not pnl.empty    else 0.0,
        "worst_trade":      pnl.min()     if not pnl.empty    else 0.0,
        "total_pnl":        pnl.sum(),
        "max_consec_wins":  max_cw,
        "max_consec_loss":  max_cl,
        "max_drawdown":     max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_return_pct": total_return_pct,
        "sharpe_ratio":     sharpe,
        "calmar_ratio":     calmar_ratio,
        "exits_by_type":    df["ExitType"].value_counts().to_dict(),
        "exits_by_dir":     df["Direction"].value_counts().to_dict(),
    }


# =============================================================================
# Scoring
# =============================================================================

def _score_result_row(row: Dict[str, Any], min_trades: int = 20) -> float:
    """
    Composite robustness score — higher is better.

    Design philosophy
    -----------------
    The score rewards three things in descending priority:

      1. Quality    — profit factor and Calmar ratio. A high PF with low
                      drawdown is the hallmark of a robust edge, not luck.
                      Calmar specifically rewards return-per-unit-of-drawdown,
                      which is the practical metric a live trader cares about.

      2. Consistency — win rate and Sharpe. Win rate matters psychologically
                      (a strategy with 35% WR is brutal to trade even at 3:1).
                      Sharpe rewards smooth equity curves over spiky ones.

      3. Volume     — trade count. Too few trades means the result is
                      statistically meaningless. The penalty is progressive:
                      sharper below half of min_trades.

    Penalised terms
    ---------------
    Drawdown is penalised separately from Calmar. Even if Calmar is high,
    a deep absolute drawdown erodes the initial account on low-balance live
    accounts. The coefficient is kept lighter than the reward terms —
    excessive DD penalty forces the scorer to prefer strategies that barely
    trade (low DD by inactivity).

    Inf PF handling
    ---------------
    A PF of inf means zero losing trades. This is a sign of too few trades
    or look-ahead overfitting, not genuine edge. It is capped at 8.0, which
    is an excellent but not suspicious PF, ensuring these runs don't dominate.
    """
    trades  = float(row.get("total_trades",     0) or 0)
    pf      = float(row.get("profit_factor",    0) or 0)
    ret     = float(row.get("total_return_pct", 0) or 0)
    dd_pct  = float(row.get("max_drawdown_pct", 0) or 0)
    wr      = float(row.get("win_rate",         0) or 0)
    sharpe  = float(row.get("sharpe_ratio",     0) or 0)
    calmar  = float(row.get("calmar_ratio",     0) or 0)

    # Cap inf profit factor — it signals a suspiciously clean run
    pf = min(pf, 8.0) if not math.isinf(pf) else 8.0

    # Trade-count penalty: zero at min_trades, steeper below half
    if trades >= min_trades:
        trade_penalty = 0.0
    elif trades >= min_trades / 2:
        trade_penalty = (min_trades - trades) * 1.0
    else:
        trade_penalty = (min_trades - trades) * 3.0

    # Negative return is a hard disqualifier — no PF manipulation should rescue it
    if ret <= 0:
        return round(-9999.0 - trade_penalty, 4)

    score = (
        pf     * 30.0    # quality gate — range roughly 0–240
        + calmar  * 15.0    # return per unit of drawdown — range roughly 0–300
        + wr      * 80.0    # win rate — range 0–80
        + sharpe  * 12.0    # equity curve smoothness
        + ret     *  0.8    # raw return (low weight — already in calmar)
        - dd_pct  *  2.5    # drawdown penalty — independent of calmar
        - trade_penalty
    )
    return round(score, 4)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable run configuration.

    start_date / end_date control the dataset fetched via fetch_range().
    cut_off is the leading candle count used exclusively for indicator warm-up.
    """
    symbol:           str      = SYMBOLS["Gold"]
    timeframe:        str      = "5m"
    start_date:       datetime = field(default_factory=lambda: datetime(2022, 1, 1))
    end_date:         datetime = field(default_factory=lambda: datetime(2023, 12, 31))
    cut_off:          int      = 300
    initial_balance:  float    = 200.0
    use_close_signal: bool     = True
    output_file:      str      = "trades_golden_ribbon.csv"
    plot_results:     bool     = False


@dataclass
class StudyConfig:
    backtest:              BacktestConfig = field(default_factory=BacktestConfig)
    temp_trade_log:        str            = "_study_trades_tmp.csv"
    summary_output:        str            = "golden_ribbon_parameter_study_results.csv"
    delete_temp_trade_log: bool           = True
    top_n_to_print:        int            = 15
    sort_by:               str            = "score"
    min_trades:            int            = 20


# =============================================================================
# Plotting — single-run equity view
# =============================================================================

def _plot_results(
    df: pd.DataFrame,
    stats: dict,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping chart.")
        return

    pnl          = df["PnL"]
    cumulative   = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown     = running_peak - cumulative

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Golden Ribbon  |  {symbol} {timeframe.upper()}  |  "
        f"{start_date.date()} → {end_date.date()}\n"
        f"Win rate: {stats['win_rate']*100:.1f}%   "
        f"PF: {stats['profit_factor']:.2f}   "
        f"Return: {stats['total_return_pct']:+.1f}%   "
        f"Max DD: {stats['max_drawdown_pct']:.1f}%   "
        f"Calmar: {stats['calmar_ratio']:.2f}",
        fontsize=11,
        fontweight="bold",
    )

    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1.5], hspace=0.45)

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

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(range(len(drawdown)), -drawdown.values, 0, color="#DC2626", alpha=0.35, label="Drawdown")
    ax2.plot(-drawdown.values, color="#DC2626", linewidth=0.8)
    ax2.axhline(0, color="#6B7280", linewidth=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, linewidth=0.4, alpha=0.5)

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
# Best-outcome highlight
# =============================================================================

_PARAM_COLS = [
    "ema_fast_period", "ema_slow_period", "ema_trend_period",
    "rsi_period", "rsi_slope_offset",
    "rsi_buy_level", "rsi_sell_level",
    "atr_period", "ema_offset",
]

_METRIC_COLS = [
    "score", "total_trades", "win_rate", "profit_factor",
    "calmar_ratio", "max_drawdown_pct", "total_return_pct", "sharpe_ratio",
    "exit_sl", "exit_tp", "exit_strategy_close",
]


def _print_best_outcome(best: Dict[str, Any]) -> None:
    """
    Print a clearly formatted block with the best parameter set and its
    metrics. This is the output intended as input to the next test stage.
    """
    sep   = "═" * 60
    thin  = "─" * 60

    print(f"\n{sep}")
    print(f"  BEST PARAMETER SET  —  use for next-level testing")
    print(sep)

    print(f"\n  {'Indicator parameters':}")
    print(thin)
    params = {k: best.get(k) for k in _PARAM_COLS if k in best}
    for k, v in params.items():
        print(f"    {k:<28} {v}")

    print(f"\n  {'Performance metrics':}")
    print(thin)
    metrics = {k: best.get(k) for k in _METRIC_COLS if k in best}
    fmt_map = {
        "win_rate":         lambda v: f"{v * 100:.2f}%",
        "profit_factor":    lambda v: f"{v:.3f}",
        "calmar_ratio":     lambda v: f"{v:.3f}",
        "max_drawdown_pct": lambda v: f"{v:.2f}%",
        "total_return_pct": lambda v: f"{v:+.2f}%",
        "sharpe_ratio":     lambda v: f"{v:.4f}",
        "score":            lambda v: f"{v:.4f}",
    }
    for k, v in metrics.items():
        formatted = fmt_map[k](v) if k in fmt_map and v is not None else str(v)
        print(f"    {k:<28} {formatted}")

    print(f"\n{sep}")
    print(f"  NEXT STAGE CONFIG  —  copy into BacktestConfig / strategy_params")
    print(sep)

    def _fmt(v: Any) -> str:
        return f'"{v}"' if isinstance(v, str) else str(v)

    print(f"\n  strategy_params = {{")
    for k, v in params.items():
        print(f'      "{k}": {_fmt(v)},')
    print(f"  }}")
    print(f"\n{sep}\n")


# =============================================================================
# Core backtest runner (single parameter set, reuses a pre-fetched series)
# =============================================================================

def run_single_backtest(
    series,
    strategy_params: Dict[str, Any],
    config: BacktestConfig,
    trade_log_path: str,
) -> Dict[str, Any]:
    """
    Execute one full backtest run over the provided series and return a
    result dict suitable for appending to the study summary DataFrame.

    Parameters
    ----------
    series            Pre-fetched MarketSeries — shared across all study runs
                      to avoid redundant MT5 fetches.
    strategy_params   Keyword arguments forwarded directly to GoldenRibbonStrategy.
    config            BacktestConfig for balance, cut_off, and close-signal flag.
    trade_log_path    CSV path for this run's trade log (overwritten each call).
    """
    warmup_series = series.subseries(0, config.cut_off)
    strategy      = GoldenRibbonStrategy(
        warmup_series,
        use_close_signal=config.use_close_signal,
        **strategy_params,
    )
    risk_engine   = RiskManager()
    balance       = config.initial_balance
    active_trade  = None
    active_idx    = None

    _write_csv_header(trade_log_path)

    for i in range(config.cut_off, len(series._candles)):
        candle = series._candles[i]
        signal = strategy.update(candle)

        if active_trade is not None and i > active_idx:
            # 2a. SL / TP — intrabar check, highest priority
            exit_result = TradeSimulation.check_exit(active_trade, candle)
            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl
                strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                _append_trade_row(trade_log_path, active_trade, exit_price, exit_type, pnl, balance)
                active_trade = None
                active_idx   = None
                continue

            # 2b. Strategy CLOSE signal — indicator-based, fires at candle close
            if config.use_close_signal and signal is not None and signal.signal == "CLOSE":
                exit_price = candle.close
                pnl = (
                    (exit_price - active_trade.entry) * active_trade.position_size
                    if active_trade.direction.upper() == "BUY"
                    else (active_trade.entry - exit_price) * active_trade.position_size
                )
                balance += pnl
                strategy.on_trade_closed(exit_type="StrategyClose", exit_price=exit_price)
                _append_trade_row(trade_log_path, active_trade, exit_price, "StrategyClose", pnl, balance)
                active_trade = None
                active_idx   = None
                continue

            continue

        # Step 3 — open entry when flat
        if active_trade is None and signal is not None and signal.signal in ("BUY", "SELL"):
            active_trade = risk_engine.build_trade(signal, balance)
            active_idx   = i
            strategy.sync_trade(active_trade)

    # Force-close any trade still open at end of data
    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price  = last_candle.close
        pnl = (
            (exit_price - active_trade.entry) * active_trade.position_size
            if active_trade.direction.upper() == "BUY"
            else (active_trade.entry - exit_price) * active_trade.position_size
        )
        balance += pnl
        strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)
        _append_trade_row(trade_log_path, active_trade, exit_price, "ForceClose", pnl, balance)

    # Build result row
    df = pd.read_csv(trade_log_path)
    if df.empty:
        stats = _empty_stats()
    else:
        df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
        stats     = _compute_stats(df, config.initial_balance)

    result_row: Dict[str, Any] = {
        **strategy_params,
        "symbol":          config.symbol,
        "timeframe":       config.timeframe,
        "start_date":      config.start_date.date(),
        "end_date":        config.end_date.date(),
        "cut_off":         config.cut_off,
        "initial_balance": config.initial_balance,
        "use_close_signal":config.use_close_signal,
        "final_balance":   balance,
        **stats,
    }

    exit_counts = stats.get("exits_by_type", {})
    result_row["exit_sl"]             = exit_counts.get("SL",            0)
    result_row["exit_tp"]             = exit_counts.get("TP",            0)
    result_row["exit_strategy_close"] = exit_counts.get("StrategyClose", 0)
    result_row["exit_force_close"]    = exit_counts.get("ForceClose",    0)

    return result_row


# =============================================================================
# Single-run convenience entry point
# =============================================================================

def backtest_golden_ribbon(
    strategy_params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Single-run backtest mode. Saves a trade-log CSV, prints stats,
    and optionally plots results.
    """
    strategy_params = strategy_params or {}
    config          = config or BacktestConfig()

    print("\nGolden Ribbon Backtest")
    print(f"Symbol: {config.symbol} | Timeframe: {config.timeframe}")
    print(f"Period: {config.start_date.date()} → {config.end_date.date()}")
    print(f"Balance: {config.initial_balance:.2f} | Close signal: {config.use_close_signal}")
    print(f"Trade log: {config.output_file}")
    print(f"Strategy params: {strategy_params}\n")

    provider = MT5MarketDataProvider()
    series   = provider.fetch_range(
        config.symbol,
        TIMEFRAMES[config.timeframe],
        config.start_date,
        config.end_date,
    )

    row = run_single_backtest(
        series=series,
        strategy_params=strategy_params,
        config=config,
        trade_log_path=config.output_file,
    )

    df = pd.read_csv(config.output_file)
    if df.empty:
        print("No trades were executed.")
        return df

    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)

    sep = "=" * 56
    print(sep)
    print(f"  Final balance:     {row['final_balance']:.2f}")
    print(f"  Total trades:      {row['total_trades']}")
    print(f"  Win rate:          {row['win_rate'] * 100:.2f}%")
    print(f"  Profit factor:     {row['profit_factor']:.3f}")
    print(f"  Calmar ratio:      {row['calmar_ratio']:.3f}")
    print(f"  Total return:      {row['total_return_pct']:+.2f}%")
    print(f"  Max drawdown %:    {row['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio:      {row['sharpe_ratio']:.4f}")
    print(f"  Exit SL/TP/SC/FC:  {row['exit_sl']} / {row['exit_tp']} "
          f"/ {row['exit_strategy_close']} / {row['exit_force_close']}")
    print(sep)

    if config.plot_results:
        stats = _compute_stats(df, config.initial_balance)
        _plot_results(df, stats, config.symbol, config.timeframe,
                      config.start_date, config.end_date)

    return df


# =============================================================================
# Parameter set builders
# =============================================================================

def _is_valid_ribbon(params: Dict[str, Any]) -> bool:
    """
    Guard invalid ribbon combinations.

    Rules
    -----
    fast < slow          — crossed lines produce no cross signal ever.
    slow - fast >= 6     — minimum ribbon separation to avoid spaghetti signals.
    trend > slow         — trend EMA must be slower than the ribbon to function
                           as a structural bias filter; equal values make the
                           filter redundant and the trend gate fires too early.
    """
    fast  = params.get("ema_fast_period",  0)
    slow  = params.get("ema_slow_period",  0)
    trend = params.get("ema_trend_period", 0)
    return fast < slow and (slow - fast) >= 6 and trend > slow


def build_grid(param_ranges: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Full Cartesian product of all param_ranges, invalid ribbon combos removed."""
    keys   = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]
    raw    = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    valid  = [p for p in raw if _is_valid_ribbon(p)]

    removed = len(raw) - len(valid)
    if removed:
        print(f"  Grid filter: removed {removed} invalid ribbon combinations "
              f"({len(valid)} valid remain).")
    return valid


def build_pairwise_grids(
    pair_ranges: Dict[Tuple[str, str], Dict[str, Sequence[Any]]],
    fixed_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build independent grids for each parameter pair, merged with fixed_params.
    Each pair is swept in isolation — cross-pair interactions are not explored.
    Invalid ribbon combos are removed via _is_valid_ribbon.
    """
    fixed_params = fixed_params or {}
    seen:      set             = set()
    all_rows:  List[Dict[str, Any]] = []

    for _, ranges in pair_ranges.items():
        for combo in build_grid(ranges):
            row = {**fixed_params, **combo}
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    return all_rows


# =============================================================================
# Parameter study runner
# =============================================================================

def run_parameter_study(
    param_sets: Iterable[Dict[str, Any]],
    study_config: StudyConfig,
) -> pd.DataFrame:
    """
    Run every parameter set in param_sets, collect results, sort by score,
    save to CSV, and print the top-N summary plus a highlighted best outcome.

    The MarketSeries is fetched once and reused across all runs to avoid
    hammering the MT5 terminal with repeated historical data requests.
    """
    provider = MT5MarketDataProvider()
    bt       = study_config.backtest

    print("\n  Golden Ribbon Parameter Study")
    print(f"  Symbol: {bt.symbol} | Timeframe: {bt.timeframe}")
    print(f"  Period: {bt.start_date.date()} → {bt.end_date.date()}")
    print(f"  Close signal: {bt.use_close_signal} | Min trades: {study_config.min_trades}")
    print(f"  Summary: {study_config.summary_output}\n")

    series = provider.fetch_range(
        bt.symbol,
        TIMEFRAMES[bt.timeframe],
        bt.start_date,
        bt.end_date,
    )
    total_candles = len(series._candles)
    print(f"  Loaded {total_candles:,} candles "
          f"({total_candles - bt.cut_off:,} tradeable after warm-up)\n")

    temp_log   = Path(study_config.temp_trade_log)
    results:   List[Dict[str, Any]] = []
    param_sets = list(param_sets)
    total_runs = len(param_sets)

    print(f"  Total runs: {total_runs}\n")

    for idx, params in enumerate(param_sets, start=1):
        tag = (
            f"fast={params.get('ema_fast_period')} "
            f"slow={params.get('ema_slow_period')} "
            f"trend={params.get('ema_trend_period')} "
            f"rsi_b={params.get('rsi_buy_level')} "
            f"rsi_s={params.get('rsi_sell_level')} "
            f"rsi_p={params.get('rsi_period')} "
            f"atr={params.get('atr_period')}"
        )
        print(f"  [{idx:>4}/{total_runs}]  {tag}")

        row          = run_single_backtest(
            series=series,
            strategy_params=params,
            config=bt,
            trade_log_path=str(temp_log),
        )
        row["score"] = _score_result_row(row, min_trades=study_config.min_trades)
        results.append(row)

        if study_config.delete_temp_trade_log and temp_log.exists():
            temp_log.unlink(missing_ok=True)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        results_df.to_csv(study_config.summary_output, index=False)
        print("  No results generated.")
        return results_df

    sort_col   = study_config.sort_by if study_config.sort_by in results_df.columns else "score"
    results_df = results_df.sort_values(
        by=[sort_col, "profit_factor", "calmar_ratio", "total_return_pct", "win_rate"],
        ascending=False,
    )
    results_df.to_csv(study_config.summary_output, index=False)

    # Apply min-trade filter for display only — full results still saved to CSV
    display_df = results_df[
        results_df["total_trades"] >= study_config.min_trades
    ] if study_config.min_trades > 0 else results_df

    print(f"\n  {'─' * 56}")
    print(f"  Top {study_config.top_n_to_print} results "
          f"(min {study_config.min_trades} trades)")
    print(f"  {'─' * 56}\n")

    display_cols = [
        c for c in [
            "score",
            "ema_fast_period", "ema_slow_period", "ema_trend_period",
            "rsi_period", "rsi_buy_level", "rsi_sell_level",
            "atr_period",
            "total_trades", "win_rate", "profit_factor",
            "calmar_ratio", "max_drawdown_pct",
            "total_return_pct", "sharpe_ratio",
        ] if c in display_df.columns
    ]

    preview = display_df.head(study_config.top_n_to_print)

    if preview.empty:
        print("  No runs met the minimum trade threshold.")
    else:
        print(preview[display_cols].to_string(index=False))

    print(f"\n  Full results saved to: {study_config.summary_output}")

    # ── Best-outcome highlight ─────────────────────────────────────────────
    if not display_df.empty:
        best = display_df.iloc[0].to_dict()
        _print_best_outcome(best)

    return results_df


# =============================================================================
# Parameter presets — carefully tuned for Gold 5M
# =============================================================================

def preset_default_params() -> Dict[str, Any]:
    """
    Baseline parameters — the starting point before any study.
    These are the theoretically motivated defaults for Gold 5M.
    """
    strategy_params = {
        "ema_fast_period": 13,
        "ema_slow_period": 21,
        "ema_trend_period": 100,
        "rsi_period": 14,
        "rsi_slope_offset": 2,
        "rsi_buy_level": 52.0,
        "rsi_sell_level": 48.0,
        "atr_period": 10,
        "ema_offset": 1,
    }
    # return {
    #     "ema_fast_period": 7,
    #     "ema_slow_period": 21,
    #     "ema_trend_period": 50,
    #     "rsi_period": 10,
    #     "rsi_slope_offset": 2,
    #     "rsi_buy_level": 54.0,
    #     "rsi_sell_level": 48.0,
    #     "atr_period": 14,
    #     "ema_offset": 1,
    #     "max_candles":      200,
    # }
    return strategy_params


def preset_full_grid() -> List[Dict[str, Any]]:
    """
    Full Cartesian grid across all tunable dimensions.

    Parameter rationale (Gold 5M context)
    ───────────────────────────────────────
    ema_fast_period [7, 9, 11, 13]
        Odd numbers only — symmetric around 9 (the classic ribbon fast).
        7 = very reactive, catches moves early but more false crosses.
        13 = slower, higher signal quality, fewer trades.

    ema_slow_period [17, 21, 25]
        Must maintain separation of ≥ 6 from fast (enforced by _is_valid_ribbon).
        17 keeps the ribbon responsive; 25 adds structure.

    ema_trend_period [50, 100]
        50 bars × 5min = 4-hour intraday trend. Fast enough to flip on session
        reversals. 100 bars = 8-hour multi-session trend, more conservative.
        200 is excluded — on 5M it takes 1000 candles (~16 hours) to fully
        weight a new trend, making it too laggy for intraday scalping.

    rsi_period [10, 14]
        10 = more reactive RSI, better for fast ribbon crosses on 5M.
        14 = standard, smoother, fewer false momentum reads.

    rsi_slope_offset [2, 3]
        Controls how many bars back RSI slope is measured.
        2 = sensitive to very recent momentum.
        3 = looks slightly further back, reduces noise from single-candle spikes.
        1 is too noisy; 4 is too slow for a 5M scalping system.

    rsi_buy_level / rsi_sell_level [52/48, 54/46, 56/44]
        The 48/52 pair is the minimum no-trade buffer around 50.
        54/46 provides a moderate quality gate — meaningful for Gold which
        frequently oscillates around 50 in low-volatility periods.
        56/44 is the strictest filter — highest quality signals, fewest trades.
        Symmetric pairs only: buy and sell levels are always equidistant from 50.

    atr_period [10, 14]
        10 = adapts faster to sudden volatility changes (useful around news).
        14 = smoothed, less sensitive to single-candle ATR spikes.
    """
    return build_grid({
        "ema_fast_period":  [11, 13, 15],
        "ema_slow_period":  [21, 25, 29],
        "ema_trend_period": [80, 100, 200],
        "rsi_period":       [14],
        "rsi_slope_offset": [1, 2],
        "rsi_buy_level":    [48, 50, 52.0],
        "rsi_sell_level":   [48.0, 50.0, 52.0],
        "atr_period":       [10, 14],
        "ema_fast_slope_threshold": [0.04, 0.05]
    })


def preset_ribbon_focus() -> List[Dict[str, Any]]:
    """
    Focused study on ribbon sensitivity only (fast × slow interaction).
    All other parameters are fixed to the theoretically motivated defaults.
    Useful for isolating the cross-detection component from RSI / ATR noise.
    """
    fixed = preset_default_params()
    return build_pairwise_grids(
        pair_ranges={
            ("ema_fast_period", "ema_slow_period"): {
                "ema_fast_period": [7, 8, 9, 10, 11, 12, 13],
                "ema_slow_period": [15, 17, 19, 21, 23, 25, 27],
            },
        },
        fixed_params=fixed,
    )


def preset_rsi_focus() -> List[Dict[str, Any]]:
    """
    Focused study on RSI sensitivity — period, slope offset, and gate levels.
    Ribbon and ATR are fixed. Useful after ribbon_focus identifies the best
    ribbon pair and you want to tune the momentum filter independently.
    """
    fixed = preset_default_params()
    return build_pairwise_grids(
        pair_ranges={
            ("rsi_period", "rsi_slope_offset"): {
                "rsi_period":       [8, 10, 12, 14],
                "rsi_slope_offset": [1, 2, 3, 4],
            },
            ("rsi_buy_level", "rsi_sell_level"): {
                "rsi_buy_level":  [50.5, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 58.0],
                "rsi_sell_level": [49.5, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 42.0],
            },
        },
        fixed_params=fixed,
    )


def preset_trend_atr_focus() -> List[Dict[str, Any]]:
    """
    Focused study on the trend proxy and volatility filter.
    Sweep ema_trend_period and atr_period together while keeping the
    ribbon and RSI fixed. Useful for understanding how structural bias
    and volatility gating interact.
    """
    fixed = preset_default_params()
    return build_pairwise_grids(
        pair_ranges={
            ("ema_trend_period", "atr_period"): {
                "ema_trend_period": [30, 40, 50, 60, 75, 100],
                "atr_period":       [7, 10, 14, 20],
            },
        },
        fixed_params=fixed,
    )


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    # ── Choose mode ───────────────────────────────────────────────────────
    # "backtest"        → single run with preset_default_params()
    # "study_full"      → full Cartesian grid (slow, thorough)
    # "study_ribbon"    → ribbon-only sweep (fast, diagnostic)
    # "study_rsi"       → RSI-only sweep    (fast, diagnostic)
    # "study_trend_atr" → trend+ATR sweep   (fast, diagnostic)
    MODE = "study_full"

    backtest_config = BacktestConfig(
        symbol           = SYMBOLS["Gold"],
        timeframe        = "5m",
        start_date       = datetime(2024, 1, 1),
        end_date         = datetime(2025, 12, 30),
        cut_off          = 300,
        initial_balance  = 200.0,
        use_close_signal = False,
        output_file      = "trades_golden_ribbon.csv",
        plot_results     = False,
    )



    if MODE == "backtest":
        backtest_golden_ribbon(
            strategy_params=preset_default_params(),
            config=backtest_config,
        )

    else:
        param_sets = {
            "study_full":      preset_full_grid,
            "study_ribbon":    preset_ribbon_focus,
            "study_rsi":       preset_rsi_focus,
            "study_trend_atr": preset_trend_atr_focus,
        }[MODE]()

        study = StudyConfig(
            backtest              = backtest_config,
            temp_trade_log        = "_golden_ribbon_study_tmp_trades.csv",
            summary_output        = f"golden_ribbon_study_{MODE}.csv",
            delete_temp_trade_log = True,
            top_n_to_print        = 15,
            sort_by               = "score",
            min_trades            = 20,
        )

        run_parameter_study(param_sets=param_sets, study_config=study)