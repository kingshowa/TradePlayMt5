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
from app.core.strategy.parabolic_sar_strategy import ParabolicSarStrategy


# =============================================================================
# Symbols
# =============================================================================

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


# =============================================================================
# Exit simulation
# =============================================================================

class TradeSimulation:
    """
    Candle-level SL / TP exit checker.

    Conservative rule:
    If SL and TP are touched in the same candle, SL is assumed first.
    """

    @staticmethod
    def check_exit(trade, candle) -> Optional[Tuple[float, str, float]]:
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


# =============================================================================
# Statistics
# =============================================================================

def _empty_stats() -> Dict[str, Any]:
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "total_pnl": 0.0,
        "max_consec_wins": 0,
        "max_consec_loss": 0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "exits_by_type": {},
        "exits_by_dir": {},
    }


def _compute_stats(df: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    pnl = df["PnL"]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_cw = max_cl = cur_cw = cur_cl = 0

    for value in pnl:
        if value > 0:
            cur_cw += 1
            cur_cl = 0
        elif value < 0:
            cur_cl += 1
            cur_cw = 0

        max_cw = max(max_cw, cur_cw)
        max_cl = max(max_cl, cur_cl)

    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative

    max_dd = drawdown.max() if not drawdown.empty else 0.0
    peak_at_dd = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else 0.0

    denom = initial_balance + peak_at_dd
    max_dd_pct = max_dd / denom * 100 if denom > 0 else 0.0

    std_pnl = pnl.std()
    sharpe = (
        pnl.mean() / std_pnl * math.sqrt(len(pnl))
        if std_pnl and std_pnl > 0
        else 0.0
    )

    total_return_pct = pnl.sum() / initial_balance * 100 if initial_balance else 0.0
    calmar_ratio = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    return {
        "total_trades": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(pnl[pnl == 0]),
        "win_rate": len(wins) / len(df) if len(df) else 0.0,
        "profit_factor": profit_factor,
        "avg_win": wins.mean() if not wins.empty else 0.0,
        "avg_loss": losses.mean() if not losses.empty else 0.0,
        "best_trade": pnl.max() if not pnl.empty else 0.0,
        "worst_trade": pnl.min() if not pnl.empty else 0.0,
        "total_pnl": pnl.sum(),
        "max_consec_wins": max_cw,
        "max_consec_loss": max_cl,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar_ratio,
        "exits_by_type": df["ExitType"].value_counts().to_dict(),
        "exits_by_dir": df["Direction"].value_counts().to_dict(),
    }


# =============================================================================
# Scoring
# =============================================================================

def _score_result_row(row: Dict[str, Any], min_trades: int = 20) -> float:
    trades = float(row.get("total_trades", 0) or 0)
    profit_factor = float(row.get("profit_factor", 0) or 0)
    total_return_pct = float(row.get("total_return_pct", 0) or 0)
    max_drawdown_pct = float(row.get("max_drawdown_pct", 0) or 0)
    win_rate = float(row.get("win_rate", 0) or 0)
    sharpe_ratio = float(row.get("sharpe_ratio", 0) or 0)
    calmar_ratio = float(row.get("calmar_ratio", 0) or 0)

    profit_factor = min(profit_factor, 8.0) if not math.isinf(profit_factor) else 8.0

    if trades >= min_trades:
        trade_penalty = 0.0
    elif trades >= min_trades / 2:
        trade_penalty = (min_trades - trades) * 1.0
    else:
        trade_penalty = (min_trades - trades) * 3.0

    if total_return_pct <= 0:
        return round(-9999.0 - trade_penalty, 4)

    score = (
        profit_factor * 30.0
        + calmar_ratio * 15.0
        + win_rate * 80.0
        + sharpe_ratio * 12.0
        + total_return_pct * 0.8
        - max_drawdown_pct * 2.5
        - trade_penalty
    )

    return round(score, 4)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = SYMBOLS["Gold"]
    timeframe: str = "5m"
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2025, 12, 31))
    cut_off: int = 300
    initial_balance: float = 200.0
    use_close_signal: bool = True
    output_file: str = "trades_parabolic_sar.csv"
    plot_results: bool = False


@dataclass
class StudyConfig:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    temp_trade_log: str = "_psar_study_tmp_trades.csv"
    summary_output: str = "parabolic_sar_parameter_study_results.csv"
    delete_temp_trade_log: bool = True
    top_n_to_print: int = 15
    sort_by: str = "score"
    min_trades: int = 20


# =============================================================================
# Columns
# =============================================================================

_PARAM_COLS = [
    "psar_step",
    "psar_max_step",
    "use_ema_trend",
    "ema_trend_period",
    "ema_offset",
    "ema_slope_threshold",
    "use_adx",
    "adx_period",
    "adx_threshold",
    "require_adx_bias",
    "close_signal_adx_limit",
    "atr_period",
]

_METRIC_COLS = [
    "score",
    "total_trades",
    "win_rate",
    "profit_factor",
    "calmar_ratio",
    "max_drawdown_pct",
    "total_return_pct",
    "sharpe_ratio",
    "exit_sl",
    "exit_tp",
    "exit_strategy_close",
    "exit_force_close",
]


# =============================================================================
# Plotting
# =============================================================================

def _plot_results(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping chart.")
        return

    pnl = df["PnL"]
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Parabolic SAR | {symbol} {timeframe.upper()} | "
        f"{start_date.date()} → {end_date.date()}\n"
        f"Win rate: {stats['win_rate'] * 100:.1f}%   "
        f"PF: {stats['profit_factor']:.2f}   "
        f"Return: {stats['total_return_pct']:+.1f}%   "
        f"Max DD: {stats['max_drawdown_pct']:.1f}%   "
        f"Calmar: {stats['calmar_ratio']:.2f}",
        fontsize=11,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[3, 1.2, 1.5],
        hspace=0.45,
    )

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative.values, linewidth=1.4, label="Cumulative PnL")
    ax1.axhline(0, linewidth=0.6, linestyle="--")

    wins_idx = df.index[pnl > 0].tolist()
    losses_idx = df.index[pnl < 0].tolist()

    ax1.scatter(
        wins_idx,
        cumulative.iloc[wins_idx].values,
        s=18,
        zorder=3,
        label="Win",
    )
    ax1.scatter(
        losses_idx,
        cumulative.iloc[losses_idx].values,
        s=18,
        zorder=3,
        label="Loss",
    )

    ax1.set_ylabel("Cumulative PnL")
    ax1.set_xlabel("Trade #")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(
        range(len(drawdown)),
        -drawdown.values,
        0,
        alpha=0.35,
        label="Drawdown",
    )
    ax2.plot(-drawdown.values, linewidth=0.8)
    ax2.axhline(0, linewidth=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    ax3 = fig.add_subplot(gs[2])

    wins_pnl = pnl[pnl > 0].values
    losses_pnl = pnl[pnl <= 0].values

    if len(wins_pnl):
        ax3.hist(wins_pnl, bins=30, alpha=0.65, label=f"Wins ({len(wins_pnl)})")

    if len(losses_pnl):
        ax3.hist(losses_pnl, bins=30, alpha=0.65, label=f"Losses ({len(losses_pnl)})")

    ax3.axvline(0, linewidth=0.8, linestyle="--")
    ax3.set_xlabel("PnL per trade")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)
    ax3.grid(True, linewidth=0.4, alpha=0.5)

    plt.show()


# =============================================================================
# Best outcome printer
# =============================================================================

def _print_best_outcome(best: Dict[str, Any]) -> None:
    sep = "═" * 64
    thin = "─" * 64

    print(f"\n{sep}")
    print("  BEST PARABOLIC SAR PARAMETER SET")
    print(sep)

    print("\n  Indicator / strategy parameters")
    print(thin)

    params = {key: best.get(key) for key in _PARAM_COLS if key in best}

    for key, value in params.items():
        print(f"    {key:<30} {value}")

    print("\n  Performance metrics")
    print(thin)

    fmt_map = {
        "score": lambda value: f"{value:.4f}",
        "win_rate": lambda value: f"{value * 100:.2f}%",
        "profit_factor": lambda value: f"{value:.3f}",
        "calmar_ratio": lambda value: f"{value:.3f}",
        "max_drawdown_pct": lambda value: f"{value:.2f}%",
        "total_return_pct": lambda value: f"{value:+.2f}%",
        "sharpe_ratio": lambda value: f"{value:.4f}",
    }

    metrics = {key: best.get(key) for key in _METRIC_COLS if key in best}

    for key, value in metrics.items():
        formatted = fmt_map[key](value) if key in fmt_map and value is not None else str(value)
        print(f"    {key:<30} {formatted}")

    print(f"\n{sep}")
    print("  COPY THIS INTO strategy_params")
    print(sep)

    print("\nstrategy_params = {")
    for key, value in params.items():
        if value is None:
            print(f'    "{key}": None,')
        elif isinstance(value, str):
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')
    print("}\n")


# =============================================================================
# Parameter cleanup
# =============================================================================

def _strategy_params_for_constructor(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert study params into clean constructor params.

    The summary CSV keeps disabled params as None for readability, but the
    strategy constructor should not receive values like ema_trend_period=None.
    """

    clean = dict(params)

    if clean.get("use_ema_trend") is False:
        clean.pop("ema_trend_period", None)
        clean.pop("ema_offset", None)
        clean.pop("ema_slope_threshold", None)

    if clean.get("use_adx") is False:
        clean.pop("adx_period", None)
        clean.pop("adx_threshold", None)
        clean.pop("require_adx_bias", None)
        clean.pop("close_signal_adx_limit", None)

    return clean


# =============================================================================
# Core backtest runner
# =============================================================================

def run_single_backtest(
    series,
    strategy_params: Dict[str, Any],
    config: BacktestConfig,
    trade_log_path: str,
) -> Dict[str, Any]:
    warmup_series = series.subseries(0, config.cut_off)

    strategy = ParabolicSarStrategy(
        warmup_series,
        use_close_signal=config.use_close_signal,
        **_strategy_params_for_constructor(strategy_params),
    )

    risk_engine = RiskManager()

    balance = config.initial_balance
    active_trade = None
    active_idx = None

    _write_csv_header(trade_log_path)

    for i in range(config.cut_off, len(series._candles)):
        candle = series._candles[i]
        signal = strategy.update(candle)

        # Step 1 — manage open trade first
        if active_trade is not None and i > active_idx:
            exit_result = TradeSimulation.check_exit(active_trade, candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl

                strategy.on_trade_closed(
                    exit_type=exit_type,
                    exit_price=exit_price,
                )

                _append_trade_row(
                    trade_log_path,
                    active_trade,
                    exit_price,
                    exit_type,
                    pnl,
                    balance,
                )

                active_trade = None
                active_idx = None
                continue

            if config.use_close_signal and signal is not None and signal.signal == "CLOSE":
                exit_price = candle.close

                if active_trade.direction.upper() == "BUY":
                    pnl = (exit_price - active_trade.entry) * active_trade.position_size
                else:
                    pnl = (active_trade.entry - exit_price) * active_trade.position_size

                balance += pnl

                strategy.on_trade_closed(
                    exit_type="StrategyClose",
                    exit_price=exit_price,
                )

                _append_trade_row(
                    trade_log_path,
                    active_trade,
                    exit_price,
                    "StrategyClose",
                    pnl,
                    balance,
                )

                active_trade = None
                active_idx = None
                continue

            continue

        # Step 2 — open new trade when flat
        if (
            active_trade is None
            and signal is not None
            and signal.signal in ("BUY", "SELL")
        ):
            active_trade = risk_engine.build_trade(signal, balance)
            active_idx = i
            strategy.sync_trade(active_trade)

    # Force-close any trade still open at end of data
    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close

        if active_trade.direction.upper() == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size

        balance += pnl

        strategy.on_trade_closed(
            exit_type="ForceClose",
            exit_price=exit_price,
        )

        _append_trade_row(
            trade_log_path,
            active_trade,
            exit_price,
            "ForceClose",
            pnl,
            balance,
        )

    df = pd.read_csv(trade_log_path)

    if df.empty:
        stats = _empty_stats()
    else:
        df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
        stats = _compute_stats(df, config.initial_balance)

    result_row = {
        **strategy_params,
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "start_date": config.start_date.date(),
        "end_date": config.end_date.date(),
        "cut_off": config.cut_off,
        "initial_balance": config.initial_balance,
        "use_close_signal": config.use_close_signal,
        "final_balance": balance,
        **stats,
    }

    exits = stats.get("exits_by_type", {})
    result_row["exit_sl"] = exits.get("SL", 0)
    result_row["exit_tp"] = exits.get("TP", 0)
    result_row["exit_strategy_close"] = exits.get("StrategyClose", 0)
    result_row["exit_force_close"] = exits.get("ForceClose", 0)

    directions = stats.get("exits_by_dir", {})
    result_row["buy_trades"] = directions.get("BUY", 0)
    result_row["sell_trades"] = directions.get("SELL", 0)

    return result_row


# =============================================================================
# Parameter-grid builders
# =============================================================================

def build_grid(param_ranges: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """
    Basic Cartesian grid.
    Use this only when every parameter matters in every combination.
    """

    keys = list(param_ranges.keys())
    values = [param_ranges[key] for key in keys]

    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]


def _deduplicate_param_sets(param_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate dictionaries while preserving order.
    """

    seen = set()
    unique: List[Dict[str, Any]] = []

    for params in param_sets:
        key = tuple(sorted(params.items()))

        if key not in seen:
            seen.add(key)
            unique.append(params)

    return unique


def build_psar_strategy_grid(
    psar_ranges: Dict[str, Sequence[Any]],
    atr_ranges: Dict[str, Sequence[Any]],
    ema_ranges: Dict[str, Sequence[Any]],
    adx_ranges: Dict[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    """
    Toggle-aware parameter grid for ParabolicSarStrategy.

    Avoids duplicate tests:
    - EMA params are expanded only when use_ema_trend=True.
    - ADX params are expanded only when use_adx=True.

    When a filter is disabled, its related params are stored as None in the
    results CSV so it is clear that they were ignored.
    """

    base_sets = build_grid({
        **psar_ranges,
        **atr_ranges,
    })

    ema_enabled_sets = build_grid({
        "use_ema_trend": [True],
        **ema_ranges,
    })

    ema_disabled_sets = [{
        "use_ema_trend": False,
        "ema_trend_period": None,
        "ema_offset": None,
        "ema_slope_threshold": None,
    }]

    ema_sets = ema_enabled_sets + ema_disabled_sets

    adx_enabled_sets = build_grid({
        "use_adx": [True],
        **adx_ranges,
    })

    adx_disabled_sets = [{
        "use_adx": False,
        "adx_period": None,
        "adx_threshold": None,
        "require_adx_bias": None,
        "close_signal_adx_limit": None,
    }]

    adx_sets = adx_enabled_sets + adx_disabled_sets

    param_sets: List[Dict[str, Any]] = []

    for base in base_sets:
        for ema in ema_sets:
            for adx in adx_sets:
                param_sets.append({
                    **base,
                    **ema,
                    **adx,
                })

    return _deduplicate_param_sets(param_sets)


# =============================================================================
# Parameter study runner
# =============================================================================

def run_parameter_study(
    param_sets: Iterable[Dict[str, Any]],
    study_config: StudyConfig,
) -> pd.DataFrame:
    provider = MT5MarketDataProvider()
    bt = study_config.backtest

    print("\n  Parabolic SAR Parameter Study")
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

    if total_candles <= bt.cut_off:
        raise ValueError(
            f"Not enough candles. Loaded {total_candles}, cut_off={bt.cut_off}."
        )

    print(
        f"  Loaded {total_candles:,} candles "
        f"({total_candles - bt.cut_off:,} tradeable after warm-up)\n"
    )

    temp_log = Path(study_config.temp_trade_log)
    results: List[Dict[str, Any]] = []

    param_sets = list(param_sets)
    total_runs = len(param_sets)

    print(f"  Total runs: {total_runs}\n")

    for idx, params in enumerate(param_sets, start=1):
        ema_tag = (
            f"EMA:{params.get('ema_trend_period')}/slope={params.get('ema_slope_threshold')}"
            if params.get("use_ema_trend")
            else "EMA:OFF"
        )

        adx_tag = (
            f"ADX:{params.get('adx_threshold')}/bias={params.get('require_adx_bias')}/close={params.get('close_signal_adx_limit')}"
            if params.get("use_adx")
            else "ADX:OFF"
        )

        tag = (
            f"PSAR={params.get('psar_step')}/{params.get('psar_max_step')} "
            f"{ema_tag} "
            f"{adx_tag} "
            f"ATR={params.get('atr_period')}"
        )

        print(f"  [{idx:>4}/{total_runs}] {tag}")

        try:
            row = run_single_backtest(
                series=series,
                strategy_params=params,
                config=bt,
                trade_log_path=str(temp_log),
            )
            row["score"] = _score_result_row(
                row,
                min_trades=study_config.min_trades,
            )
            row["error"] = ""

        except Exception as exc:
            row = {
                **params,
                "symbol": bt.symbol,
                "timeframe": bt.timeframe,
                "start_date": bt.start_date.date(),
                "end_date": bt.end_date.date(),
                "cut_off": bt.cut_off,
                "initial_balance": bt.initial_balance,
                "use_close_signal": bt.use_close_signal,
                **_empty_stats(),
                "final_balance": bt.initial_balance,
                "exit_sl": 0,
                "exit_tp": 0,
                "exit_strategy_close": 0,
                "exit_force_close": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "score": -99999.0,
                "error": str(exc),
            }

        results.append(row)

        if study_config.delete_temp_trade_log and temp_log.exists():
            temp_log.unlink(missing_ok=True)

        # Save after every completed test so progress is never lost.
        pd.DataFrame(results).to_csv(study_config.summary_output, index=False)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        results_df.to_csv(study_config.summary_output, index=False)
        print("  No results generated.")
        return results_df

    sort_col = (
        study_config.sort_by
        if study_config.sort_by in results_df.columns
        else "score"
    )

    results_df = results_df.sort_values(
        by=[
            sort_col,
            "profit_factor",
            "calmar_ratio",
            "total_return_pct",
            "win_rate",
        ],
        ascending=False,
    )

    results_df.to_csv(study_config.summary_output, index=False)

    if study_config.min_trades > 0:
        display_df = results_df[results_df["total_trades"] >= study_config.min_trades]
    else:
        display_df = results_df

    print(f"\n  {'─' * 64}")
    print(
        f"  Top {study_config.top_n_to_print} results "
        f"(min {study_config.min_trades} trades)"
    )
    print(f"  {'─' * 64}\n")

    display_cols = [
        col for col in [
            "score",
            "psar_step",
            "psar_max_step",
            "use_ema_trend",
            "ema_trend_period",
            "ema_slope_threshold",
            "use_adx",
            "adx_threshold",
            "require_adx_bias",
            "close_signal_adx_limit",
            "atr_period",
            "total_trades",
            "win_rate",
            "profit_factor",
            "calmar_ratio",
            "max_drawdown_pct",
            "total_return_pct",
            "sharpe_ratio",
            "exit_sl",
            "exit_tp",
            "exit_strategy_close",
        ]
        if col in display_df.columns
    ]

    preview = display_df.head(study_config.top_n_to_print)

    if preview.empty:
        print("  No runs met the minimum trade threshold.")
    else:
        print(preview[display_cols].to_string(index=False))

    print(f"\n  Full results saved to: {study_config.summary_output}")

    if not display_df.empty:
        best = display_df.iloc[0].to_dict()
        _print_best_outcome(best)

    return results_df


# =============================================================================
# Single-run convenience
# =============================================================================

def backtest_parabolic_sar(
    strategy_params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    strategy_params = strategy_params or preset_default_params()
    config = config or BacktestConfig()

    provider = MT5MarketDataProvider()

    series = provider.fetch_range(
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

    print("\nParabolic SAR Backtest")
    print("=" * 56)
    print(f"Final balance:     {row['final_balance']:.2f}")
    print(f"Total trades:      {row['total_trades']}")
    print(f"Win rate:          {row['win_rate'] * 100:.2f}%")
    print(f"Profit factor:     {row['profit_factor']:.3f}")
    print(f"Calmar ratio:      {row['calmar_ratio']:.3f}")
    print(f"Total return:      {row['total_return_pct']:+.2f}%")
    print(f"Max drawdown %:    {row['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio:      {row['sharpe_ratio']:.4f}")
    print(
        f"Exit SL/TP/SC/FC:  {row['exit_sl']} / {row['exit_tp']} / "
        f"{row['exit_strategy_close']} / {row['exit_force_close']}"
    )
    print("=" * 56)

    if config.plot_results and not df.empty:
        df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
        stats = _compute_stats(df, config.initial_balance)

        _plot_results(
            df=df,
            stats=stats,
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date,
        )

    return df


# =============================================================================
# Presets
# =============================================================================

def preset_default_params() -> Dict[str, Any]:
    """
    Current PSAR-focused default.

    Based on your PSAR-only study, 0.025 / 0.10 is the strongest candidate
    for Gold 5m, while ATR remains the risk-sizing component.
    """

    return {
        "psar_step": 0.03,
        "psar_max_step": 0.15,

        "use_ema_trend": True,
        "ema_trend_period": None,
        "ema_offset": None,
        "ema_slope_threshold": None,

        "use_adx": True,
        "adx_period": None,
        "adx_threshold": 25,
        "require_adx_bias": None,
        "close_signal_adx_limit": None,

        "atr_period": 10,
    }


def preset_full_grid() -> List[Dict[str, Any]]:
    """
    Toggle-aware full grid.

    Old duplicate-heavy count:
    4 × 3 × 2 × 2 × 1 × 2 × 2 × 1 × 3 × 2 × 3 × 2 = 6,912

    New toggle-aware count:
    PSAR/ATR base = 4 × 3 × 2 = 24
    EMA branch = enabled(2 × 1 × 2) + disabled(1) = 5
    ADX branch = enabled(1 × 3 × 2 × 3) + disabled(1) = 19
    Total = 24 × 5 × 19 = 2,280
    """

    return build_psar_strategy_grid(
        psar_ranges={
            "psar_step": [0.01, 0.015, 0.02, 0.025, 0.03],
            "psar_max_step": [0.05, 0.10, 0.15, 0.20],
        },
        atr_ranges={
            "atr_period": [14],
        },
        ema_ranges={
            "ema_trend_period": [20, 50, 100, 200],
            "ema_offset": [1, 3],
            "ema_slope_threshold": [0.0, 0.02, 0.04],
        },
        adx_ranges={
            "adx_period": [14],
            "adx_threshold": [18.0, 20.0, 25.0],
            "require_adx_bias": [False, True],
            "close_signal_adx_limit": [18.0, 20.0, 22.0], #

        },
    )


def preset_psar_focus() -> List[Dict[str, Any]]:
    """
    PSAR-only focus.

    EMA and ADX are disabled, so no EMA/ADX duplicates are generated.
    """

    fixed = preset_default_params()
    param_sets: List[Dict[str, Any]] = []

    for params in build_grid({
        "psar_step": [0.01, 0.015, 0.02, 0.025, 0.03],
        "psar_max_step": [0.05, 0.10, 0.15, 0.20],
    }):
        if params["psar_step"] <= params["psar_max_step"]:
            param_sets.append({
                **fixed,
                **params,
            })

    return _deduplicate_param_sets(param_sets)


def preset_filter_focus() -> List[Dict[str, Any]]:
    """
    Toggle-aware filter study.

    Keeps PSAR and ATR fixed, then tests EMA and ADX branches without
    meaningless duplicate combinations.
    """

    fixed = preset_default_params()

    return build_psar_strategy_grid(
        psar_ranges={
            "psar_step": [fixed["psar_step"]],
            "psar_max_step": [fixed["psar_max_step"]],
        },
        atr_ranges={
            "atr_period": [fixed["atr_period"]],
        },
        ema_ranges={
            "ema_trend_period": [20, 50, 100, 200],
            "ema_offset": [1, 3],
            "ema_slope_threshold": [0.0, 0.02, 0.05],
        },
        adx_ranges={
            "adx_period": [14],
            "adx_threshold": [18.0, 20.0, 25.0, 30.0],
            "require_adx_bias": [False, True],
            # "close_signal_adx_limit": [18.0, 20.0, 22.0],
        },
    )


def preset_exit_focus() -> List[Dict[str, Any]]:
    """
    Exit-focus study.

    close_signal_adx_limit only matters when ADX is enabled, so this preset
    forces use_adx=True.
    """

    fixed = preset_default_params()

    ema_branch = (
        {
            "use_ema_trend": fixed["use_ema_trend"],
            "ema_trend_period": fixed["ema_trend_period"],
            "ema_offset": fixed["ema_offset"],
            "ema_slope_threshold": fixed["ema_slope_threshold"],
        }
    )

    return [
        _normalize_toggle_params({
            **ema_branch,
            **params,
        })
        for params in build_grid({
            "psar_step": [0.01, 0.015, 0.02, 0.025],
            "psar_max_step": [0.10, 0.15, 0.20],

            "use_adx": [True],
            "adx_period": [14],
            "adx_threshold": [18.0, 20.0, 25.0],
            "require_adx_bias": [False, True],
            "close_signal_adx_limit": [15.0, 18.0, 20.0, 22.0, 25.0, 30.0],

            "atr_period": [fixed["atr_period"]],
        })
    ]


def _normalize_toggle_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize disabled branches to None.

    Useful for custom presets that are not built through build_psar_strategy_grid().
    """

    normalized = dict(params)

    if normalized.get("use_ema_trend") is False:
        normalized["ema_trend_period"] = None
        normalized["ema_offset"] = None
        normalized["ema_slope_threshold"] = None

    if normalized.get("use_adx") is False:
        normalized["adx_period"] = None
        normalized["adx_threshold"] = None
        normalized["require_adx_bias"] = None
        normalized["close_signal_adx_limit"] = None

    return normalized


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    MODE = "study_full"
    # Options:
    # "backtest"
    # "study_full"
    # "study_psar"
    # "study_filters"
    # "study_exit"

    backtest_config = BacktestConfig(
        symbol=SYMBOLS["Gold"],
        timeframe="5m",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2026, 6, 30),
        cut_off=300,
        initial_balance=200.0,
        use_close_signal=True,
        output_file="trades_parabolic_sar.csv",
        plot_results=False,
    )

    if MODE == "backtest":
        backtest_parabolic_sar(
            strategy_params=preset_default_params(),
            config=backtest_config,
        )

    else:
        param_sets = {
            "study_full": preset_full_grid,
            "study_psar": preset_psar_focus,
            "study_filters": preset_filter_focus,
            "study_exit": preset_exit_focus,
        }[MODE]()

        study = StudyConfig(
            backtest=backtest_config,
            temp_trade_log="_psar_study_tmp_trades.csv",
            summary_output=f"parabolic_sar_study_{MODE}.csv",
            delete_temp_trade_log=True,
            top_n_to_print=15,
            sort_by="score",
            min_trades=20,
        )

        run_parameter_study(
            param_sets=param_sets,
            study_config=study,
        )