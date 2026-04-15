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
    @staticmethod
    def check_exit(trade, candle) -> Optional[Tuple]:
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

    max_dd = drawdown.max() if not drawdown.empty else 0.0
    peak_at_dd = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else 0.0
    denom = initial_balance + peak_at_dd
    max_dd_pct = max_dd / denom * 100 if denom > 0 else 0.0

    std_pnl = pnl.std()
    sharpe = pnl.mean() / std_pnl * math.sqrt(len(pnl)) if std_pnl and std_pnl > 0 else 0.0

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


def _score_result_row(row: Dict[str, Any], min_trades: int = 20) -> float:
    trades = float(row.get("total_trades", 0) or 0)
    pf = float(row.get("profit_factor", 0) or 0)
    ret = float(row.get("total_return_pct", 0) or 0)
    dd_pct = float(row.get("max_drawdown_pct", 0) or 0)
    wr = float(row.get("win_rate", 0) or 0)
    sharpe = float(row.get("sharpe_ratio", 0) or 0)
    calmar = float(row.get("calmar_ratio", 0) or 0)

    pf = min(pf, 8.0) if not math.isinf(pf) else 8.0

    if trades >= min_trades:
        trade_penalty = 0.0
    elif trades >= min_trades / 2:
        trade_penalty = (min_trades - trades) * 1.0
    else:
        trade_penalty = (min_trades - trades) * 3.0

    if ret <= 0:
        return round(-9999.0 - trade_penalty, 4)

    score = (
        pf * 30.0
        + calmar * 15.0
        + wr * 80.0
        + sharpe * 12.0
        + ret * 0.8
        - dd_pct * 2.5
        - trade_penalty
    )

    return round(score, 4)


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


def _print_best_outcome(best: Dict[str, Any]) -> None:
    sep = "═" * 64
    thin = "─" * 64

    print(f"\n{sep}")
    print("  BEST PARABOLIC SAR PARAMETER SET")
    print(sep)

    print("\n  Indicator / strategy parameters")
    print(thin)

    params = {k: best.get(k) for k in _PARAM_COLS if k in best}
    for k, v in params.items():
        print(f"    {k:<30} {v}")

    print("\n  Performance metrics")
    print(thin)

    fmt_map = {
        "score": lambda v: f"{v:.4f}",
        "win_rate": lambda v: f"{v * 100:.2f}%",
        "profit_factor": lambda v: f"{v:.3f}",
        "calmar_ratio": lambda v: f"{v:.3f}",
        "max_drawdown_pct": lambda v: f"{v:.2f}%",
        "total_return_pct": lambda v: f"{v:+.2f}%",
        "sharpe_ratio": lambda v: f"{v:.4f}",
    }

    metrics = {k: best.get(k) for k in _METRIC_COLS if k in best}
    for k, v in metrics.items():
        formatted = fmt_map[k](v) if k in fmt_map and v is not None else str(v)
        print(f"    {k:<30} {formatted}")

    print(f"\n{sep}")
    print("  COPY THIS INTO strategy_params")
    print(sep)
    print("\nstrategy_params = {")
    for k, v in params.items():
        if isinstance(v, str):
            print(f'    "{k}": "{v}",')
        else:
            print(f'    "{k}": {v},')
    print("}\n")


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
        **strategy_params,
    )

    risk_engine = RiskManager()
    balance = config.initial_balance
    active_trade = None
    active_idx = None

    _write_csv_header(trade_log_path)

    for i in range(config.cut_off, len(series._candles)):
        candle = series._candles[i]
        signal = strategy.update(candle)

        if active_trade is not None and i > active_idx:
            exit_result = TradeSimulation.check_exit(active_trade, candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl
                strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                _append_trade_row(trade_log_path, active_trade, exit_price, exit_type, pnl, balance)

                active_trade = None
                active_idx = None
                continue

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
                active_idx = None
                continue

            continue

        if active_trade is None and signal is not None and signal.signal in ("BUY", "SELL"):
            active_trade = risk_engine.build_trade(signal, balance)
            active_idx = i
            strategy.sync_trade(active_trade)

    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close
        pnl = (
            (exit_price - active_trade.entry) * active_trade.position_size
            if active_trade.direction.upper() == "BUY"
            else (active_trade.entry - exit_price) * active_trade.position_size
        )

        balance += pnl
        strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)
        _append_trade_row(trade_log_path, active_trade, exit_price, "ForceClose", pnl, balance)

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

    dirs = stats.get("exits_by_dir", {})
    result_row["buy_trades"] = dirs.get("BUY", 0)
    result_row["sell_trades"] = dirs.get("SELL", 0)

    return result_row


def build_grid(param_ranges: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


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

    print(f"  Loaded {total_candles:,} candles "
          f"({total_candles - bt.cut_off:,} tradeable after warm-up)\n")

    temp_log = Path(study_config.temp_trade_log)
    results: List[Dict[str, Any]] = []
    param_sets = list(param_sets)
    total_runs = len(param_sets)

    print(f"  Total runs: {total_runs}\n")

    for idx, params in enumerate(param_sets, start=1):
        tag = (
            f"psar={params.get('psar_step')}/{params.get('psar_max_step')} "
            f"ema={params.get('use_ema_trend')}:{params.get('ema_trend_period')} "
            f"adx={params.get('use_adx')}:{params.get('adx_threshold')} "
            f"closeADX={params.get('close_signal_adx_limit')}"
        )

        print(f"  [{idx:>4}/{total_runs}] {tag}")

        try:
            row = run_single_backtest(
                series=series,
                strategy_params=params,
                config=bt,
                trade_log_path=str(temp_log),
            )
            row["score"] = _score_result_row(row, min_trades=study_config.min_trades)
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

        # Log all completed outcomes after every test
        pd.DataFrame(results).to_csv(study_config.summary_output, index=False)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        results_df.to_csv(study_config.summary_output, index=False)
        print("  No results generated.")
        return results_df

    sort_col = study_config.sort_by if study_config.sort_by in results_df.columns else "score"

    results_df = results_df.sort_values(
        by=[sort_col, "profit_factor", "calmar_ratio", "total_return_pct", "win_rate"],
        ascending=False,
    )

    results_df.to_csv(study_config.summary_output, index=False)

    display_df = (
        results_df[results_df["total_trades"] >= study_config.min_trades]
        if study_config.min_trades > 0
        else results_df
    )

    print(f"\n  {'─' * 64}")
    print(f"  Top {study_config.top_n_to_print} results "
          f"(min {study_config.min_trades} trades)")
    print(f"  {'─' * 64}\n")

    display_cols = [
        c for c in [
            "score",
            "psar_step",
            "psar_max_step",
            "use_ema_trend",
            "ema_trend_period",
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
        ] if c in display_df.columns
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


def preset_default_params() -> Dict[str, Any]:
    return {
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


def preset_full_grid() -> List[Dict[str, Any]]:
    return build_grid({
        "psar_step": [0.01, 0.015, 0.02, 0.025],
        "psar_max_step": [0.10, 0.15, 0.20],

        "use_ema_trend": [True, False],
        "ema_trend_period": [100, 200],
        "ema_offset": [3],
        "ema_slope_threshold": [0.0, 0.02],

        "use_adx": [True, False],
        "adx_period": [14],
        "adx_threshold": [18.0, 20.0, 25.0],
        "require_adx_bias": [False, True],

        "close_signal_adx_limit": [18.0, 20.0, 22.0],

        "atr_period": [10, 14],
    })


def preset_psar_focus() -> List[Dict[str, Any]]:
    fixed = preset_default_params()
    param_sets = []

    for params in build_grid({
        "psar_step": [0.01, 0.02, 0.03, 0.04, 0.05],
        "psar_max_step": [0.10, 0.15, 0.20, 0.25],
    }):
        if params["psar_step"] <= params["psar_max_step"]:
            param_sets.append({**fixed, **params})

    return param_sets


def preset_filter_focus() -> List[Dict[str, Any]]:
    fixed = preset_default_params()

    return [
        {**fixed, **params}
        for params in build_grid({
            "use_ema_trend": [True, False],
            "ema_trend_period": [50, 100, 200],
            "ema_slope_threshold": [0.0, 0.02, 0.05],
            "use_adx": [True, False],
            "adx_threshold": [18.0, 20.0, 25.0, 30.0],
            "require_adx_bias": [False, True],
        })
    ]


def preset_exit_focus() -> List[Dict[str, Any]]:
    fixed = preset_default_params()

    return [
        {**fixed, **params}
        for params in build_grid({
            "close_signal_adx_limit": [15.0, 18.0, 20.0, 22.0, 25.0, 30.0],
            "psar_step": [0.01, 0.015, 0.02],
            "psar_max_step": [0.10, 0.15, 0.20],
        })
    ]


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
    print(f"Exit SL/TP/SC/FC:  {row['exit_sl']} / {row['exit_tp']} / "
          f"{row['exit_strategy_close']} / {row['exit_force_close']}")
    print("=" * 56)

    return df


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
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
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