from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.strategy.parabolic_sar_strategy_v1 import ParabolicSarStrategy


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
    "BTC": "BTCUSDm",
    "ETH": "ETHUSDm",
}


# =============================================================================
# Trade object
# =============================================================================

@dataclass
class BacktestTrade:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    stop_distance: float
    rr: float
    candle: object
    pattern_name: str
    reason: str
    sl_source: str
    psar_sl: Optional[float]
    atr_sl: Optional[float]


# =============================================================================
# Risk manager for study
# =============================================================================

class StudyRiskManager:
    """
    Backtest-only risk manager.

    Rules:
    - Risk 1% of current balance by default.
    - RR = 2 by default.
    - ATR stop = 1.5 * ATR from candle low/high by default.
    - PSAR stop = raw PSAR dot supplied by strategy signal.
    - Final SL is selected by sl_mode.

    sl_mode:
    - WIDER   -> choose farther stop between PSAR and ATR.
    - TIGHTER -> choose closer stop between PSAR and ATR.
    - PSAR    -> use PSAR stop only.
    - ATR     -> use ATR stop only.
    """

    VALID_SL_MODES = {"WIDER", "TIGHTER", "PSAR", "ATR"}

    def __init__(
        self,
        risk_percent: float = 0.01,
        rr: float = 2.0,
        atr_multiplier: float = 1.5,
        sl_mode: str = "WIDER",
    ):
        if risk_percent <= 0:
            raise ValueError("risk_percent must be greater than 0")

        if rr <= 0:
            raise ValueError("rr must be greater than 0")

        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be greater than 0")

        sl_mode = sl_mode.upper()
        if sl_mode not in self.VALID_SL_MODES:
            raise ValueError(f"Invalid sl_mode '{sl_mode}'. Use {self.VALID_SL_MODES}")

        self.risk_percent = risk_percent
        self.rr = rr
        self.atr_multiplier = atr_multiplier
        self.sl_mode = sl_mode

    def build_trade(self, signal, balance: float) -> Optional[BacktestTrade]:
        direction = signal.signal.upper()

        if direction not in ("BUY", "SELL"):
            return None

        candle = signal.candle
        entry = candle.close
        atr = signal.atr
        psar_sl = signal.sl

        if atr is None or atr <= 0:
            return None

        atr_sl = self._atr_stop(direction, candle, atr)
        final_sl, sl_source = self._select_stop(
            direction=direction,
            entry=entry,
            psar_sl=psar_sl,
            atr_sl=atr_sl,
        )

        if final_sl is None:
            return None

        stop_distance = abs(entry - final_sl)

        if stop_distance <= 0:
            return None

        risk_amount = balance * self.risk_percent
        position_size = risk_amount / stop_distance

        if direction == "BUY":
            take_profit = entry + (stop_distance * self.rr)
        else:
            take_profit = entry - (stop_distance * self.rr)

        return BacktestTrade(
            direction=direction,
            entry=entry,
            stop_loss=final_sl,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            rr=self.rr,
            candle=candle,
            pattern_name=signal.pattern_name,
            reason=signal.reason,
            sl_source=sl_source,
            psar_sl=psar_sl,
            atr_sl=atr_sl,
        )

    def _atr_stop(self, direction: str, candle, atr: float) -> float:
        if direction == "BUY":
            return candle.low - (self.atr_multiplier * atr)

        return candle.high + (self.atr_multiplier * atr)

    def _select_stop(
        self,
        direction: str,
        entry: float,
        psar_sl: Optional[float],
        atr_sl: Optional[float],
    ) -> Tuple[Optional[float], str]:
        candidates: List[Tuple[str, float]] = []

        if psar_sl is not None and self._valid_stop(direction, entry, psar_sl):
            candidates.append(("PSAR", float(psar_sl)))

        if atr_sl is not None and self._valid_stop(direction, entry, atr_sl):
            candidates.append(("ATR", float(atr_sl)))

        if not candidates:
            return None, "NONE"

        if self.sl_mode == "PSAR":
            for source, value in candidates:
                if source == "PSAR":
                    return value, source
            return None, "NONE"

        if self.sl_mode == "ATR":
            for source, value in candidates:
                if source == "ATR":
                    return value, source
            return None, "NONE"

        if direction == "BUY":
            if self.sl_mode == "WIDER":
                source, value = min(candidates, key=lambda item: item[1])
            else:
                source, value = max(candidates, key=lambda item: item[1])
        else:
            if self.sl_mode == "WIDER":
                source, value = max(candidates, key=lambda item: item[1])
            else:
                source, value = min(candidates, key=lambda item: item[1])

        return value, source

    def _valid_stop(self, direction: str, entry: float, stop: float) -> bool:
        if direction == "BUY":
            return stop < entry

        return stop > entry


# =============================================================================
# Exit simulation
# =============================================================================

class TradeSimulation:
    """
    Candle-level SL / TP exit checker.

    Conservative rule:
    If both SL and TP are touched in the same candle, SL is assumed first.
    """

    @staticmethod
    def check_exit(trade: BacktestTrade, candle) -> Optional[Tuple[float, str, float]]:
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        if direction == "BUY":
            if candle.low <= sl:
                return sl, "SL", (sl - entry) * size

            if candle.high >= tp:
                return tp, "TP", (tp - entry) * size

        else:
            if candle.high >= sl:
                return sl, "SL", (entry - sl) * size

            if candle.low <= tp:
                return tp, "TP", (entry - tp) * size

        return None


# =============================================================================
# CSV columns
# =============================================================================

_TRADE_HEADER = [
    "EntryTime",
    "Direction",
    "Entry",
    "SL",
    "TP",
    "PositionSize",
    "RiskAmount",
    "StopDistance",
    "RR",
    "SLSource",
    "PSAR_SL",
    "ATR_SL",
    "ExitPrice",
    "ExitType",
    "PnL",
    "BalanceAfter",
    "Pattern",
    "Reason",
]

_PARAM_COLS = [
    "psar_step",
    "psar_max_step",
    "atr_period",
    "use_ema_trend",
    "ema_trend_period",
    "ema_offset",
    "ema_slope_threshold",
    "use_adx",
    "adx_period",
    "adx_threshold",
    "require_adx_bias",
]

_METRIC_COLS = [
    "score",
    "total_trades",
    "wins",
    "losses",
    "win_rate",
    "profit_factor",
    "total_pnl",
    "total_return_pct",
    "max_drawdown",
    "max_drawdown_pct",
    "sharpe_ratio",
    "calmar_ratio",
    "exit_sl",
    "exit_tp",
    "exit_strategy_close",
    "exit_force_close",
    "buy_trades",
    "sell_trades",
    "sl_source_psar",
    "sl_source_atr",
]


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = SYMBOLS["Gold"]
    timeframe: str = "1m"
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2026, 3, 31))
    cut_off: int = 500
    initial_balance: float = 200.0
    use_close_signal: bool = True

    risk_percent: float = 0.01
    rr: float = 2.0
    atr_multiplier: float = 1.5
    sl_mode: str = "WIDER"


@dataclass
class StudyConfig:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    temp_trade_log: str = "_psar_corrected_study_tmp_trades.csv"
    summary_output: str = "parabolic_sar_corrected_parameter_study_results.csv"
    best_output: str = "parabolic_sar_corrected_best_candidates.csv"
    top_by_category_output: str = "parabolic_sar_corrected_top_by_category.csv"
    delete_temp_trade_log: bool = True
    top_n_to_print: int = 20
    top_n_per_category: int = 5
    sort_by: str = "score"
    min_trades: int = 80


# =============================================================================
# CSV helpers
# =============================================================================

def _write_trade_header(path: str) -> None:
    with open(path, "w", newline="") as file:
        csv.writer(file).writerow(_TRADE_HEADER)


def _append_trade_row(
    path: str,
    trade: BacktestTrade,
    exit_price: float,
    exit_type: str,
    pnl: float,
    balance_after: float,
) -> None:
    with open(path, "a", newline="") as file:
        csv.writer(file).writerow([
            trade.candle.time,
            trade.direction,
            f"{trade.entry:.5f}",
            f"{trade.stop_loss:.5f}",
            f"{trade.take_profit:.5f}",
            f"{trade.position_size:.6f}",
            f"{trade.risk_amount:.4f}",
            f"{trade.stop_distance:.5f}",
            f"{trade.rr:.2f}",
            trade.sl_source,
            f"{trade.psar_sl:.5f}" if trade.psar_sl is not None else "",
            f"{trade.atr_sl:.5f}" if trade.atr_sl is not None else "",
            f"{exit_price:.5f}",
            exit_type,
            f"{pnl:.4f}",
            f"{balance_after:.4f}",
            trade.pattern_name,
            trade.reason,
        ])


# =============================================================================
# Stats
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
        "sl_sources": {},
    }


def _compute_stats(df: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    if df.empty:
        return _empty_stats()

    pnl = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)

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

    cumulative_pnl = pnl.cumsum()
    equity = initial_balance + cumulative_pnl
    running_peak = equity.cummax()
    drawdown = running_peak - equity

    max_drawdown = drawdown.max() if not drawdown.empty else 0.0
    peak_at_dd = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else initial_balance

    max_drawdown_pct = (
        max_drawdown / peak_at_dd * 100.0
        if peak_at_dd and peak_at_dd > 0
        else 0.0
    )

    total_return_pct = pnl.sum() / initial_balance * 100.0 if initial_balance else 0.0

    std_pnl = pnl.std()
    sharpe_ratio = (
        pnl.mean() / std_pnl * math.sqrt(len(pnl))
        if std_pnl and std_pnl > 0
        else 0.0
    )

    calmar_ratio = (
        total_return_pct / max_drawdown_pct
        if max_drawdown_pct > 0
        else 0.0
    )

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
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "exits_by_type": df["ExitType"].value_counts().to_dict(),
        "exits_by_dir": df["Direction"].value_counts().to_dict(),
        "sl_sources": df["SLSource"].value_counts().to_dict(),
    }


def _score_result_row(row: Dict[str, Any], min_trades: int = 80) -> float:
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
        trade_penalty = (min_trades - trades) * 1.5
    else:
        trade_penalty = (min_trades - trades) * 4.0

    if total_return_pct <= 0:
        return round(-9999.0 - trade_penalty, 4)

    score = (
        profit_factor * 35.0
        + calmar_ratio * 18.0
        + win_rate * 80.0
        + sharpe_ratio * 14.0
        + total_return_pct * 0.7
        - max_drawdown_pct * 3.0
        - trade_penalty
    )

    return round(score, 4)


# =============================================================================
# Param-grid builders
# =============================================================================

def build_grid(param_ranges: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_ranges.keys())
    values = [param_ranges[key] for key in keys]

    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]


def _deduplicate_param_sets(param_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    Toggle-aware grid.

    Avoids duplicates:
    - EMA params are expanded only when use_ema_trend=True.
    - ADX params are expanded only when use_adx=True.
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

    adx_enabled_sets = build_grid({
        "use_adx": [True],
        **adx_ranges,
    })

    adx_disabled_sets = [{
        "use_adx": False,
        "adx_period": None,
        "adx_threshold": None,
        "require_adx_bias": None,
    }]

    param_sets: List[Dict[str, Any]] = []

    for base in base_sets:
        for ema in ema_enabled_sets + ema_disabled_sets:
            for adx in adx_enabled_sets + adx_disabled_sets:
                param_sets.append({
                    **base,
                    **ema,
                    **adx,
                })

    return _deduplicate_param_sets(param_sets)


def _strategy_params_for_constructor(params: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(params)

    if clean.get("use_ema_trend") is False:
        clean.pop("ema_trend_period", None)
        clean.pop("ema_offset", None)
        clean.pop("ema_slope_threshold", None)

    if clean.get("use_adx") is False:
        clean.pop("adx_period", None)
        clean.pop("adx_threshold", None)
        clean.pop("require_adx_bias", None)

    return clean


# =============================================================================
# Backtest runner
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

    risk_engine = StudyRiskManager(
        risk_percent=config.risk_percent,
        rr=config.rr,
        atr_multiplier=config.atr_multiplier,
        sl_mode=config.sl_mode,
    )

    balance = config.initial_balance
    active_trade: Optional[BacktestTrade] = None
    active_idx: Optional[int] = None

    _write_trade_header(trade_log_path)

    for i in range(config.cut_off, len(series._candles)):
        candle = series._candles[i]

        # Signal is based on the close of this candle.
        signal = strategy.update(candle)

        # Manage existing trade first.
        if active_trade is not None and active_idx is not None and i > active_idx:
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

                if active_trade.direction == "BUY":
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

            # Do not open a second trade while one is active.
            continue

        # Open new trade only when flat.
        if (
            active_trade is None
            and signal is not None
            and signal.signal in ("BUY", "SELL")
        ):
            trade = risk_engine.build_trade(signal, balance)

            if trade is None:
                continue

            active_trade = trade
            active_idx = i
            strategy.sync_trade(active_trade)

    # Force-close at end.
    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close

        if active_trade.direction == "BUY":
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

    exits = stats.get("exits_by_type", {})
    directions = stats.get("exits_by_dir", {})
    sl_sources = stats.get("sl_sources", {})

    return {
        **strategy_params,

        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "start_date": config.start_date.date(),
        "end_date": config.end_date.date(),
        "cut_off": config.cut_off,
        "initial_balance": config.initial_balance,
        "final_balance": balance,

        "use_close_signal": config.use_close_signal,
        "risk_percent": config.risk_percent,
        "rr": config.rr,
        "atr_multiplier": config.atr_multiplier,
        "sl_mode": config.sl_mode,

        **stats,

        "exit_sl": exits.get("SL", 0),
        "exit_tp": exits.get("TP", 0),
        "exit_strategy_close": exits.get("StrategyClose", 0),
        "exit_force_close": exits.get("ForceClose", 0),

        "buy_trades": directions.get("BUY", 0),
        "sell_trades": directions.get("SELL", 0),

        "sl_source_psar": sl_sources.get("PSAR", 0),
        "sl_source_atr": sl_sources.get("ATR", 0),
    }


# =============================================================================
# Study analysis helpers
# =============================================================================

def _strategy_category(row: Dict[str, Any]) -> str:
    use_ema = bool(row.get("use_ema_trend"))
    use_adx = bool(row.get("use_adx"))

    if use_ema and use_adx:
        return "EMA_ADX"
    if use_ema and not use_adx:
        return "EMA_ONLY"
    if use_adx and not use_ema:
        return "ADX_ONLY"
    return "PSAR_ONLY"


def _print_best_outcome(best: Dict[str, Any]) -> None:
    sep = "═" * 72
    thin = "─" * 72

    print(f"\n{sep}")
    print("  BEST PARAMETER SET")
    print(sep)

    print("\n  Parameters")
    print(thin)

    for key in _PARAM_COLS:
        if key in best:
            print(f"    {key:<30} {best.get(key)}")

    print("\n  Metrics")
    print(thin)

    for key in _METRIC_COLS:
        if key in best:
            value = best.get(key)

            if key in {"win_rate"}:
                formatted = f"{float(value) * 100:.2f}%"
            elif key in {"profit_factor", "calmar_ratio", "sharpe_ratio"}:
                formatted = f"{float(value):.4f}"
            elif key in {"total_return_pct", "max_drawdown_pct"}:
                formatted = f"{float(value):.2f}%"
            elif isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)

            print(f"    {key:<30} {formatted}")

    print(f"\n{sep}")
    print("  COPY INTO strategy_params")
    print(sep)

    print("\nstrategy_params = {")
    for key in _PARAM_COLS:
        if key not in best:
            continue

        value = best.get(key)

        if value is None or pd.isna(value):
            print(f'    "{key}": None,')
        elif isinstance(value, str):
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')
    print("}\n")


def _write_top_outputs(
    results_df: pd.DataFrame,
    study_config: StudyConfig,
) -> None:
    valid_df = results_df.copy()

    if study_config.min_trades > 0:
        valid_df = valid_df[valid_df["total_trades"] >= study_config.min_trades]

    if valid_df.empty:
        pd.DataFrame().to_csv(study_config.best_output, index=False)
        pd.DataFrame().to_csv(study_config.top_by_category_output, index=False)
        return

    best_cols = [
        col for col in [
            "category",
            *_PARAM_COLS,
            *_METRIC_COLS,
            "risk_percent",
            "rr",
            "atr_multiplier",
            "sl_mode",
        ]
        if col in valid_df.columns
    ]

    valid_df.head(study_config.top_n_to_print).to_csv(
        study_config.best_output,
        columns=best_cols,
        index=False,
    )

    category_frames = []

    for category, group in valid_df.groupby("category"):
        category_frames.append(group.head(study_config.top_n_per_category))

    top_by_category = pd.concat(category_frames, ignore_index=True) if category_frames else pd.DataFrame()

    if not top_by_category.empty:
        top_by_category.to_csv(
            study_config.top_by_category_output,
            columns=best_cols,
            index=False,
        )
    else:
        top_by_category.to_csv(study_config.top_by_category_output, index=False)


# =============================================================================
# Study runner
# =============================================================================

def run_parameter_study(
    param_sets: Iterable[Dict[str, Any]],
    study_config: StudyConfig,
) -> pd.DataFrame:
    bt = study_config.backtest

    print("\n  Parabolic SAR Corrected-Risk Parameter Study")
    print(f"  Symbol: {bt.symbol} | Timeframe: {bt.timeframe}")
    print(f"  Period: {bt.start_date.date()} → {bt.end_date.date()}")
    print(f"  Initial balance: {bt.initial_balance:.2f}")
    print(f"  Risk: {bt.risk_percent * 100:.2f}% | RR: {bt.rr:.2f} | ATR x{bt.atr_multiplier:.2f} | SL mode: {bt.sl_mode}")
    print(f"  Close signal: {bt.use_close_signal} | Min trades: {study_config.min_trades}")
    print(f"  Summary output: {study_config.summary_output}\n")

    provider = MT5MarketDataProvider()

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

    param_sets = list(param_sets)
    total_runs = len(param_sets)

    print(f"  Total runs: {total_runs}\n")

    results: List[Dict[str, Any]] = []
    temp_log = Path(study_config.temp_trade_log)

    for idx, params in enumerate(param_sets, start=1):
        ema_tag = (
            f"EMA:{params.get('ema_trend_period')}/slope={params.get('ema_slope_threshold')}"
            if params.get("use_ema_trend")
            else "EMA:OFF"
        )

        adx_tag = (
            f"ADX:{params.get('adx_period')}/{params.get('adx_threshold')}/bias={params.get('require_adx_bias')}"
            if params.get("use_adx")
            else "ADX:OFF"
        )

        tag = (
            f"PSAR={params.get('psar_step')}/{params.get('psar_max_step')} "
            f"ATR={params.get('atr_period')} "
            f"{ema_tag} "
            f"{adx_tag}"
        )

        print(f"  [{idx:>5}/{total_runs}] {tag}")

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
            row["category"] = _strategy_category(row)
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
                "final_balance": bt.initial_balance,
                "use_close_signal": bt.use_close_signal,
                "risk_percent": bt.risk_percent,
                "rr": bt.rr,
                "atr_multiplier": bt.atr_multiplier,
                "sl_mode": bt.sl_mode,
                **_empty_stats(),
                "exit_sl": 0,
                "exit_tp": 0,
                "exit_strategy_close": 0,
                "exit_force_close": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "sl_source_psar": 0,
                "sl_source_atr": 0,
                "score": -99999.0,
                "category": _strategy_category(params),
                "error": str(exc),
            }

        results.append(row)

        if study_config.delete_temp_trade_log and temp_log.exists():
            temp_log.unlink(missing_ok=True)

        # Save after every completed run.
        pd.DataFrame(results).to_csv(study_config.summary_output, index=False)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        results_df.to_csv(study_config.summary_output, index=False)
        print("  No results generated.")
        return results_df

    sort_col = study_config.sort_by if study_config.sort_by in results_df.columns else "score"

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

    _write_top_outputs(results_df, study_config)

    valid_df = (
        results_df[results_df["total_trades"] >= study_config.min_trades]
        if study_config.min_trades > 0
        else results_df
    )

    print(f"\n  {'─' * 88}")
    print(f"  Top {study_config.top_n_to_print} results | min trades: {study_config.min_trades}")
    print(f"  {'─' * 88}\n")

    display_cols = [
        col for col in [
            "score",
            "category",
            "psar_step",
            "psar_max_step",
            "atr_period",
            "use_ema_trend",
            "ema_trend_period",
            "use_adx",
            "adx_period",
            "adx_threshold",
            "require_adx_bias",
            "total_trades",
            "win_rate",
            "profit_factor",
            "total_return_pct",
            "max_drawdown_pct",
            "exit_sl",
            "exit_tp",
            "exit_strategy_close",
            "sl_source_psar",
            "sl_source_atr",
        ]
        if col in valid_df.columns
    ]

    preview = valid_df.head(study_config.top_n_to_print)

    if preview.empty:
        print("  No runs met the minimum trade threshold.")
    else:
        print(preview[display_cols].to_string(index=False))

    print(f"\n  Full results saved to: {study_config.summary_output}")
    print(f"  Best candidates saved to: {study_config.best_output}")
    print(f"  Top by category saved to: {study_config.top_by_category_output}")

    if not valid_df.empty:
        best = valid_df.iloc[0].to_dict()
        _print_best_outcome(best)

    return results_df


# =============================================================================
# Presets
# =============================================================================

def preset_default_params() -> Dict[str, Any]:
    """
    Current robust candidate based on corrected-risk testing.
    """

    return {
        "psar_step": 0.01,
        "psar_max_step": 0.15,

        "atr_period": 10,

        "use_ema_trend": False,
        "ema_trend_period": None,
        "ema_offset": None,
        "ema_slope_threshold": None,

        "use_adx": True,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "require_adx_bias": True,
    }


def preset_psar_focus() -> List[Dict[str, Any]]:
    """
    PSAR-only sweep.

    Use this first when testing a new symbol/timeframe.
    """

    return build_psar_strategy_grid(
        psar_ranges={
            "psar_step": [0.003, 0.004, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03],
            "psar_max_step": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
        },
        atr_ranges={
            "atr_period": [14],
        },
        ema_ranges={
            "ema_trend_period": [200],
            "ema_offset": [3],
            "ema_slope_threshold": [0.0],
        },
        adx_ranges={
            "adx_period": [],
            "adx_threshold": [],
            "require_adx_bias": [],
        },
    )


def preset_adx_focus() -> List[Dict[str, Any]]:
    """
    ADX-only study around known promising PSAR area.
    """

    fixed_psar_steps = [0.015, 0.02]
    fixed_psar_max = [0.08, 0.10]

    param_sets = []

    for base in build_grid({
        "psar_step": fixed_psar_steps,
        "psar_max_step": fixed_psar_max,
        "atr_period": [7, 10, 14],
    }):
        for adx in build_grid({
            "use_adx": [True],
            "adx_period": [10, 14, 20],
            "adx_threshold": [15.0, 18.0, 20.0, 22.0, 25.0],
            "require_adx_bias": [False, True],
        }):
            param_sets.append({
                **base,
                "use_ema_trend": False,
                "ema_trend_period": None,
                "ema_offset": None,
                "ema_slope_threshold": None,
                **adx,
            })

    return _deduplicate_param_sets(param_sets)


def preset_ema_focus() -> List[Dict[str, Any]]:
    """
    EMA-only study.
    """

    param_sets = []

    for base in build_grid({
        "psar_step": [0.005, 0.025],
        "psar_max_step": [0.05, 0.10],
        "atr_period": [14],
    }):
        for ema in build_grid({
            "use_ema_trend": [True],
            "ema_trend_period": [34, 50, 100, 200],
            "ema_offset": [3, 5],
            "ema_slope_threshold": [0.0],
        }):
            param_sets.append({
                **base,
                **ema,
                "use_adx": False,
                "adx_period": None,
                "adx_threshold": None,
                "require_adx_bias": None,
            })

    return _deduplicate_param_sets(param_sets)


def preset_full_grid() -> List[Dict[str, Any]]:
    """
    Toggle-aware full grid.

    This is larger. Use after psar/adx/ema focus studies.
    """

    return build_psar_strategy_grid(
        psar_ranges={
            "psar_step": [0.005, 0.01, 0.015, 0.020, 0.025],
            "psar_max_step": [0.05, 0.10, 0.15, 0.20],
        },
        atr_ranges={
            "atr_period": [10, 14],
        },
        ema_ranges={
            "ema_trend_period": [34, 50, 75, 100, 200],
            "ema_offset": [3],
            "ema_slope_threshold": [0.0, 0.005, 0.01, 0.02],
        },
        adx_ranges={
            "adx_period": [10, 14],
            "adx_threshold": [20.0, 25.0],
            "require_adx_bias": [False, True],
        },
    )


def preset_final_focused_grid() -> List[Dict[str, Any]]:
    """
    Focused combined study after PSAR and ADX studies.

    Goal:
    Test whether EMA improves the already-strong PSAR + ADX + ATR family.

    Based on current winners:
    - PSAR: 0.02 with max 0.08 / 0.10
    - ADX: period 10/14, threshold 15/18
    - ATR: still test 7/10/14
    """

    param_sets = []

    for params in build_grid({
        # Best PSAR area from PSAR + ADX studies
        "psar_step": [0.005, 0.01, 0.015, 0.02, 0.025],
        "psar_max_step": [0.05, 0.10, 0.15, 0.20, 0.25],

        # Keep ATR open for now
        "atr_period": [14],

        # Now force EMA ON because this stage tests EMA + ADX together
        "use_ema_trend": [True],
        "ema_trend_period": [50, 100, 200],
        "ema_offset": [3],
        "ema_slope_threshold": [0.0],

        # Keep ADX ON and focused around winners
        # "use_adx": [True],
        # "adx_period": [10, 14],
        # "adx_threshold": [15.0, 18.0],
        # "require_adx_bias": [False, True],
    }):
        param_sets.append(params)

    return _deduplicate_param_sets(param_sets)

def preset_btc_ema_micro_grid() -> List[Dict[str, Any]]:
    param_sets = []

    for params in build_grid({
        "psar_step": [0.005, 0.006, 0.0075],
        "psar_max_step": [0.04, 0.05, 0.06, 0.08],
        "atr_period": [14],
        "use_ema_trend": [True],
        "ema_trend_period": [200],
        "ema_offset": [3],
        "ema_slope_threshold": [0.0],
        "use_adx": [False],
    }):
        param_sets.append(params)

    return _deduplicate_param_sets(param_sets)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    MODE = "study_final_focused"
    # Recommended order:
    # 1. study_psar
    # 2. study_adx
    # 3. study_ema
    # 4. study_final_focused
    # 4.a study_preset_btc_ema_micro_grid
    # 5. study_full
    #
    # Options:
    # - study_psar
    # - study_adx
    # - study_ema
    # - study_final_focused
    # - study_full

    backtest_config = BacktestConfig(
        symbol=SYMBOLS["BTC"],
        timeframe="1m",
        start_date=datetime(2026, 2, 12),
        end_date=datetime(2026, 4, 14),
        cut_off=500,
        initial_balance=200.0,
        use_close_signal=True,

        risk_percent=0.01,
        rr=2.0,
        atr_multiplier=2.0,
        sl_mode="WIDER",
    )

    param_sets = {
        "study_psar": preset_psar_focus,
        "study_adx": preset_adx_focus,
        "study_ema": preset_ema_focus,
        "study_final_focused": preset_final_focused_grid,
        "study_preset_btc_ema_micro_grid": preset_btc_ema_micro_grid,
        "study_full": preset_full_grid,
    }[MODE]()

    study_config = StudyConfig(
        backtest=backtest_config,
        temp_trade_log="_psar_corrected_study_tmp_trades.csv",
        summary_output=f"parabolic_sar_corrected_{MODE}.csv",
        best_output=f"parabolic_sar_corrected_{MODE}_best.csv",
        top_by_category_output=f"parabolic_sar_corrected_{MODE}_top_by_category.csv",
        delete_temp_trade_log=True,
        top_n_to_print=20,
        top_n_per_category=5,
        sort_by="score",
        min_trades=80,
    )

    run_parameter_study(
        param_sets=param_sets,
        study_config=study_config,
    )