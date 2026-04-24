# Candle-only precision PSAR backtest. It does not call strategy.on_tick().
# It calls strategy.check_candle_psar_cross(candle) before strategy.update(candle).

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from app.core.market.candle import Candle
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.strategy.precision_psar_strategy import PrecisionPsarStrategy


SYMBOLS = {
    "Gold": "XAUUSDm",
    "Silver": "XAGUSDm",
    "Platinum": "XPTUSDm",
    "Oil": "USOILm",
    "Euro": "EURUSDm",
    "EuroJpy": "EURJPYm",
    "BTC": "BTCUSDm",
    "ETH": "ETHUSDm",
}


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


class BacktestRiskManager:
    """
    Backtest-only risk model matching the live RiskManager contract.

    The strategy returns a raw PSAR-derived SL estimate through signal.sl.
    This risk manager compares that PSAR SL with the ATR stop and selects the
    final stop using sl_mode.
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
            raise ValueError(f"Invalid sl_mode. Use one of {self.VALID_SL_MODES}")

        self.risk_percent = risk_percent
        self.rr = rr
        self.atr_multiplier = atr_multiplier
        self.sl_mode = sl_mode

    def build_trade(self, signal, balance: float) -> Optional[BacktestTrade]:
        direction = str(signal.signal).upper()
        candle = signal.candle
        entry = float(candle.close)
        atr = signal.atr
        psar_sl = signal.sl

        if direction not in ("BUY", "SELL"):
            return None
        if atr is None or atr <= 0:
            return None

        atr_sl = self._atr_stop(direction, candle, float(atr))
        final_sl, sl_source = self._select_stop(direction, entry, psar_sl, atr_sl)
        if final_sl is None:
            return None

        stop_distance = abs(entry - final_sl)
        if stop_distance <= 0:
            return None

        risk_amount = balance * self.risk_percent
        position_size = risk_amount / stop_distance
        take_profit = (
            entry + stop_distance * self.rr
            if direction == "BUY"
            else entry - stop_distance * self.rr
        )

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
            return float(candle.low) - self.atr_multiplier * atr
        return float(candle.high) + self.atr_multiplier * atr

    def _select_stop(
        self,
        direction: str,
        entry: float,
        psar_sl: Optional[float],
        atr_sl: Optional[float],
    ):
        candidates = []

        if psar_sl is not None and self._valid_stop(direction, entry, float(psar_sl)):
            candidates.append(("PSAR", float(psar_sl)))
        if atr_sl is not None and self._valid_stop(direction, entry, float(atr_sl)):
            candidates.append(("ATR", float(atr_sl)))

        if not candidates:
            return None, "NONE"

        if self.sl_mode == "PSAR":
            return next(((v, s) for s, v in candidates if s == "PSAR"), (None, "NONE"))
        if self.sl_mode == "ATR":
            return next(((v, s) for s, v in candidates if s == "ATR"), (None, "NONE"))

        if direction == "BUY":
            source, value = (
                min(candidates, key=lambda item: item[1])
                if self.sl_mode == "WIDER"
                else max(candidates, key=lambda item: item[1])
            )
        else:
            source, value = (
                max(candidates, key=lambda item: item[1])
                if self.sl_mode == "WIDER"
                else min(candidates, key=lambda item: item[1])
            )

        return value, source

    def _valid_stop(self, direction: str, entry: float, stop: float) -> bool:
        return stop < entry if direction == "BUY" else stop > entry


class TradeSimulation:
    """
    Conservative candle-level SL / TP checker.

    If SL and TP are both touched in a candle, SL is assumed first.
    """

    @staticmethod
    def check_exit(trade: BacktestTrade, candle, use_tp: bool = True):
        direction = trade.direction.upper()
        entry = trade.entry
        sl = trade.stop_loss
        tp = trade.take_profit
        size = trade.position_size

        if direction == "BUY":
            if candle.low <= sl:
                return sl, "SL", (sl - entry) * size
            if use_tp and tp is not None and candle.high >= tp:
                return tp, "TP", (tp - entry) * size
        else:
            if candle.high >= sl:
                return sl, "SL", (entry - sl) * size
            if use_tp and tp is not None and candle.low <= tp:
                return tp, "TP", (entry - tp) * size
        return None


_CSV_HEADER = [
    "EntryTime", "Direction", "Entry", "SL", "TP", "PositionSize",
    "RiskAmount", "StopDistance", "RR", "SLSource", "PSAR_SL", "ATR_SL",
    "ExitPrice", "ExitType", "PnL", "BalanceAfter", "Pattern", "Reason",
]

_SUMMARY_HEADER = [
    "RunTimestamp", "Comment", "Symbol", "Timeframe", "StartDate", "EndDate",
    "Candles", "CutOff", "InitialBalance", "FinalBalance", "UseTP",
    "RiskPercent", "RR", "ATRMultiplier", "SLMode", "IntrabarPathMode",
    "SameCandleTP", "TotalTrades", "Wins", "Losses", "Breakeven", "WinRate",
    "ProfitFactor", "AvgWin", "AvgLoss", "BestTrade", "WorstTrade", "TotalPnL",
    "MaxConsecWins", "MaxConsecLoss", "MaxDrawdown", "MaxDrawdownPct",
    "TotalReturnPct", "SharpeRatio", "ExitSL", "ExitTP", "ExitForceClose",
    "BuyTrades", "SellTrades", "SLSourcePSAR", "SLSourceATR", "StrategyParams",
    "TradeLogFile",
]


def _write_csv_header(path: str) -> None:
    with open(path, "w", newline="") as file:
        csv.writer(file).writerow(_CSV_HEADER)


def _append_trade_row(path: str, trade: BacktestTrade, exit_price: float, exit_type: str, pnl: float, balance_after: float) -> None:
    with open(path, "a", newline="") as file:
        csv.writer(file).writerow([
            trade.candle.time,
            trade.direction,
            f"{trade.entry:.5f}",
            f"{trade.stop_loss:.5f}",
            f"{trade.take_profit:.5f}" if trade.take_profit is not None else "",
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


def _compute_stats(df: pd.DataFrame, initial_balance: float) -> dict:
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
    equity = initial_balance + cumulative
    running_peak = equity.cummax()
    drawdown = running_peak - equity
    max_dd = drawdown.max() if not drawdown.empty else 0.0
    peak_at_max_dd = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else initial_balance
    max_dd_pct = max_dd / peak_at_max_dd * 100 if peak_at_max_dd > 0 else 0.0
    std_pnl = pnl.std()
    sharpe = pnl.mean() / std_pnl * math.sqrt(len(pnl)) if std_pnl and std_pnl > 0 else 0.0

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
        "total_return_pct": pnl.sum() / initial_balance * 100 if initial_balance else 0.0,
        "sharpe_ratio": sharpe,
        "exits_by_type": df["ExitType"].value_counts().to_dict(),
        "exits_by_dir": df["Direction"].value_counts().to_dict(),
        "sl_sources": df["SLSource"].value_counts().to_dict(),
    }


def _print_stats(stats: dict, symbol: str, timeframe: str, initial_balance: float, final_balance: float, start_date: datetime, end_date: datetime) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Precision PSAR Live-Tick Backtest — {symbol} {timeframe.upper()}")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(sep)
    print(f"  {'Initial balance':<30} {initial_balance:>10.2f}")
    print(f"  {'Final balance':<30} {final_balance:>10.2f}")
    print(f"  {'Net PnL':<30} {stats['total_pnl']:>+10.2f}")
    print(f"  {'Return':<30} {stats['total_return_pct']:>+9.2f}%")
    print(sep)
    print(f"  {'Total trades':<30} {stats['total_trades']:>10}")
    print(f"  {'Wins / Losses / BE':<30} {stats['wins']:>4} / {stats['losses']:>4} / {stats['breakeven']:>4}")
    print(f"  {'Win rate':<30} {stats['win_rate'] * 100:>9.2f}%")
    print(f"  {'Profit factor':<30} {stats['profit_factor']:>10.3f}")
    print(sep)
    print(f"  {'Avg win':<30} {stats['avg_win']:>+10.2f}")
    print(f"  {'Avg loss':<30} {stats['avg_loss']:>+10.2f}")
    print(f"  {'Best trade':<30} {stats['best_trade']:>+10.2f}")
    print(f"  {'Worst trade':<30} {stats['worst_trade']:>+10.2f}")
    print(sep)
    print(f"  {'Max drawdown':<30} {stats['max_drawdown']:>+10.2f}")
    print(f"  {'Max drawdown %':<30} {stats['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Sharpe ratio':<30} {stats['sharpe_ratio']:>10.3f}")
    print(sep)


def _plot_results(df, stats, symbol, timeframe, start_date, end_date):
    pnl = df["PnL"]
    cumulative = pnl.cumsum()
    equity = cumulative
    running_peak = equity.cummax()
    drawdown = running_peak - equity

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Precision PSAR Live-Tick Backtest — {symbol} {timeframe.upper()}   "
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
    path, comment_name, stats, strategy_params, symbol, timeframe, start_date,
    end_date, candles_count, cut_off, initial_balance, final_balance, use_tp,
    risk_percent, rr, atr_multiplier, sl_mode, intrabar_path_mode,
    allow_same_candle_tp, trade_log_file,
):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    exits = stats.get("exits_by_type", {})
    dirs = stats.get("exits_by_dir", {})
    sl_sources = stats.get("sl_sources", {})

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
        "UseTP": use_tp,
        "RiskPercent": risk_percent,
        "RR": rr,
        "ATRMultiplier": atr_multiplier,
        "SLMode": sl_mode,
        "IntrabarPathMode": intrabar_path_mode,
        "SameCandleTP": allow_same_candle_tp,
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
        "ExitForceClose": exits.get("ForceClose", 0),
        "BuyTrades": dirs.get("BUY", 0),
        "SellTrades": dirs.get("SELL", 0),
        "SLSourcePSAR": sl_sources.get("PSAR", 0),
        "SLSourceATR": sl_sources.get("ATR", 0),
        "StrategyParams": json.dumps(strategy_params, sort_keys=True),
        "TradeLogFile": trade_log_file,
    }

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_SUMMARY_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _intrabar_prices(candle: Candle, mode: str = "conservative") -> list[Tuple[str, float]]:
    """
    Approximate tick sequence from OHLC.

    This does not know the real intrabar path. The default is conservative-ish:
    it moves first toward the side nearer to open, then the other extreme,
    then close. This avoids always giving the strategy the favorable path.
    """
    mode = mode.lower()
    if mode == "ohlc":
        return [("open", candle.open), ("high", candle.high), ("low", candle.low), ("close", candle.close)]
    if mode == "olhc":
        return [("open", candle.open), ("low", candle.low), ("high", candle.high), ("close", candle.close)]
    if mode == "conservative":
        high_dist = abs(candle.high - candle.open)
        low_dist = abs(candle.open - candle.low)
        if high_dist <= low_dist:
            return [("open", candle.open), ("high", candle.high), ("low", candle.low), ("close", candle.close)]
        return [("open", candle.open), ("low", candle.low), ("high", candle.high), ("close", candle.close)]
    raise ValueError("intrabar_path_mode must be one of: conservative, ohlc, olhc")


def _remaining_candle_after_entry(candle: Candle, entry_price: float, direction: str, current_step: str, path_mode: str) -> Candle:
    """
    Build a conservative remaining-candle proxy after entry.

    With OHLC data we cannot know exact post-entry path. We keep all adverse
    extremes that could still happen. TP can be disabled on entry candle through
    allow_same_candle_tp=False.
    """
    direction = direction.upper()
    return Candle(
        time=candle.time,
        open=entry_price,
        high=max(candle.high, entry_price),
        low=min(candle.low, entry_price),
        close=candle.close,
        volume=candle.volume,
    )


def _trail_trade_with_psar(active_trade: BacktestTrade, psar_sl: Optional[float]) -> bool:
    if psar_sl is None:
        return False
    psar_sl = float(psar_sl)
    direction = active_trade.direction.upper()
    if direction == "BUY" and psar_sl > active_trade.stop_loss and psar_sl < active_trade.entry:
        active_trade.stop_loss = psar_sl
        return True
    if direction == "SELL" and psar_sl < active_trade.stop_loss and psar_sl > active_trade.entry:
        active_trade.stop_loss = psar_sl
        return True
    return False


def backtest_precision_psar(
    symbol: str = SYMBOLS["BTC"],
    timeframe: str = "1m",
    start_date: datetime = datetime(2026, 4, 16),
    end_date: datetime = datetime(2026, 4, 28),
    cut_off: int = 500,
    initial_balance: float = 200.0,
    use_tp: bool = True,
    output_file: str = "trades_precision_psar.csv",
    strategy_params: Optional[dict] = None,
    comment_name: str = "live_tick_validation",
    summary_file: str = "precision_psar_experiment_results.csv",
    plot: bool = True,
    risk_percent: float = 0.01,
    rr: float = 2.0,
    atr_multiplier: float = 1.5,
    sl_mode: str = "WIDER",
    intrabar_path_mode: str = "conservative",
    allow_same_candle_tp: bool = False,
) -> Optional[dict]:
    print("\n  Precision PSAR Backtest — Candle Dot Cross")
    print(f"  Symbol: {symbol} | Timeframe: {timeframe}")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(f"  Initial balance: {initial_balance:.2f}")
    print(f"  Risk: {risk_percent * 100:.2f}% | RR: {rr:.2f} | ATR SL: {atr_multiplier:.2f}x | SL mode: {sl_mode}")
    print(f"  Take profit: {use_tp}")
    print(f"  Intrabar path: {intrabar_path_mode} | Same-candle TP: {allow_same_candle_tp}")
    print(f"  Warm-up candles: {cut_off}")
    print(f"  Comment: {comment_name}")

    strategy_params = dict(strategy_params or {})
    # Candle-only contract: check_candle_psar_cross() creates entries before update();
    # update() only refreshes confirmed indicators after candle processing.
    strategy_params["entry_on_update"] = False
    strategy_params["use_close_signal"] = False
    print(f"  Strategy params: {strategy_params if strategy_params else 'DEFAULTS'}")

    provider = MT5MarketDataProvider()
    series = provider.fetch_range(symbol, TIMEFRAMES[timeframe], start_date, end_date)
    total_candles = len(series._candles)
    print(f"  Loaded {total_candles:,} candles ({total_candles - cut_off:,} tradeable after warm-up)")

    if total_candles <= cut_off:
        print("  ERROR: not enough candles for selected cut_off.")
        return None

    warmup_series = series.subseries(0, cut_off)
    strategy = PrecisionPsarStrategy(warmup_series, **strategy_params)
    risk_engine = BacktestRiskManager(risk_percent=risk_percent, rr=rr, atr_multiplier=atr_multiplier, sl_mode=sl_mode)

    balance = initial_balance
    active_trade: Optional[BacktestTrade] = None
    active_trade_entry_idx: Optional[int] = None

    _write_csv_header(output_file)

    for i in range(cut_off, total_candles):
        candle = series._candles[i]

        # 1) Use the previously confirmed indicator state, just like live trading
        # before the candle has closed. Since this is a candle-only backtest,
        # do NOT call on_tick(). Instead check whether this candle's OHLC range
        # crossed the next PSAR trigger computed from the previous confirmed state.
        opened_this_candle = False
        if active_trade is None:
            signal = strategy.check_candle_psar_cross(candle)
            if signal is not None and signal.signal in ("BUY", "SELL"):
                trade = risk_engine.build_trade(signal, balance)
                if trade is not None:
                    active_trade = trade
                    active_trade_entry_idx = i
                    opened_this_candle = True
                    strategy.sync_trade(active_trade)

                    # Entry-candle risk. SL is always checked; TP is optional because
                    # OHLC cannot prove TP happened after the entry trigger.
                    remaining = _remaining_candle_after_entry(
                        candle,
                        trade.entry,
                        trade.direction,
                        current_step="cross",
                        path_mode=intrabar_path_mode,
                    )
                    exit_result = TradeSimulation.check_exit(
                        trade,
                        remaining,
                        use_tp=(use_tp and allow_same_candle_tp),
                    )
                    if exit_result is not None:
                        exit_price, exit_type, pnl = exit_result
                        balance += pnl
                        strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                        _append_trade_row(output_file, trade, exit_price, exit_type, pnl, balance)
                        active_trade = None
                        active_trade_entry_idx = None

        # 2) Manage a trade that was already active before this candle.
        if active_trade is not None and not opened_this_candle and i > active_trade_entry_idx:
            exit_result = TradeSimulation.check_exit(active_trade, candle, use_tp=use_tp)
            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl
                strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)
                _append_trade_row(output_file, active_trade, exit_price, exit_type, pnl, balance)
                active_trade = None
                active_trade_entry_idx = None

        # 3) Only now is the candle considered closed. Update indicators.
        strategy.update(candle)

        # 4) After confirmed candle close, trail the still-open trade with the
        # newly confirmed PSAR. This mirrors live trader trailing after update().
        if active_trade is not None:
            psar_sl = strategy.get_psar_sl(active_trade.direction)
            _trail_trade_with_psar(active_trade, psar_sl)

    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = float(last_candle.close)
        if active_trade.direction == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size
        balance += pnl
        strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)
        _append_trade_row(output_file, active_trade, exit_price, "ForceClose", pnl, balance)

    print(f"\n  Backtest complete. Reading results from {output_file}...")
    df = pd.read_csv(output_file)
    if df.empty:
        print("  No trades were executed.")
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
        use_tp=use_tp,
        risk_percent=risk_percent,
        rr=rr,
        atr_multiplier=atr_multiplier,
        sl_mode=sl_mode,
        intrabar_path_mode=intrabar_path_mode,
        allow_same_candle_tp=allow_same_candle_tp,
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
        "use_tp": use_tp,
        "risk_percent": risk_percent,
        "rr": rr,
        "atr_multiplier": atr_multiplier,
        "sl_mode": sl_mode,
        "intrabar_path_mode": intrabar_path_mode,
        "allow_same_candle_tp": allow_same_candle_tp,
        "strategy_params": strategy_params,
        **stats,
    }


if __name__ == "__main__":
    strategy_params = {
        "psar_step": 0.025,
        "psar_max_step": 0.05,
        "atr_period": 14,
        "use_ema_trend": True,
        "ema_trend_period": 50,
        "ema_offset": 3,
        "ema_slope_threshold": 0.0,
    }

    backtest_precision_psar(
        symbol=SYMBOLS["BTC"],
        timeframe="1m",
        start_date=datetime(2026, 4, 16),
        end_date=datetime(2026, 4, 28),
        cut_off=500,
        initial_balance=200.0,
        use_tp=True,
        output_file="trades_precision_psar.csv",
        strategy_params=strategy_params,
        comment_name="candle dot-cross validation on unseen data",
        summary_file="precision_psar_experiment_results.csv",
        plot=True,
        risk_percent=0.01,
        rr=2.0,
        atr_multiplier=1.5,
        sl_mode="WIDER",
        intrabar_path_mode="conservative",
        allow_same_candle_tp=False,
    )
