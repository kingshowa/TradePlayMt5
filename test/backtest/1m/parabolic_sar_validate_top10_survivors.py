from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.strategy.parabolic_sar_strategy_v1 import ParabolicSarStrategy


# =============================================================================
# CONFIG
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

TOP_FILE_NAME = "parabolic_sar_corrected_study_final_focused.csv"

OUTPUT_ALL_RESULTS = "parabolic_sar_top10_validation_results.csv"
OUTPUT_SURVIVORS = "parabolic_sar_top10_survivors.csv"

SYMBOL = SYMBOLS["ETH"]
TIMEFRAME = "1m"

INITIAL_BALANCE = 200.0
CUT_OFF = 300
TOP_N = 10

USE_CLOSE_SIGNAL = True

RISK_PERCENT = 0.01
RR = 2.0
ATR_MULTIPLIER = 1.5

# WIDER = choose farther stop between PSAR dot and ATR stop.
# ATR   = use ATR stop only.
# PSAR  = use PSAR dot only.
# TIGHTER = choose closer stop.
SL_MODE = "WIDER"

# Validation windows.
# Since your optimization was on 2024 Jan-Dec, test on unseen 2025/2026.
VALIDATION_WINDOWS = [
    {
        "name": "validation_2025_jly_spt",
        "start": datetime(2026, 3, 16),
        "end": datetime(2026, 3, 22),
    },
    {
        "name": "validation_2025_oct_dec",
        "start": datetime(2026, 3, 23),
        "end": datetime(2026, 3, 29),
    },
]

# Survival rules.
MIN_TRADES_PER_WINDOW = 80
MIN_PROFIT_FACTOR = 1.03
MAX_DRAWDOWN_PCT = 40.0
REQUIRE_POSITIVE_RETURN = True


# =============================================================================
# TRADE + RISK
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


class ValidationRiskManager:
    VALID_SL_MODES = {"WIDER", "TIGHTER", "PSAR", "ATR"}

    def __init__(
        self,
        risk_percent: float = 0.01,
        rr: float = 2.0,
        atr_multiplier: float = 1.5,
        sl_mode: str = "WIDER",
    ):
        self.risk_percent = risk_percent
        self.rr = rr
        self.atr_multiplier = atr_multiplier
        self.sl_mode = sl_mode.upper()

        if self.sl_mode not in self.VALID_SL_MODES:
            raise ValueError(f"Invalid sl_mode: {sl_mode}")

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


class TradeSimulation:
    @staticmethod
    def check_exit(trade: BacktestTrade, candle):
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
# HELPERS
# =============================================================================

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in ("true", "1", "yes", "y")


def _optional_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None

    if value == "":
        return None

    return int(float(value))


def _optional_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None

    if value == "":
        return None

    return float(value)


def _load_top_candidates(path: Path, top_n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"No rows found in {path}")

    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    df = df.head(top_n).copy()

    candidates: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        use_ema = _to_bool(row.get("use_ema_trend", False))
        use_adx = _to_bool(row.get("use_adx", False))

        params: Dict[str, Any] = {
            "candidate_id": len(candidates) + 1,

            "psar_step": float(row["psar_step"]),
            "psar_max_step": float(row["psar_max_step"]),

            "atr_period": int(float(row["atr_period"])),

            "use_ema_trend": use_ema,
            "use_adx": use_adx,
        }

        if use_ema:
            params["ema_trend_period"] = _optional_int(row.get("ema_trend_period"))
            params["ema_offset"] = _optional_int(row.get("ema_offset")) or 3
            params["ema_slope_threshold"] = _optional_float(row.get("ema_slope_threshold")) or 0.0

        if use_adx:
            params["adx_period"] = _optional_int(row.get("adx_period"))
            params["adx_threshold"] = _optional_float(row.get("adx_threshold"))
            params["require_adx_bias"] = _to_bool(row.get("require_adx_bias", False))

        candidates.append(params)

    return candidates


def _strategy_params_for_constructor(params: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(params)
    clean.pop("candidate_id", None)

    if not clean.get("use_ema_trend"):
        clean.pop("ema_trend_period", None)
        clean.pop("ema_offset", None)
        clean.pop("ema_slope_threshold", None)

    if not clean.get("use_adx"):
        clean.pop("adx_period", None)
        clean.pop("adx_threshold", None)
        clean.pop("require_adx_bias", None)

    return clean


def _compute_stats(trades: List[Dict[str, Any]], initial_balance: float) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "exit_sl": 0,
            "exit_tp": 0,
            "exit_strategy_close": 0,
            "exit_force_close": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "sl_source_psar": 0,
            "sl_source_atr": 0,
        }

    df = pd.DataFrame(trades)
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    cumulative = pnl.cumsum()
    equity = initial_balance + cumulative
    running_peak = equity.cummax()
    drawdown = running_peak - equity

    max_drawdown = drawdown.max() if not drawdown.empty else 0.0
    peak_at_dd = running_peak.loc[drawdown.idxmax()] if not drawdown.empty else initial_balance

    max_drawdown_pct = (
        max_drawdown / peak_at_dd * 100
        if peak_at_dd and peak_at_dd > 0
        else 0.0
    )

    std_pnl = pnl.std()
    sharpe_ratio = (
        pnl.mean() / std_pnl * math.sqrt(len(pnl))
        if std_pnl and std_pnl > 0
        else 0.0
    )

    total_return_pct = pnl.sum() / initial_balance * 100 if initial_balance else 0.0

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
        "total_pnl": pnl.sum(),
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "exit_sl": int((df["exit_type"] == "SL").sum()),
        "exit_tp": int((df["exit_type"] == "TP").sum()),
        "exit_strategy_close": int((df["exit_type"] == "StrategyClose").sum()),
        "exit_force_close": int((df["exit_type"] == "ForceClose").sum()),
        "buy_trades": int((df["direction"] == "BUY").sum()),
        "sell_trades": int((df["direction"] == "SELL").sum()),
        "sl_source_psar": int((df["sl_source"] == "PSAR").sum()),
        "sl_source_atr": int((df["sl_source"] == "ATR").sum()),
    }


def _score(stats: Dict[str, Any]) -> float:
    trades = float(stats["total_trades"])
    pf = float(stats["profit_factor"])
    ret = float(stats["total_return_pct"])
    dd = float(stats["max_drawdown_pct"])
    wr = float(stats["win_rate"])
    sharpe = float(stats["sharpe_ratio"])
    calmar = float(stats["calmar_ratio"])

    pf = 8.0 if math.isinf(pf) else min(pf, 8.0)

    if ret <= 0:
        return -9999.0

    trade_penalty = 0.0
    if trades < MIN_TRADES_PER_WINDOW:
        trade_penalty = (MIN_TRADES_PER_WINDOW - trades) * 3.0

    return round(
        pf * 35.0
        + calmar * 18.0
        + wr * 80.0
        + sharpe * 14.0
        + ret * 0.7
        - dd * 3.0
        - trade_penalty,
        4,
    )


def _survived(stats: Dict[str, Any]) -> bool:
    if stats["total_trades"] < MIN_TRADES_PER_WINDOW:
        return False

    if REQUIRE_POSITIVE_RETURN and stats["total_return_pct"] <= 0:
        return False

    if stats["profit_factor"] < MIN_PROFIT_FACTOR:
        return False

    if stats["max_drawdown_pct"] > MAX_DRAWDOWN_PCT:
        return False

    return True


# =============================================================================
# BACKTEST
# =============================================================================

def run_candidate_on_series(series, params: Dict[str, Any]) -> Dict[str, Any]:
    warmup_series = series.subseries(0, CUT_OFF)

    strategy = ParabolicSarStrategy(
        warmup_series,
        use_close_signal=USE_CLOSE_SIGNAL,
        **_strategy_params_for_constructor(params),
    )

    risk_engine = ValidationRiskManager(
        risk_percent=RISK_PERCENT,
        rr=RR,
        atr_multiplier=ATR_MULTIPLIER,
        sl_mode=SL_MODE,
    )

    balance = INITIAL_BALANCE
    active_trade: Optional[BacktestTrade] = None
    active_idx: Optional[int] = None
    trades: List[Dict[str, Any]] = []

    for i in range(CUT_OFF, len(series._candles)):
        candle = series._candles[i]
        signal = strategy.update(candle)

        if active_trade is not None and active_idx is not None and i > active_idx:
            exit_result = TradeSimulation.check_exit(active_trade, candle)

            if exit_result is not None:
                exit_price, exit_type, pnl = exit_result
                balance += pnl

                strategy.on_trade_closed(exit_type=exit_type, exit_price=exit_price)

                trades.append({
                    "direction": active_trade.direction,
                    "entry": active_trade.entry,
                    "sl": active_trade.stop_loss,
                    "tp": active_trade.take_profit,
                    "exit_price": exit_price,
                    "exit_type": exit_type,
                    "pnl": pnl,
                    "balance_after": balance,
                    "sl_source": active_trade.sl_source,
                })

                active_trade = None
                active_idx = None
                continue

            if USE_CLOSE_SIGNAL and signal is not None and signal.signal == "CLOSE":
                exit_price = candle.close

                if active_trade.direction == "BUY":
                    pnl = (exit_price - active_trade.entry) * active_trade.position_size
                else:
                    pnl = (active_trade.entry - exit_price) * active_trade.position_size

                balance += pnl

                strategy.on_trade_closed(exit_type="StrategyClose", exit_price=exit_price)

                trades.append({
                    "direction": active_trade.direction,
                    "entry": active_trade.entry,
                    "sl": active_trade.stop_loss,
                    "tp": active_trade.take_profit,
                    "exit_price": exit_price,
                    "exit_type": "StrategyClose",
                    "pnl": pnl,
                    "balance_after": balance,
                    "sl_source": active_trade.sl_source,
                })

                active_trade = None
                active_idx = None
                continue

            continue

        if active_trade is None and signal is not None and signal.signal in ("BUY", "SELL"):
            trade = risk_engine.build_trade(signal, balance)

            if trade is None:
                continue

            active_trade = trade
            active_idx = i
            strategy.sync_trade(active_trade)

    if active_trade is not None:
        last_candle = series._candles[-1]
        exit_price = last_candle.close

        if active_trade.direction == "BUY":
            pnl = (exit_price - active_trade.entry) * active_trade.position_size
        else:
            pnl = (active_trade.entry - exit_price) * active_trade.position_size

        balance += pnl

        strategy.on_trade_closed(exit_type="ForceClose", exit_price=exit_price)

        trades.append({
            "direction": active_trade.direction,
            "entry": active_trade.entry,
            "sl": active_trade.stop_loss,
            "tp": active_trade.take_profit,
            "exit_price": exit_price,
            "exit_type": "ForceClose",
            "pnl": pnl,
            "balance_after": balance,
            "sl_source": active_trade.sl_source,
        })

    stats = _compute_stats(trades, INITIAL_BALANCE)
    stats["final_balance"] = balance
    stats["score"] = _score(stats)

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    top_file = script_dir / TOP_FILE_NAME

    if not top_file.exists():
        raise FileNotFoundError(
            f"Could not find {TOP_FILE_NAME} in {script_dir}"
        )

    candidates = _load_top_candidates(top_file, TOP_N)

    print("\nTop-10 Survivor Validation")
    print("=" * 72)
    print(f"Top file: {top_file}")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Candidates: {len(candidates)}")
    print(f"Risk: {RISK_PERCENT * 100:.2f}% | RR: {RR} | ATR x{ATR_MULTIPLIER} | SL mode: {SL_MODE}")
    print("=" * 72)

    provider = MT5MarketDataProvider()

    all_rows: List[Dict[str, Any]] = []

    for window in VALIDATION_WINDOWS:
        print(f"\nFetching {window['name']}: {window['start'].date()} → {window['end'].date()}")

        series = provider.fetch_range(
            SYMBOL,
            TIMEFRAMES[TIMEFRAME],
            window["start"],
            window["end"],
        )

        if len(series._candles) <= CUT_OFF:
            print(f"Skipping {window['name']} — not enough candles.")
            continue

        print(f"Loaded {len(series._candles):,} candles.")

        for candidate in candidates:
            print(
                f"  Candidate {candidate['candidate_id']:>2} | "
                f"PSAR={candidate['psar_step']}/{candidate['psar_max_step']} | "
                f"ATR={candidate['atr_period']}"
            )

            stats = run_candidate_on_series(series, candidate)
            survived = _survived(stats)

            row = {
                **candidate,
                "window": window["name"],
                "start_date": window["start"].date(),
                "end_date": window["end"].date(),
                "survived_window": survived,
                **stats,
            }

            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    if results_df.empty:
        print("No validation results generated.")
        return

    results_path = script_dir / OUTPUT_ALL_RESULTS
    results_df.to_csv(results_path, index=False)

    # Candidate-level survival summary.
    summary_rows: List[Dict[str, Any]] = []

    for candidate_id, group in results_df.groupby("candidate_id"):
        windows_tested = len(group)
        windows_survived = int(group["survived_window"].sum())

        summary = {
            "candidate_id": candidate_id,
            "windows_tested": windows_tested,
            "windows_survived": windows_survived,
            "survival_rate": windows_survived / windows_tested if windows_tested else 0.0,

            "avg_score": group["score"].mean(),
            "avg_return_pct": group["total_return_pct"].mean(),
            "avg_profit_factor": group["profit_factor"].replace([float("inf")], 8.0).mean(),
            "avg_drawdown_pct": group["max_drawdown_pct"].mean(),
            "avg_trades": group["total_trades"].mean(),

            "min_return_pct": group["total_return_pct"].min(),
            "min_profit_factor": group["profit_factor"].replace([float("inf")], 8.0).min(),
            "max_drawdown_pct_seen": group["max_drawdown_pct"].max(),
        }

        first = group.iloc[0].to_dict()

        for key in [
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
        ]:
            if key in first:
                summary[key] = first[key]

        summary["survived_all_windows"] = windows_survived == windows_tested

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=[
            "survived_all_windows",
            "survival_rate",
            "avg_score",
            "avg_profit_factor",
            "avg_return_pct",
        ],
        ascending=False,
    )

    survivors_df = summary_df[summary_df["survived_all_windows"] == True].copy()

    survivors_path = script_dir / OUTPUT_SURVIVORS
    survivors_df.to_csv(survivors_path, index=False)

    print("\nValidation complete.")
    print(f"All validation results: {results_path}")
    print(f"Survivors: {survivors_path}")

    print("\nTop survivor summary:")
    if survivors_df.empty:
        print("No candidate survived all validation windows.")
        print("Check the full validation results and consider relaxing thresholds slightly.")
    else:
        display_cols = [
            "candidate_id",
            "survival_rate",
            "avg_score",
            "avg_return_pct",
            "avg_profit_factor",
            "avg_drawdown_pct",
            "avg_trades",
            "psar_step",
            "psar_max_step",
            "atr_period",
            "ema_trend_period",
            "adx_period",
            "adx_threshold",
            "require_adx_bias",
        ]

        display_cols = [c for c in display_cols if c in survivors_df.columns]
        print(survivors_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()