"""
Session Trade Analysis for Golden Ribbon trade logs.

Reads trades_golden_ribbon.csv, classifies each trade by entry time, then reports
which sessions/hours perform best for the strategy.

Default assumption:
- EntryTime is in your broker/server time as written in the CSV.
- Session windows below are configurable. Adjust them if your broker time is not UTC.

Outputs:
- session_summary.csv
- hourly_summary.csv
- weekday_summary.csv
- session_hour_matrix.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = "trades_golden_ribbon.csv"
OUTPUT_DIR = "session_analysis_results"

# If your EntryTime is broker/server time and you want to shift it before
# classification, set this offset. Example: broker time UTC+2 and you want UTC:
# TIME_SHIFT_HOURS = -2
TIME_SHIFT_HOURS = 0

# Minimum number of trades required before a session/hour is trusted.
MIN_TRADES_FOR_VERDICT = 30

# Session windows use the adjusted EntryTime after TIME_SHIFT_HOURS.
# Format: (session_name, start_hour, end_hour), end is exclusive.
# Handles midnight wrap if start_hour > end_hour.
SESSION_WINDOWS = [
    ("Asia", 0, 7),
    ("London", 7, 12),
    ("London_NY_Overlap", 12, 16),
    ("New_York", 16, 21),
    ("Rollover_OffHours", 21, 24),
]

# Optional: define broader windows too. These are useful because Gold can behave
# differently during the full London/NY active period.
COMPOSITE_WINDOWS = [
    ("London_plus_Overlap", 7, 16),
    ("NY_plus_Overlap", 12, 21),
    ("Main_Active_Window", 7, 21),
]


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class VerdictRules:
    min_trades: int = MIN_TRADES_FOR_VERDICT
    good_pf: float = 1.20
    acceptable_pf: float = 1.05
    max_dd_good_pct: float = 15.0
    max_dd_acceptable_pct: float = 25.0


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _max_drawdown_pct(pnl: pd.Series, initial_balance: float | None = None) -> float:
    """Drawdown percentage using cumulative PnL curve.

    If initial_balance is unavailable, use cumulative PnL peak denominator.
    If BalanceAfter exists, this script also reports balance-based drawdown separately.
    """
    if pnl.empty:
        return 0.0
    cumulative = pnl.cumsum()
    if initial_balance is not None:
        equity = initial_balance + cumulative
        peak = equity.cummax()
        dd = peak - equity
        denom = peak.replace(0, pd.NA)
        dd_pct = (dd / denom * 100).fillna(0)
        return float(dd_pct.max())

    peak = cumulative.cummax()
    dd = peak - cumulative
    peak_safe = peak.replace(0, pd.NA)
    dd_pct = (dd / peak_safe * 100).fillna(0)
    return float(dd_pct.max())


def _sharpe_like(pnl: pd.Series) -> float:
    if len(pnl) < 2:
        return 0.0
    std = pnl.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(pnl.mean() / std * math.sqrt(len(pnl)))


def summarize_group(df: pd.DataFrame, group_col: str, initial_balance: float | None = None) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col, dropna=False):
        pnl = g["PnL"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        total = len(g)
        rows.append({
            group_col: name,
            "trades": total,
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "win_rate_pct": (len(wins) / total * 100) if total else 0.0,
            "total_pnl": pnl.sum(),
            "avg_pnl": pnl.mean() if total else 0.0,
            "median_pnl": pnl.median() if total else 0.0,
            "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0,
            "profit_factor": _profit_factor(pnl),
            "max_dd_pct": _max_drawdown_pct(pnl, initial_balance),
            "sharpe_like": _sharpe_like(pnl),
            "sl_count": int((g["ExitType"] == "SL").sum()) if "ExitType" in g else 0,
            "tp_count": int((g["ExitType"] == "TP").sum()) if "ExitType" in g else 0,
            "strategy_close_count": int((g["ExitType"] == "StrategyClose").sum()) if "ExitType" in g else 0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["profit_factor", "total_pnl"], ascending=[False, False]).reset_index(drop=True)


def score_rows(summary: pd.DataFrame, rules: VerdictRules) -> pd.DataFrame:
    if summary.empty:
        return summary
    s = summary.copy()

    # Penalize tiny samples heavily. PF and Sharpe matter, but only if enough trades exist.
    sample_factor = (s["trades"] / rules.min_trades).clip(upper=1.0)
    finite_pf = s["profit_factor"].replace([float("inf")], 5.0).fillna(0)

    s["quality_score"] = (
        finite_pf * 40
        + s["sharpe_like"].fillna(0) * 20
        + s["win_rate_pct"].fillna(0) * 0.2
        + s["total_pnl"].fillna(0) * 0.02
        - s["max_dd_pct"].fillna(0) * 0.8
    ) * sample_factor

    def verdict(row) -> str:
        if row["trades"] < rules.min_trades:
            return "IGNORE: too few trades"
        if row["profit_factor"] >= rules.good_pf and row["total_pnl"] > 0 and row["max_dd_pct"] <= rules.max_dd_good_pct:
            return "TRADE: strong candidate"
        if row["profit_factor"] >= rules.acceptable_pf and row["total_pnl"] > 0 and row["max_dd_pct"] <= rules.max_dd_acceptable_pct:
            return "TEST: acceptable, needs validation"
        if row["total_pnl"] > 0 and row["profit_factor"] > 1.0:
            return "WEAK: profitable but fragile"
        return "AVOID"

    s["verdict"] = s.apply(verdict, axis=1)
    return s.sort_values("quality_score", ascending=False).reset_index(drop=True)


# =============================================================================
# Session classification
# =============================================================================

def hour_in_window(hour: int, start: int, end: int) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def classify_session(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    for name, start, end in SESSION_WINDOWS:
        if hour_in_window(hour, start, end):
            return name
    return "Unclassified"


def classify_composite(ts: pd.Timestamp) -> list[str]:
    hour = int(ts.hour)
    labels = []
    for name, start, end in COMPOSITE_WINDOWS:
        if hour_in_window(hour, start, end):
            labels.append(name)
    return labels


# =============================================================================
# Main analysis
# =============================================================================

def analyze_trade_sessions(
    input_file: str = INPUT_FILE,
    output_dir: str = OUTPUT_DIR,
    time_shift_hours: int = TIME_SHIFT_HOURS,
    min_trades_for_verdict: int = MIN_TRADES_FOR_VERDICT,
) -> None:
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"EntryTime", "PnL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["EntryTime"] = pd.to_datetime(df["EntryTime"], errors="coerce")
    df = df.dropna(subset=["EntryTime"]).copy()
    df["AdjustedEntryTime"] = df["EntryTime"] + pd.to_timedelta(time_shift_hours, unit="h")
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)

    # Estimate initial balance if possible from first row: BalanceAfter - PnL.
    initial_balance = None
    if "BalanceAfter" in df.columns and not df.empty:
        df["BalanceAfter"] = pd.to_numeric(df["BalanceAfter"], errors="coerce")
        first = df.iloc[0]
        if pd.notna(first.get("BalanceAfter")):
            initial_balance = float(first["BalanceAfter"] - first["PnL"])

    df["session"] = df["AdjustedEntryTime"].apply(classify_session)
    df["hour"] = df["AdjustedEntryTime"].dt.hour
    df["weekday"] = df["AdjustedEntryTime"].dt.day_name()
    df["date"] = df["AdjustedEntryTime"].dt.date

    rules = VerdictRules(min_trades=min_trades_for_verdict)

    session_summary = score_rows(summarize_group(df, "session", initial_balance), rules)
    hourly_summary = score_rows(summarize_group(df, "hour", initial_balance), rules)
    weekday_summary = score_rows(summarize_group(df, "weekday", initial_balance), rules)

    # Composite windows are not mutually exclusive. Create separate summary rows.
    composite_rows = []
    for name, start, end in COMPOSITE_WINDOWS:
        mask = df["AdjustedEntryTime"].dt.hour.apply(lambda h: hour_in_window(int(h), start, end))
        g = df[mask].copy()
        if g.empty:
            continue
        g["composite_session"] = name
        composite_rows.append(summarize_group(g, "composite_session", initial_balance))
    composite_summary = pd.concat(composite_rows, ignore_index=True) if composite_rows else pd.DataFrame()
    composite_summary = score_rows(composite_summary, rules) if not composite_summary.empty else composite_summary

    # Session-hour matrix by PF and PnL for quick diagnosis.
    matrix = df.pivot_table(
        index="session",
        columns="hour",
        values="PnL",
        aggfunc=["count", "sum", "mean"],
        fill_value=0,
    )

    session_summary.to_csv(out_dir / "session_summary.csv", index=False)
    hourly_summary.to_csv(out_dir / "hourly_summary.csv", index=False)
    weekday_summary.to_csv(out_dir / "weekday_summary.csv", index=False)
    composite_summary.to_csv(out_dir / "composite_session_summary.csv", index=False)
    matrix.to_csv(out_dir / "session_hour_matrix.csv")

    print("\nGolden Ribbon Session Analysis")
    print("=" * 72)
    print(f"Input file:           {input_path}")
    print(f"Trades analysed:      {len(df)}")
    print(f"Time shift hours:     {time_shift_hours}")
    print(f"Estimated balance:    {initial_balance if initial_balance is not None else 'unknown'}")
    print(f"Output folder:        {out_dir.resolve()}")

    print("\nSession verdicts")
    print("-" * 72)
    cols = ["session", "trades", "win_rate_pct", "profit_factor", "total_pnl", "max_dd_pct", "sharpe_like", "quality_score", "verdict"]
    print(session_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not composite_summary.empty:
        print("\nComposite window verdicts")
        print("-" * 72)
        cols2 = ["composite_session", "trades", "win_rate_pct", "profit_factor", "total_pnl", "max_dd_pct", "sharpe_like", "quality_score", "verdict"]
        print(composite_summary[cols2].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nBest hours")
    print("-" * 72)
    hcols = ["hour", "trades", "win_rate_pct", "profit_factor", "total_pnl", "max_dd_pct", "quality_score", "verdict"]
    print(hourly_summary[hcols].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Final recommendation.
    tradable = session_summary[session_summary["verdict"].str.startswith(("TRADE", "TEST"))]
    if tradable.empty:
        print("\nFINAL VERDICT: No session is strong enough yet. Keep all sessions enabled only for research, or add more filters.")
    else:
        best = tradable.iloc[0]
        print("\nFINAL VERDICT")
        print("-" * 72)
        print(
            f"Best primary session: {best['session']} | "
            f"PF={best['profit_factor']:.3f}, "
            f"PnL={best['total_pnl']:.2f}, "
            f"DD={best['max_dd_pct']:.2f}%, "
            f"Trades={int(best['trades'])}."
        )
        print("Use this as a first session filter candidate, then validate it on unseen data before hard-coding it.")


if __name__ == "__main__":
    analyze_trade_sessions(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        time_shift_hours=TIME_SHIFT_HOURS,
        min_trades_for_verdict=MIN_TRADES_FOR_VERDICT,
    )
