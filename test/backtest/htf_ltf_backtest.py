from datetime import timedelta, datetime
from typing import Optional
import csv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.risk.risk_manager import RiskManager

from app.core.strategy.htf_ltf.htf_analyzer import HTFAnalyzer, HTFConfig, Regime
from app.core.strategy.htf_ltf.ltf_analyzer import LTFAnalyzer, LTFConfig
from app.core.strategy.htf_ltf.htf_ltf_strategy import HTFLTFStrategy


# =========================
# Symbols
# =========================

SYMBOLS = {
    "Gold":     "XAUUSDm",
    "Silver":   "XAGUSDm",
    "Platinum": "XPTUSDm",
    "Oil":      "BNO",
    "Euro":     "EURUSDm",
    "EuroJpy":  "EURJPYm",
    "BTC":      "BTCUSDm",
    "ETH":      "ETHUSDm",
}

OUTPUT_CSV = "trades_outcome_htf_ltf.csv"

# Candle duration per timeframe key — used for look-ahead-free HTF sync
_TF_DURATION = {
    "1m":  timedelta(minutes=1),
    "3m":  timedelta(minutes=3),
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h":  timedelta(hours=1),
}


# =========================
# Dual-TF synchronisation
# =========================

class HTFSync:
    """
    Feeds closed HTF candles into the strategy as the LTF loop advances.

    A 5m candle with timestamp T is only emitted once a 1m candle with
    timestamp >= T + 5m arrives.  This prevents any look-ahead bias: the
    HTF candle is only "known" after its bar has fully closed.

    Usage inside the 1m loop:
        htf_sync = HTFSync(htf_candles, htf_timeframe="5m")
        for ltf_candle in ltf_candles:
            htf_sync.advance(ltf_candle.time, strategy.on_htf_candle)
            signal = strategy.on_ltf_candle(ltf_candle)
    """

    def __init__(self, htf_candles: list, htf_timeframe: str = "5m"):
        self._candles  = htf_candles
        self._ptr      = 0
        self._duration = _TF_DURATION.get(htf_timeframe, timedelta(minutes=5))

    def advance(self, ltf_time, on_htf_candle_fn) -> int:
        """
        Emit every HTF candle whose bar has fully closed by ltf_time.
        Calls on_htf_candle_fn(candle) for each one.
        Returns the number of HTF candles emitted this tick.
        """
        emitted = 0
        while (
            self._ptr < len(self._candles)
            and self._candles[self._ptr].time + self._duration <= ltf_time
        ):
            on_htf_candle_fn(self._candles[self._ptr])
            self._ptr += 1
            emitted += 1
        return emitted


# =========================
# Backtest
# =========================

def backtest_htf_ltf(
    symbol:             str   = SYMBOLS["Gold"],
    htf_timeframe:      str   = "5m",
    ltf_timeframe:      str   = "1m",
    ltf_bars:           int   = 5000,   # total 1m candles to fetch
    htf_bars:           int   = 2000,   # total 5m candles to fetch
    ltf_cutoff:         int   = 150,    # 1m warmup bars fed to initialize()
    htf_cutoff:         int   = 60,     # 5m warmup bars fed to initialize()
    balance:            float = 200.0,
    use_strategy_close: bool  = True,   # honour CLOSE signals from strategy
    use_tp:             bool  = False,  # honour hard TP level
    htf_ema_enabled:    bool  = True,   # toggle 5m EMA filter on/off
    # --- HTF config ---
    htf_ha_min_bars:    int   = 2,
    htf_ema_period:     int   = 20,
    # --- LTF config ---
    ltf_ema_period:     int   = 9,
    ltf_rsi_period:     int   = 14,
    ltf_rsi_strict:     bool  = False,
    ltf_atr_period:     int   = 14,
    ltf_atr_tp_mult:    float = 2.0,
) -> pd.DataFrame:
    """
    Run a candle-by-candle backtest of HTFLTFStrategy.

    Returns a DataFrame of all closed trades (same rows written to CSV).

    Dual-TF flow
    ------------
    1m series is the main loop driver.
    5m candles are fed through HTFSync — a 5m candle at time T is only
    released once a 1m candle at T + 5m (or later) arrives, preventing
    look-ahead bias.

    Parameters
    ----------
    symbol             : MT5 instrument symbol.
    htf_timeframe      : HTF bar size string (must exist in TIMEFRAMES).
    ltf_timeframe      : LTF bar size string.
    ltf_bars / htf_bars: Total bars to fetch per timeframe.
    ltf_cutoff         : First N 1m bars used for strategy initialisation.
    htf_cutoff         : First N 5m bars used for strategy initialisation.
    balance            : Starting account balance.
    use_strategy_close : When True, a CLOSE signal exits the position at
                         the candle close (labelled Strategy_Close or TP
                         if price already past target).
    use_tp             : When True, positions also exit on hard TP hit.
    htf_ema_enabled    : Enable/disable the 5m EMA confirmation filter.
    """
    print(
        f"\n🚀 HTF-LTF Backtest Started"
        f"\n   Symbol : {symbol}"
        f"\n   HTF    : {htf_timeframe}  |  LTF : {ltf_timeframe}"
        f"\n   HTF EMA: {'ON' if htf_ema_enabled else 'OFF'}"
        f"\n   Strat Close: {use_strategy_close}  |  Hard TP: {use_tp}"
    )

    # ── 1. Fetch data ─────────────────────────────────────────────────
    provider     = MT5MarketDataProvider()
    # ltf_series   = provider.fetch(symbol, TIMEFRAMES[ltf_timeframe], ltf_bars)
    # htf_series   = provider.fetch(symbol, TIMEFRAMES[htf_timeframe], htf_bars)

    START_DATE = datetime(2025, 1, 1)
    END_DATE = datetime(2026, 12, 30)

    ltf_series = provider.fetch_range(
        symbol,
        TIMEFRAMES[ltf_timeframe],
        START_DATE,
        END_DATE
    )

    htf_series = provider.fetch_range(
        symbol,
        TIMEFRAMES[htf_timeframe],
        START_DATE,
        END_DATE
    )

    ltf_candles  = ltf_series.candles()
    htf_candles  = htf_series.candles()

    if len(ltf_candles) < ltf_cutoff + 1:
        raise ValueError(f"Not enough 1m bars. Got {len(ltf_candles)}, need > {ltf_cutoff}.")
    if len(htf_candles) < htf_cutoff + 1:
        raise ValueError(f"Not enough 5m bars. Got {len(htf_candles)}, need > {htf_cutoff}.")

    # ── 2. Build and initialise strategy ─────────────────────────────
    htf_cfg = HTFConfig(
        ha_min_bars=htf_ha_min_bars,
        ema_period=htf_ema_period,
        ema_enabled=htf_ema_enabled,
    )
    ltf_cfg = LTFConfig(
        ema_period=ltf_ema_period,
        rsi_period=ltf_rsi_period,
        rsi_strict_slope=ltf_rsi_strict,
        atr_period=ltf_atr_period,
        atr_tp_multiplier=ltf_atr_tp_mult,
    )

    htf_analyzer = HTFAnalyzer(htf_cfg)
    ltf_analyzer = LTFAnalyzer(ltf_cfg)
    strategy     = HTFLTFStrategy(htf_analyzer, ltf_analyzer)

    strategy.initialize(
        htf_series=htf_series.subseries(0, htf_cutoff),
        ltf_series=ltf_series.subseries(0, ltf_cutoff),
    )

    risk_engine  = RiskManager()
    htf_sync     = HTFSync(htf_candles, htf_timeframe)

    # Fast-forward the HTF sync pointer past the warmup window so it
    # only starts emitting candles that arrive after initialisation.
    warmup_end_time = ltf_candles[ltf_cutoff - 1].time
    htf_sync.advance(warmup_end_time, lambda _: None)

    # ── 3. Prepare CSV ────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EntryTime", "ExitTime", "Direction",
            "Entry", "SL", "TP",
            "PositionSize", "ExitPrice", "ExitType", "PnL",
            "Regime", "HABars", "RSIAtEntry", "ATRAtEntry",
            "Pattern", "Reason",
        ])

    # ── 4. State ──────────────────────────────────────────────────────
    blc          = balance
    active_trade = None   # trade object from RiskManager
    trade_meta   = {}     # extra context not stored on the trade object
    trades_count = 0

    # ── 5. Main loop (1m candle-by-candle) ───────────────────────────
    for i in range(ltf_cutoff, len(ltf_candles)):
        candle = ltf_candles[i]

        # Feed any newly-closed 5m candles before processing this 1m bar
        htf_sync.advance(candle.time, strategy.on_htf_candle)

        signal = strategy.on_ltf_candle(candle)

        # ── A. Handle active trade ────────────────────────────────────
        if active_trade:
            exit_price: Optional[float] = None
            exit_type:  Optional[str]   = None

            # SL — always checked first
            if active_trade.direction == "BUY":
                if candle.low <= active_trade.stop_loss:
                    exit_price, exit_type = active_trade.stop_loss, "SL"
            else:
                if candle.high >= active_trade.stop_loss:
                    exit_price, exit_type = active_trade.stop_loss, "SL"

            # Hard TP (optional)
            if exit_type is None and use_tp:
                if active_trade.direction == "BUY" and candle.high >= active_trade.take_profit:
                    exit_price, exit_type = active_trade.take_profit, "TP"
                elif active_trade.direction == "SELL" and candle.low <= active_trade.take_profit:
                    exit_price, exit_type = active_trade.take_profit, "TP"

            # Strategy CLOSE signal (optional)
            if exit_type is None and use_strategy_close and signal and signal.signal == "CLOSE":
                if active_trade.direction == "BUY":
                    exit_type = "TP" if candle.close >= active_trade.take_profit else "Strategy_Close"
                else:
                    exit_type = "TP" if candle.close <= active_trade.take_profit else "Strategy_Close"
                exit_price = candle.close
                print(f"   ↩ CLOSE | {signal.reason} | price={exit_price:.2f}")

            # Record exit
            if exit_type and exit_price is not None:
                if active_trade.direction == "BUY":
                    pnl = (exit_price - active_trade.entry) * active_trade.position_size
                else:
                    pnl = (active_trade.entry - exit_price) * active_trade.position_size

                blc += pnl

                with open(OUTPUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        active_trade.candle.time,           # EntryTime
                        candle.time,                        # ExitTime
                        active_trade.direction,
                        f"{active_trade.entry:.5f}",
                        f"{active_trade.stop_loss:.5f}",
                        f"{active_trade.take_profit:.5f}",
                        f"{active_trade.position_size:.4f}",
                        f"{exit_price:.5f}",
                        exit_type,
                        f"{pnl:.2f}",
                        trade_meta.get("regime", ""),
                        trade_meta.get("ha_bars", ""),
                        trade_meta.get("rsi", ""),
                        trade_meta.get("atr", ""),
                        active_trade.pattern_name,
                        active_trade.reason,
                    ])

                active_trade = None
                trade_meta   = {}
                continue   # skip entry check on the same candle

        # ── B. Open new position ──────────────────────────────────────
        if signal and signal.signal in ("BUY", "SELL"):
            active_trade = risk_engine.build_trade(signal, blc)
            trade_meta   = {
                "regime":  strategy.regime.value,
                "ha_bars": strategy.htf.trend.candle_count if strategy.htf.trend else 0,
                "rsi":     f"{strategy.ltf.rsi:.2f}" if strategy.ltf.rsi else "",
                "atr":     f"{strategy.ltf.atr:.5f}" if strategy.ltf.atr else "",
            }
            trades_count += 1
            print(
                f"   {'🟢' if signal.signal == 'BUY' else '🔴'} "
                f"{signal.signal} | price={candle.close:.2f} "
                f"SL={signal.sl:.2f} TP={signal.tp:.2f} "
                f"regime={strategy.regime.value}"
            )

    # ── 6. Summary ────────────────────────────────────────────────────
    print(f"\n✅ HTF-LTF backtest complete.")
    print(f"   Trades  : {trades_count}")
    print(f"   Balance : {balance:.2f} → {blc:.2f}  (PnL: {blc - balance:+.2f})")

    # ── 7. Results & equity curve ─────────────────────────────────────
    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        print("⚠️  No trades recorded.")
        return df

    _print_stats(df)
    _plot_results(df, symbol)

    return df


# =========================
# Stats & Plotting
# =========================

def _print_stats(df: pd.DataFrame) -> None:
    df["PnL"] = df["PnL"].astype(float)

    wins   = df[df["PnL"] > 0]["PnL"]
    losses = df[df["PnL"] < 0]["PnL"]

    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = len(wins) / len(df) * 100
    avg_win  = wins.mean()  if not wins.empty  else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0

    # Max drawdown from cumulative PnL peak
    cum_pnl  = df["PnL"].cumsum()
    peak     = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd   = drawdown.min()

    # Average R (risk = entry - SL distance; reward = PnL / position_size)
    df["RiskPts"]   = abs(df["Entry"].astype(float) - df["SL"].astype(float))
    df["PnLPerUnit"] = df["PnL"].astype(float) / df["PositionSize"].astype(float)
    df["R"]          = df.apply(
        lambda r: r["PnLPerUnit"] / r["RiskPts"] if r["RiskPts"] > 0 else 0, axis=1
    )
    avg_r = df["R"].mean()

    print("\n─────────────────────────────────────────")
    print(f"  Trades        : {len(df)}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Profit factor : {profit_factor:.2f}")
    print(f"  Avg R         : {avg_r:.2f}R")
    print(f"  Avg win       : {avg_win:.2f}")
    print(f"  Avg loss      : {avg_loss:.2f}")
    print(f"  Max drawdown  : {max_dd:.2f}")
    print("─────────────────────────────────────────")

    # Exit type breakdown
    print("\n  Exit breakdown:")
    for exit_type, count in df["ExitType"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {exit_type:<20} {count:>4}  ({pct:.1f}%)")

    # Per-regime breakdown
    if "Regime" in df.columns:
        print("\n  Per-regime PnL:")
        for regime, grp in df.groupby("Regime"):
            r_wins = (grp["PnL"] > 0).sum()
            print(
                f"    {regime:<14} trades={len(grp):>3}  "
                f"wr={r_wins/len(grp)*100:.0f}%  "
                f"pnl={grp['PnL'].sum():.2f}"
            )
    print()


def _plot_results(df: pd.DataFrame, symbol: str) -> None:
    df["PnL"]           = df["PnL"].astype(float)
    df["CumulativePnL"] = df["PnL"].cumsum()

    cum_pnl  = df["CumulativePnL"]
    peak     = cum_pnl.cummax()
    drawdown = cum_pnl - peak

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    # ── Panel 1: Equity curve ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum_pnl.values, label="Equity curve", linewidth=1.2, color="#2196F3")
    ax1.axhline(0, color="#888", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_title(f"HTF-LTF Backtest — {symbol}", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # Colour individual trade bars by direction
    for idx, row in df.iterrows():
        colour = "#4CAF50" if row["PnL"] > 0 else "#F44336"
        ax1.axvline(idx, color=colour, alpha=0.15, linewidth=0.8)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="#F44336", alpha=0.4)
    ax2.plot(drawdown.values, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================
# Run
# =========================

if __name__ == "__main__":
    backtest_htf_ltf(
        symbol          = SYMBOLS["Gold"],
        htf_timeframe   = "5m",
        ltf_timeframe   = "1m",
        ltf_bars        = 5000,
        htf_bars        = 2000,
        ltf_cutoff      = 150,
        htf_cutoff      = 60,
        balance         = 1000.0,
        use_strategy_close = True,
        use_tp          = True,
        htf_ema_enabled = False,
    )