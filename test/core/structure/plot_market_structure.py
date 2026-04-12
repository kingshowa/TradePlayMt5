import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_market_context(series, context, context_state, title="Market Context"):
    candles = series._candles

    fig, ax = plt.subplots(figsize=(16, 8))

    candle_width = 0.6

    # --------------------------------------------------
    # Candles
    # --------------------------------------------------
    for i, candle in enumerate(candles):
        open_price = candle.open
        close_price = candle.close
        high_price = candle.high
        low_price = candle.low

        color = "green" if close_price >= open_price else "red"

        # Wick
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)

        # Body
        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)
        if body_height == 0:
            body_height = 0.01

        rect = Rectangle(
            (i - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color
        )
        ax.add_patch(rect)

    # --------------------------------------------------
    # Swings from SwingPoint
    # --------------------------------------------------
    swing_highs = list(context.swing_highs)
    swing_lows = list(context.swing_lows)

    swing_high_x = [sp.index for sp in swing_highs]
    swing_high_y = [sp.price for sp in swing_highs]

    swing_low_x = [sp.index for sp in swing_lows]
    swing_low_y = [sp.price for sp in swing_lows]

    if swing_high_x:
        ax.scatter(swing_high_x, swing_high_y, marker="^", s=80, label="Swing Highs")

    if swing_low_x:
        ax.scatter(swing_low_x, swing_low_y, marker="v", s=80, label="Swing Lows")

    # --------------------------------------------------
    # Swing Labels
    # --------------------------------------------------
    for sp in swing_highs:
        if sp.label:
            ax.text(
                sp.index,
                sp.price,
                f" {sp.label}",
                fontsize=9,
                va="bottom",
                ha="left"
            )

    for sp in swing_lows:
        if sp.label:
            ax.text(
                sp.index,
                sp.price,
                f" {sp.label}",
                fontsize=9,
                va="top",
                ha="left"
            )

    # --------------------------------------------------
    # Support Zone
    # --------------------------------------------------
    if context_state.support_zone:
        s_low, s_high = context_state.support_zone
        ax.axhspan(s_low, s_high, color="blue", alpha=0.2, label="Support")

    # --------------------------------------------------
    # Resistance Zone
    # --------------------------------------------------
    if context_state.resistance_zone:
        r_low, r_high = context_state.resistance_zone
        ax.axhspan(r_low, r_high, color="purple", alpha=0.2, label="Resistance")

    # --------------------------------------------------
    # Last swing levels
    # --------------------------------------------------
    if context_state.last_swing_high is not None:
        ax.axhline(
            context_state.last_swing_high,
            linestyle="--",
            color="purple",
            alpha=0.7,
            label="Last Swing High"
        )

    if context_state.last_swing_low is not None:
        ax.axhline(
            context_state.last_swing_low,
            linestyle="--",
            color="blue",
            alpha=0.7,
            label="Last Swing Low"
        )

    # --------------------------------------------------
    # Protected structure levels
    # --------------------------------------------------
    if getattr(context_state, "protected_high", None) is not None:
        ax.axhline(
            context_state.protected_high,
            linestyle=":",
            color="magenta",
            alpha=0.9,
            label="Protected High"
        )

    if getattr(context_state, "protected_low", None) is not None:
        ax.axhline(
            context_state.protected_low,
            linestyle=":",
            color="cyan",
            alpha=0.9,
            label="Protected Low"
        )

    # --------------------------------------------------
    # Current candle marker
    # --------------------------------------------------
    if context.candles:
        last_idx = len(context.candles) - 1
        last_close = context.candles[-1].close
        ax.scatter([last_idx], [last_close], s=100, marker="o", label="Current Close")

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    structure_event = getattr(context_state, "structure_event", "NONE")

    ax.set_title(
        f"{title} | Trend={context_state.trend} | "
        f"Event={structure_event} | "
        f"Strength={context_state.trend_strength}"
    )

    ax.set_xlabel("Candle Index")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    # Remove duplicate legend items
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()