# market_regime.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Determine whether the overall NEPSE market is in a
# BULL, NEUTRAL, or BEAR regime before scoring individual stocks.
#
# WHY THIS MATTERS:
# Even a fundamentally great stock tends to fall when the whole
# market crashes. A BEAR regime doesn't mean "don't invest ever"
# but it means "be more cautious — raise the bar for what counts
# as a buy signal."
#
# HOW WE DETECT REGIME:
# We use 3 signals and combine them:
#
# 1. INDEX vs 30-DAY MA
#    If NEPSE index > its own 30-day average → market trending up
#    If NEPSE index < its own 30-day average → market trending down
#    This is the primary signal.
#
# 2. BREADTH (% of stocks going up today)
#    If 60%+ of all stocks are rising → broad bull day
#    If 60%+ of all stocks are falling → broad bear day
#    This catches "everything is moving together" days.
#
# 3. TREND MOMENTUM (is the trend accelerating or decelerating?)
#    Compare this week's index to last week's index.
#    If index is rising faster → bull strengthening
#    If index is falling faster → bear strengthening
#
# FINAL REGIME:
#    2–3 bull signals → BULL
#    2–3 bear signals → BEAR
#    Mixed signals    → NEUTRAL
# ─────────────────────────────────────────────────────────────────

import logging
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL = "sqlite:///nepse.db"
engine = create_engine(DB_URL, echo=False)


# ══════════════════════════════════════════════════════════════════
# SIGNAL 1 — INDEX vs 30-DAY MOVING AVERAGE
# ══════════════════════════════════════════════════════════════════
def signal_index_vs_ma(index_df: pd.DataFrame) -> tuple[str, str]:
    """
    Compares today's NEPSE index to its 30-day moving average.

    A moving average smooths out daily noise and shows the underlying
    trend direction. If today's price is above the average of the
    last 30 days, the trend is upward.

    Returns: (signal: "BULL"/"BEAR"/"NEUTRAL", explanation: str)
    """
    if index_df.empty or len(index_df) < 5:
        return "NEUTRAL", "Not enough index data yet (need 5+ days)"

    current = index_df["index_value"].iloc[-1]

    # calculate 30-day MA (or use whatever data we have if less than 30 days)
    ma_period = min(30, len(index_df))
    ma30 = index_df["index_value"].tail(ma_period).mean()

    # 2% threshold — avoids calling BULL/BEAR on tiny fluctuations
    pct_above = ((current - ma30) / ma30) * 100

    if pct_above > 2.0:
        return "BULL", (
            f"Index ({current:.0f}) is {pct_above:.1f}% above "
            f"{ma_period}-day MA ({ma30:.0f}) — uptrend"
        )
    elif pct_above < -2.0:
        return "BEAR", (
            f"Index ({current:.0f}) is {abs(pct_above):.1f}% below "
            f"{ma_period}-day MA ({ma30:.0f}) — downtrend"
        )
    else:
        return "NEUTRAL", (
            f"Index ({current:.0f}) within 2% of "
            f"{ma_period}-day MA ({ma30:.0f}) — no clear trend"
        )


# ══════════════════════════════════════════════════════════════════
# SIGNAL 2 — MARKET BREADTH (% of stocks rising)
# ══════════════════════════════════════════════════════════════════
def signal_breadth(target_date: str) -> tuple[str, str]:
    """
    Calculates what % of stocks are rising vs falling today.

    If 60%+ of all stocks rise on the same day → the whole market
    is moving up together → broad bull signal.

    If 60%+ fall → broad bear signal.

    This is called "market breadth" and it's a key indicator used
    by professional analysts to confirm whether an index move is
    supported by the broader market.
    """
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(CASE WHEN pct_change > 0 THEN 1 END) as gainers,
                COUNT(CASE WHEN pct_change < 0 THEN 1 END) as losers,
                COUNT(CASE WHEN pct_change = 0 THEN 1 END) as unchanged,
                COUNT(*) as total
            FROM daily_prices
            WHERE date = :d
              AND pct_change IS NOT NULL
              AND ltp >= 100
        """), {"d": target_date})
        row = result.fetchone()

    if not row or not row[3] or row[3] == 0:
        return "NEUTRAL", "No price data available for breadth calculation"

    gainers   = row[0] or 0
    losers    = row[1] or 0
    total     = row[3]
    pct_up    = (gainers / total) * 100
    pct_down  = (losers  / total) * 100

    if pct_up >= 60:
        return "BULL", (
            f"Broad advance: {gainers}/{total} stocks rising ({pct_up:.0f}%) "
            f"— strong market breadth"
        )
    elif pct_down >= 60:
        return "BEAR", (
            f"Broad decline: {losers}/{total} stocks falling ({pct_down:.0f}%) "
            f"— weak market breadth"
        )
    else:
        return "NEUTRAL", (
            f"Mixed breadth: {gainers} up, {losers} down out of {total} stocks "
            f"— no dominant direction"
        )


# ══════════════════════════════════════════════════════════════════
# SIGNAL 3 — TREND MOMENTUM (week over week)
# ══════════════════════════════════════════════════════════════════
def signal_momentum(index_df: pd.DataFrame) -> tuple[str, str]:
    """
    Compares this week's index to last week's index.

    Week-over-week momentum tells us if the trend is fresh (just
    started) or exhausted (been going a long time and may reverse).

    We look at 5-day change (approximately 1 trading week in NEPSE
    which trades Sunday–Thursday).
    """
    if len(index_df) < 10:
        return "NEUTRAL", "Not enough index history for momentum (need 10+ days)"

    current    = index_df["index_value"].iloc[-1]
    week_ago   = index_df["index_value"].iloc[-6]   # 5 trading days ago
    two_weeks  = index_df["index_value"].iloc[-11] if len(index_df) >= 11 else None

    weekly_change = ((current - week_ago) / week_ago) * 100

    # also check if momentum is accelerating or decelerating
    if two_weeks is not None:
        prev_weekly_change = ((week_ago - two_weeks) / two_weeks) * 100
        accelerating = weekly_change > prev_weekly_change
    else:
        accelerating = None

    if weekly_change > 1.5:
        accel_str = " (accelerating)" if accelerating else " (decelerating)"
        return "BULL", (
            f"Index up {weekly_change:.1f}% this week"
            + (accel_str if accelerating is not None else "")
        )
    elif weekly_change < -1.5:
        accel_str = " (accelerating down)" if accelerating is False else ""
        return "BEAR", (
            f"Index down {abs(weekly_change):.1f}% this week"
            + accel_str
        )
    else:
        return "NEUTRAL", f"Index changed {weekly_change:+.1f}% this week — minimal movement"


# ══════════════════════════════════════════════════════════════════
# COMBINE SIGNALS → FINAL REGIME
# ══════════════════════════════════════════════════════════════════
def determine_regime(target_date: str = None) -> dict:
    """
    Combines all 3 signals using majority vote.
    2 or 3 BULL signals → BULL
    2 or 3 BEAR signals → BEAR
    Otherwise          → NEUTRAL

    Returns a full dict with regime + all signal details for display.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    # load index history
    cutoff = (date.today() - timedelta(days=60)).isoformat()
    with engine.connect() as conn:
        index_df = pd.read_sql(text("""
            SELECT date, index_value FROM market_index
            WHERE date >= :cutoff ORDER BY date ASC
        """), conn, params={"cutoff": cutoff})

    # run all 3 signals
    s1_regime, s1_desc = signal_index_vs_ma(index_df)
    s2_regime, s2_desc = signal_breadth(target_date)
    s3_regime, s3_desc = signal_momentum(index_df)

    signals = [s1_regime, s2_regime, s3_regime]

    # majority vote
    bull_count = signals.count("BULL")
    bear_count = signals.count("BEAR")

    if bull_count >= 2:
        final_regime = "BULL"
    elif bear_count >= 2:
        final_regime = "BEAR"
    else:
        final_regime = "NEUTRAL"

    result = {
        "regime":        final_regime,
        "date":          target_date,
        "bull_signals":  bull_count,
        "bear_signals":  bear_count,
        "signal1_regime": s1_regime,
        "signal1_desc":   s1_desc,
        "signal2_regime": s2_regime,
        "signal2_desc":   s2_desc,
        "signal3_regime": s3_regime,
        "signal3_desc":   s3_desc,
        "summary": (
            f"NEPSE market is in {final_regime} regime "
            f"({bull_count} bull signals, {bear_count} bear signals out of 3)"
        )
    }

    logger.info(f"Market regime: {final_regime} "
                f"(bull:{bull_count} bear:{bear_count})")
    return result


# ══════════════════════════════════════════════════════════════════
# SAVE REGIME TO DATABASE (optional — for history tracking)
# ══════════════════════════════════════════════════════════════════
def save_regime(regime_data: dict):
    """Saves today's regime to market_index table as a new column."""
    with engine.connect() as conn:
        # add regime column if it doesn't exist
        try:
            conn.execute(text(
                "ALTER TABLE market_index ADD COLUMN regime TEXT"
            ))
            conn.commit()
        except Exception:
            pass  # column already exists

        conn.execute(text("""
            UPDATE market_index
            SET regime = :regime
            WHERE date = :date
        """), {"regime": regime_data["regime"], "date": regime_data["date"]})
        conn.commit()


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("Analysing NEPSE market regime...\n")
    regime_data = determine_regime()
    save_regime(regime_data)

    regime = regime_data["regime"]
    icon   = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(regime, "🟡")

    print(f"{'='*60}")
    print(f"MARKET REGIME: {icon} {regime}")
    print(f"{'='*60}")
    print(f"\n{regime_data['summary']}")
    print(f"\nSignal breakdown:")
    print(f"  Signal 1 (Index vs MA):  {regime_data['signal1_regime']:<8} — {regime_data['signal1_desc']}")
    print(f"  Signal 2 (Breadth):      {regime_data['signal2_regime']:<8} — {regime_data['signal2_desc']}")
    print(f"  Signal 3 (Momentum):     {regime_data['signal3_regime']:<8} — {regime_data['signal3_desc']}")

    print(f"\n{'='*60}")
    if regime == "BEAR":
        print("⚠️  BEAR market detected — scorer will suppress STRONG BUY signals")
        print("   Stocks need higher scores to qualify as WATCH instead")
    elif regime == "BULL":
        print("✅ BULL market — regime bonus (+1 pt) applied to all stock scores")
    else:
        print("🟡 NEUTRAL market — no regime adjustment to scores")