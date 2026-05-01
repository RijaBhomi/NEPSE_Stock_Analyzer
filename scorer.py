# scorer.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: The "Brain" — scores each stock 0–12 and assigns a signal.
#
# DESIGN DECISION: Since NEPSE sites load P/E and EPS via JavaScript
# (which simple requests scraping can't get), we score based on:
#
# What we HAVE:
#   - Price (LTP)
#   - % change today
#   - Volume
#   - 52-week high/low (from merolagani live table)
#   - Sector
#   - Moving averages (once we have 7+ days of data)
#   - RSI (once we have 14+ days of data)
#
# The 6 scoring rules (revised for available data):
#
# RULE 1 — PRICE SANITY (gate rule)
#   Stock must be above Rs 100 and not in freefall
#   This is already enforced by validator — stocks that fail don't reach scorer
#
# RULE 2 — 52-WEEK POSITION (value signal)
#   How close is the current price to its 52-week LOW vs HIGH?
#   Close to 52w low = cheap relative to its own history = potential value
#   Formula: position = (LTP - 52w_low) / (52w_high - 52w_low)
#   0.0 = at 52-week low (very cheap), 1.0 = at 52-week high (expensive)
#
# RULE 3 — RSI SIGNAL (momentum signal)
#   RSI under 30 AND rising = oversold + recovering = strong buy signal
#   Only available after 14+ days of data
#
# RULE 4 — PRICE vs MA7 (short-term trend)
#   Price above MA7 = short-term upward momentum (stabilising)
#   Only available after 7+ days of data
#
# RULE 5 — VOLUME SIGNAL (interest signal)
#   High volume on a down day = selling pressure (bad)
#   High volume on an up day = buying interest (good)
#   Compare today's volume to the stock's average volume
#
# RULE 6 — MARKET REGIME BONUS
#   If overall NEPSE market is healthy, add a bonus point
#   If NEPSE is in bear trend, reduce maximum possible score
# ─────────────────────────────────────────────────────────────────

import logging
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL = "sqlite:///nepse.db"
engine = create_engine(DB_URL, echo=False)

# sector P/E medians — used for sector context labels
SECTOR_PE_MEDIANS = {
    "Commercial Bank": 12.0,
    "Development Bank": 14.0,
    "Finance":         13.0,
    "Hydropower":      25.0,
    "Insurance":       20.0,
    "Hotels":          30.0,
    "Manufacturing":   18.0,
    "Telecom":         15.0,
    "Microfinance":    16.0,
    "Others":          20.0,
}


# ══════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════
def create_scores_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                date            TEXT NOT NULL,
                company_name    TEXT,
                sector          TEXT,
                ltp             REAL,
                pct_change      REAL,
                total_score     REAL,
                signal          TEXT,
                rule2_points    REAL,
                rule3_points    REAL,
                rule4_points    REAL,
                rule5_points    REAL,
                rule6_points    REAL,
                week52_position REAL,
                rsi             REAL,
                data_quality    TEXT,
                plain_english   TEXT,
                UNIQUE(symbol, date)
            )
        """))
        conn.commit()


# ══════════════════════════════════════════════════════════════════
# RULE 2 — 52-WEEK POSITION
# ══════════════════════════════════════════════════════════════════
def score_52week_position(ltp, week52_high, week52_low) -> tuple[float, str]:
    """
    Calculates where the current price sits in its 52-week range.

    Position = (LTP - 52w_low) / (52w_high - 52w_low)
    0.0 = at the 52-week low (cheapest it's been all year)
    1.0 = at the 52-week high (most expensive all year)
    0.3 = in the bottom 30% of its yearly range

    Scoring:
    Position 0.0–0.20 → +3 pts  (bottom 20% of range = very cheap)
    Position 0.20–0.35 → +2 pts (lower third = cheap)
    Position 0.35–0.50 → +1 pt  (middle = neutral-ish)
    Position above 0.50 → 0 pts (upper half = not a buy signal)
    """
    if not all([ltp, week52_high, week52_low]):
        return 0, "No 52-week data available"

    if week52_high <= week52_low:
        return 0, "Invalid 52-week range data"

    range_size = week52_high - week52_low
    position   = (ltp - week52_low) / range_size
    position   = max(0.0, min(1.0, position))  # clamp to 0–1

    if position <= 0.20:
        return 3, f"Price in bottom 20% of 52-week range (position: {position:.0%})"
    elif position <= 0.35:
        return 2, f"Price in lower third of 52-week range (position: {position:.0%})"
    elif position <= 0.50:
        return 1, f"Price in middle of 52-week range (position: {position:.0%})"
    else:
        return 0, f"Price in upper half of 52-week range (position: {position:.0%}) — not a value signal"


# ══════════════════════════════════════════════════════════════════
# RULE 3 — RSI SIGNAL
# ══════════════════════════════════════════════════════════════════
def score_rsi(rsi, rsi_trend) -> tuple[float, str]:
    """
    RSI scoring with trend confirmation to avoid the RSI trap.

    RSI < 30 AND RISING  → +3 pts  (oversold AND recovering)
    RSI < 30 AND NEUTRAL → +2 pts  (oversold but unclear direction)
    RSI < 30 AND FALLING → +0 pts  (oversold but still getting worse — trap!)
    RSI 30–45            → +1 pt   (cooling down, potential opportunity)
    RSI 45–60            → 0 pts   (neutral)
    RSI > 60             → 0 pts   (overbought territory — don't chase)
    RSI unavailable      → 0 pts   (not enough history yet)
    """
    if rsi is None:
        return 0, "RSI not yet available (need 14+ days of data)"

    if rsi < 30:
        if rsi_trend == "RISING":
            return 3, f"RSI {rsi:.1f} — oversold AND recovering ✅ Strong signal"
        elif rsi_trend == "FALLING":
            return 0, f"RSI {rsi:.1f} — oversold but STILL FALLING ⚠️ Avoid (RSI trap)"
        else:
            return 2, f"RSI {rsi:.1f} — oversold, direction unclear"
    elif rsi < 45:
        return 1, f"RSI {rsi:.1f} — cooling down from oversold territory"
    elif rsi < 60:
        return 0, f"RSI {rsi:.1f} — neutral zone"
    else:
        return 0, f"RSI {rsi:.1f} — overbought territory, not a buy signal"


# ══════════════════════════════════════════════════════════════════
# RULE 4 — PRICE vs MA7 (short-term stabilisation)
# ══════════════════════════════════════════════════════════════════
def score_ma_position(price_vs_ma7, price_vs_ma200,
                      days_below_ma200) -> tuple[float, str]:
    """
    Scores based on where price sits relative to moving averages.

    Price ABOVE MA7 = short-term upward momentum (good)
    Price BELOW MA200 but recently (< 60 days) = dip opportunity
    Price BELOW MA200 for 90+ days = extended downtrend (bad)

    MA200 is most important but needs 200 days of data.
    MA7 is available sooner (7 days).
    """
    points = 0
    reasons = []

    # MA7 signal — short term stabilisation
    if price_vs_ma7 == "ABOVE":
        points += 1
        reasons.append("Price above 7-day average (short-term stable)")
    elif price_vs_ma7 == "BELOW":
        reasons.append("Price below 7-day average (short-term weak)")
    else:
        reasons.append("MA7 not yet available")

    # MA200 signal — long term value
    if price_vs_ma200 == "BELOW" and days_below_ma200 is not None:
        if days_below_ma200 < 60:
            points += 2
            reasons.append(
                f"Price below 200-day MA for only {days_below_ma200} days "
                f"— recent dip, possible buy opportunity"
            )
        elif days_below_ma200 < 90:
            points += 1
            reasons.append(
                f"Price below 200-day MA for {days_below_ma200} days "
                f"— caution, extended dip"
            )
        else:
            reasons.append(
                f"Price below 200-day MA for {days_below_ma200} days "
                f"— extended downtrend ⚠️"
            )
    elif price_vs_ma200 == "ABOVE":
        reasons.append("Price above 200-day MA (long-term strong)")
    else:
        reasons.append("MA200 not yet available (need 200 days of data)")

    return points, " | ".join(reasons)


# ══════════════════════════════════════════════════════════════════
# RULE 5 — VOLUME SIGNAL
# ══════════════════════════════════════════════════════════════════
def score_volume(volume, pct_change, symbol) -> tuple[float, str]:
    """
    Volume interpretation:
    High volume on UP day = strong buying interest = good signal
    High volume on DOWN day = heavy selling = bad signal
    Low volume = stock being ignored = neutral

    We compare today's volume to the stock's 30-day average volume.
    "High volume" = more than 1.5x the average.
    """
    if not volume or not pct_change:
        return 0, "No volume data"

    # get average volume for this stock from past 30 days
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT AVG(volume) FROM daily_prices
            WHERE symbol = :symbol
              AND volume IS NOT NULL
              AND volume > 0
            ORDER BY date DESC
            LIMIT 30
        """), {"symbol": symbol})
        row = result.fetchone()
        avg_volume = row[0] if row and row[0] else None

    if not avg_volume or avg_volume == 0:
        return 0, "Insufficient volume history"

    volume_ratio = volume / avg_volume

    if volume_ratio >= 1.5 and pct_change > 0:
        return 2, f"High volume ({volume_ratio:.1f}x avg) on UP day — strong buying interest 🟢"
    elif volume_ratio >= 1.5 and pct_change < 0:
        return 0, f"High volume ({volume_ratio:.1f}x avg) on DOWN day — heavy selling ⚠️"
    elif volume_ratio >= 1.0 and pct_change > 0:
        return 1, f"Normal-high volume on UP day — moderate buying interest"
    else:
        return 0, f"Volume ratio {volume_ratio:.1f}x avg — no strong signal"


# ══════════════════════════════════════════════════════════════════
# RULE 6 — MARKET REGIME BONUS
# ══════════════════════════════════════════════════════════════════
def get_market_regime() -> str:
    """
    Gets the current NEPSE market regime from the database.
    Returns BULL, NEUTRAL, or BEAR.
    If no index data available, defaults to NEUTRAL.
    """
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT index_value FROM market_index
            ORDER BY date DESC LIMIT 30
        """))
        rows = result.fetchall()

    if len(rows) < 5:
        return "NEUTRAL"  # not enough data yet

    values = [r[0] for r in rows if r[0]]
    if not values:
        return "NEUTRAL"

    current = values[0]
    avg30   = sum(values) / len(values)

    if current > avg30 * 1.02:
        return "BULL"
    elif current < avg30 * 0.98:
        return "BEAR"
    return "NEUTRAL"


def score_market_regime(regime: str) -> tuple[float, str]:
    """
    Market regime bonus/penalty.
    BULL:    +1 pt  (rising tide lifts all boats)
    NEUTRAL:  0 pts (no adjustment)
    BEAR:     0 pts (suppress optimism, no bonus)
    """
    if regime == "BULL":
        return 1, "NEPSE market in uptrend — regime bonus +1"
    elif regime == "BEAR":
        return 0, "NEPSE market in downtrend — no regime bonus ⚠️"
    return 0, "NEPSE market neutral"


# ══════════════════════════════════════════════════════════════════
# GENERATE PLAIN ENGLISH EXPLANATION
# ══════════════════════════════════════════════════════════════════
def generate_plain_english(symbol, sector, ltp, total_score,
                           signal, rule_reasons: dict) -> str:
    """
    Converts the score breakdown into a plain English explanation
    that a beginner investor can understand.
    """
    signal_text = {
        "STRONG BUY": "This stock looks like an interesting opportunity based on the data.",
        "WATCH":      "This stock shows some positive signals. Worth monitoring.",
        "NEUTRAL":    "This stock shows mixed signals. No strong reason to buy right now.",
        "AVOID":      "This stock does not meet our screening criteria right now.",
    }.get(signal, "")

    lines = [f"{signal_text}"]

    if rule_reasons.get("r2"):
        lines.append(f"• 52-week position: {rule_reasons['r2']}")
    if rule_reasons.get("r3"):
        lines.append(f"• Momentum (RSI): {rule_reasons['r3']}")
    if rule_reasons.get("r4"):
        lines.append(f"• Trend: {rule_reasons['r4']}")
    if rule_reasons.get("r5"):
        lines.append(f"• Volume: {rule_reasons['r5']}")
    if rule_reasons.get("r6"):
        lines.append(f"• Market: {rule_reasons['r6']}")

    lines.append(
        "\n⚠️ This is a data-driven screening tool for educational purposes only. "
        "It is NOT financial advice. Always do your own research before investing."
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# MASTER SCORER — score one stock
# ══════════════════════════════════════════════════════════════════
def score_stock(stock: dict, regime: str = "NEUTRAL") -> dict | None:
    """
    Runs all scoring rules on a single stock and returns
    a complete score record ready to save to the database.
    """
    symbol = stock.get("symbol")
    ltp    = stock.get("ltp")

    if not symbol or not ltp:
        return None

    # load indicator data for this stock
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT rsi, rsi_trend, price_vs_ma7, price_vs_ma200,
                   days_below_ma200, ma200_trend
            FROM indicators
            WHERE symbol = :symbol
            ORDER BY date DESC LIMIT 1
        """), {"symbol": symbol})
        ind_row = result.fetchone()

    rsi              = ind_row[0] if ind_row else None
    rsi_trend        = ind_row[1] if ind_row else None
    price_vs_ma7     = ind_row[2] if ind_row else None
    price_vs_ma200   = ind_row[3] if ind_row else None
    days_below_ma200 = ind_row[4] if ind_row else None

    # ── run all rules ──────────────────────────────────────────
    r2_pts, r2_reason = score_52week_position(
        ltp, stock.get("week52_high"), stock.get("week52_low")
    )
    r3_pts, r3_reason = score_rsi(rsi, rsi_trend)
    r4_pts, r4_reason = score_ma_position(
        price_vs_ma7, price_vs_ma200, days_below_ma200
    )
    r5_pts, r5_reason = score_volume(
        stock.get("volume"), stock.get("pct_change"), symbol
    )
    r6_pts, r6_reason = score_market_regime(regime)

    # ── total score ────────────────────────────────────────────
    # maximum possible: 3+3+3+2+1 = 12
    total = r2_pts + r3_pts + r4_pts + r5_pts + r6_pts

    # ── signal ─────────────────────────────────────────────────
    if total >= 7:
        signal = "STRONG BUY"
    elif total >= 5:
        signal = "WATCH"
    elif total >= 3:
        signal = "NEUTRAL"
    else:
        signal = "AVOID"

    # suppress STRONG BUY if market is in BEAR regime
    if regime == "BEAR" and signal == "STRONG BUY":
        signal = "WATCH"

    # ── plain english explanation ──────────────────────────────
    rule_reasons = {
        "r2": r2_reason, "r3": r3_reason,
        "r4": r4_reason, "r5": r5_reason, "r6": r6_reason,
    }
    explanation = generate_plain_english(
        symbol, stock.get("sector", "Unknown"), ltp,
        total, signal, rule_reasons
    )

    # ── 52-week position for display ───────────────────────────
    w52_high = stock.get("week52_high")
    w52_low  = stock.get("week52_low")
    if w52_high and w52_low and w52_high > w52_low:
        position = (ltp - w52_low) / (w52_high - w52_low)
    else:
        position = None

    return {
        "symbol":          symbol,
        "date":            date.today().isoformat(),
        "company_name":    stock.get("company_name", ""),
        "sector":          stock.get("sector", "Others"),
        "ltp":             ltp,
        "pct_change":      stock.get("pct_change"),
        "total_score":     round(total, 1),
        "signal":          signal,
        "rule2_points":    r2_pts,
        "rule3_points":    r3_pts,
        "rule4_points":    r4_pts,
        "rule5_points":    r5_pts,
        "rule6_points":    r6_pts,
        "week52_position": round(position, 3) if position else None,
        "rsi":             rsi,
        "data_quality":    stock.get("data_quality", "PARTIAL"),
        "plain_english":   explanation,
    }


# ══════════════════════════════════════════════════════════════════
# RUN SCORER ON ALL VERIFIED STOCKS
# ══════════════════════════════════════════════════════════════════
def run_scorer() -> list:
    """
    Loads all verified stocks for today, scores them all,
    saves results to database, returns sorted list.
    """
    create_scores_table()

    today   = date.today().isoformat()
    regime  = get_market_regime()
    logger.info(f"Market regime: {regime}")

    # load today's verified stocks
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT p.symbol, p.ltp, p.pct_change, p.volume,
                   p.week52_high, p.week52_low, p.pe_ratio, p.eps,
                   p.scraped_at,
                   s.company_name, s.sector
            FROM daily_prices p
            LEFT JOIN stocks s ON p.symbol = s.symbol
            WHERE p.date = :today
              AND p.ltp >= 100
              AND p.ltp IS NOT NULL
        """), {"today": today})
        stocks = [dict(row._mapping) for row in result]

    if not stocks:
        logger.warning(f"No stocks found for {today}")
        return []

    logger.info(f"Scoring {len(stocks)} stocks...")
    scored  = []
    saved   = 0

    with engine.connect() as conn:
        for stock in stocks:
            result = score_stock(stock, regime)
            if not result:
                continue

            scored.append(result)

            # save to database
            try:
                conn.execute(text("""
                    INSERT OR REPLACE INTO scores
                        (symbol, date, company_name, sector, ltp, pct_change,
                         total_score, signal, rule2_points, rule3_points,
                         rule4_points, rule5_points, rule6_points,
                         week52_position, rsi, data_quality, plain_english)
                    VALUES
                        (:symbol, :date, :company_name, :sector, :ltp,
                         :pct_change, :total_score, :signal, :rule2_points,
                         :rule3_points, :rule4_points, :rule5_points,
                         :rule6_points, :week52_position, :rsi,
                         :data_quality, :plain_english)
                """), result)
                saved += 1
            except Exception as e:
                logger.debug(f"Error saving score for {result['symbol']}: {e}")

        conn.commit()

    # sort by score descending
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    logger.info(f"Scored {len(scored)} stocks, saved {saved} to database")
    return scored


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("Running scorer...\n")
    results = run_scorer()

    if not results:
        print("No results — run the scraper first: python pipeline.py --once")
    else:
        print(f"\n{'='*70}")
        print(f"TODAY'S TOP GEMS — {date.today()} ({len(results)} stocks scored)")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Symbol':<10} {'LTP':>8} {'Chg%':>7} {'Score':>6} {'Signal':<12} {'Sector'}")
        print(f"{'-'*70}")

        for i, s in enumerate(results[:20], 1):
            chg = f"{s['pct_change']:+.2f}%" if s['pct_change'] else "N/A"
            print(
                f"{i:<4} {s['symbol']:<10} "
                f"{s['ltp']:>8.1f} {chg:>7} "
                f"{s['total_score']:>6.1f} {s['signal']:<12} "
                f"{s['sector'] or 'Unknown'}"
            )

        print(f"\n{'='*70}")
        print("SIGNAL BREAKDOWN:")
        for sig in ["STRONG BUY", "WATCH", "NEUTRAL", "AVOID"]:
            count = sum(1 for s in results if s["signal"] == sig)
            print(f"  {sig:<12}: {count} stocks")

        # show full explanation for top stock
        if results:
            top = results[0]
            print(f"\n{'='*70}")
            print(f"DEEP DIVE: {top['symbol']} (Score: {top['total_score']}/12)")
            print(f"{'='*70}")
            print(top["plain_english"])