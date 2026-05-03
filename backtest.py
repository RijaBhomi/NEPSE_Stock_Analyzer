# backtest.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Validate the scoring system using historical data.
#
# THE CORE QUESTION:
# "If I had bought the stocks your system flagged as STRONG BUY
# 30 days ago, would I have made money?"
#
# HOW BACKTESTING WORKS:
# 1. Pick a past date (e.g. 30 days ago)
# 2. Run our scorer on data from THAT date (pretend it's the past)
# 3. Look at what those stocks actually did over the NEXT 30 days
# 4. Calculate: did STRONG BUY picks outperform the NEPSE index?
#
# KEY METRICS WE CALCULATE:
#
# HIT RATE:
#   Of all stocks we called STRONG BUY, what % actually went up
#   in the next 30 days? 50% = random. 60%+ = our system adds value.
#
# ALPHA:
#   Did our picks outperform the NEPSE index return?
#   If NEPSE went up 3% and our picks went up 5% → alpha = +2%
#   Positive alpha = our screening adds value beyond just "buy everything"
#
# AVERAGE RETURN:
#   Mean price change of our STRONG BUY picks over 30 days.
#
# NOTE ON DATA LIMITATIONS:
# We just started collecting data so we won't have 30-day forward
# data yet. The backtest will show "insufficient data" until we
# have 30+ days of history. This is honest and correct.
# After 30 days of daily scraping, this becomes the most impressive
# part of the project — proof that the system actually works.
# ─────────────────────────────────────────────────────────────────

import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL = "sqlite:///nepse.db"
engine = create_engine(DB_URL, echo=False)

# how many days forward to check performance
HOLDING_PERIOD_DAYS = 30


# ══════════════════════════════════════════════════════════════════
# LOAD HISTORICAL SCORES FOR A PAST DATE
# ══════════════════════════════════════════════════════════════════
def load_scores_for_date(target_date: str) -> pd.DataFrame:
    """
    Loads scores that were generated on a specific past date.
    These are the "buy recommendations" we made on that day.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT symbol, signal, total_score, ltp as entry_price, sector
            FROM scores
            WHERE date = :d
        """), conn, params={"d": target_date})
    return df


# ══════════════════════════════════════════════════════════════════
# GET PRICE N DAYS AFTER ENTRY
# ══════════════════════════════════════════════════════════════════
def get_exit_price(symbol: str, entry_date: str,
                   holding_days: int) -> float | None:
    """
    Gets the closing price of a stock N trading days after the entry date.

    We look for the closest available price on or after the target date
    because NEPSE doesn't trade every day (weekends, holidays).
    """
    # calculate approximate exit date
    entry_dt   = date.fromisoformat(entry_date)
    target_dt  = entry_dt + timedelta(days=holding_days)
    target_str = target_dt.isoformat()

    with engine.connect() as conn:
        # get the first available price on or after target date
        result = conn.execute(text("""
            SELECT ltp FROM daily_prices
            WHERE symbol = :sym
              AND date >= :target
              AND ltp IS NOT NULL
            ORDER BY date ASC
            LIMIT 1
        """), {"sym": symbol, "target": target_str})
        row = result.fetchone()

    return float(row[0]) if row else None


# ══════════════════════════════════════════════════════════════════
# GET NEPSE INDEX RETURN OVER SAME PERIOD
# ══════════════════════════════════════════════════════════════════
def get_index_return(entry_date: str, holding_days: int) -> float | None:
    """
    Gets the NEPSE index return over the same holding period.
    This is our benchmark — we need to beat this to add value.

    If we can't beat just "buying the whole market", our screener
    doesn't add value.
    """
    target_dt  = (date.fromisoformat(entry_date) +
                  timedelta(days=holding_days)).isoformat()

    with engine.connect() as conn:
        # entry index value
        r1 = conn.execute(text("""
            SELECT index_value FROM market_index
            WHERE date >= :d ORDER BY date ASC LIMIT 1
        """), {"d": entry_date})
        entry_row = r1.fetchone()

        # exit index value
        r2 = conn.execute(text("""
            SELECT index_value FROM market_index
            WHERE date >= :d ORDER BY date ASC LIMIT 1
        """), {"d": target_dt})
        exit_row = r2.fetchone()

    if not entry_row or not exit_row:
        return None

    entry_idx = float(entry_row[0])
    exit_idx  = float(exit_row[0])

    if entry_idx == 0:
        return None

    return ((exit_idx - entry_idx) / entry_idx) * 100


# ══════════════════════════════════════════════════════════════════
# RUN BACKTEST FOR ONE DATE
# ══════════════════════════════════════════════════════════════════
def backtest_single_date(entry_date: str,
                          holding_days: int = HOLDING_PERIOD_DAYS) -> dict:
    """
    Runs the full backtest for a single entry date.
    Returns a dict with all performance metrics.
    """
    scores_df = load_scores_for_date(entry_date)

    if scores_df.empty:
        return {
            "entry_date":   entry_date,
            "status":       "NO_DATA",
            "message":      f"No scores found for {entry_date}"
        }

    # check if we have exit price data yet
    test_symbol = scores_df["symbol"].iloc[0]
    test_exit   = get_exit_price(test_symbol, entry_date, holding_days)

    if test_exit is None:
        days_to_wait = holding_days - (date.today() -
                       date.fromisoformat(entry_date)).days
        return {
            "entry_date":   entry_date,
            "status":       "WAITING",
            "message":      (
                f"Need {days_to_wait} more days of data "
                f"(holding period: {holding_days} days)"
            )
        }

    # calculate returns for all stocks
    results = []
    for _, row in scores_df.iterrows():
        exit_price = get_exit_price(row["symbol"], entry_date, holding_days)
        if exit_price is None or row["entry_price"] is None:
            continue
        if row["entry_price"] == 0:
            continue

        pct_return = ((exit_price - row["entry_price"]) /
                      row["entry_price"]) * 100

        results.append({
            "symbol":      row["symbol"],
            "signal":      row["signal"],
            "score":       row["total_score"],
            "sector":      row.get("sector", "Unknown"),
            "entry_price": row["entry_price"],
            "exit_price":  exit_price,
            "pct_return":  round(pct_return, 2),
            "went_up":     pct_return > 0,
        })

    if not results:
        return {
            "entry_date": entry_date,
            "status":     "INSUFFICIENT",
            "message":    "Could not calculate returns — missing exit prices"
        }

    df = pd.DataFrame(results)

    # ── overall metrics ────────────────────────────────────────
    all_avg_return = df["pct_return"].mean()
    index_return   = get_index_return(entry_date, holding_days)

    # ── per-signal metrics ─────────────────────────────────────
    signal_stats = {}
    for signal in ["STRONG BUY", "WATCH", "NEUTRAL", "AVOID"]:
        sig_df = df[df["signal"] == signal]
        if sig_df.empty:
            continue

        hit_rate   = sig_df["went_up"].mean() * 100
        avg_return = sig_df["pct_return"].mean()
        count      = len(sig_df)

        signal_stats[signal] = {
            "count":      count,
            "hit_rate":   round(hit_rate, 1),
            "avg_return": round(avg_return, 2),
        }

    # ── strong buy vs market comparison ───────────────────────
    sb_df = df[df["signal"] == "STRONG BUY"]
    if not sb_df.empty and index_return is not None:
        alpha = sb_df["pct_return"].mean() - index_return
    else:
        alpha = None

    return {
        "entry_date":    entry_date,
        "exit_date":     (date.fromisoformat(entry_date) +
                          timedelta(days=holding_days)).isoformat(),
        "holding_days":  holding_days,
        "status":        "COMPLETE",
        "total_stocks":  len(df),
        "all_avg_return": round(all_avg_return, 2),
        "index_return":   round(index_return, 2) if index_return else None,
        "alpha":          round(alpha, 2) if alpha else None,
        "signal_stats":   signal_stats,
        "top_picks":      sb_df.nlargest(5, "pct_return")[
                              ["symbol","entry_price","exit_price","pct_return"]
                          ].to_dict("records") if not sb_df.empty else [],
        "worst_picks":    sb_df.nsmallest(3, "pct_return")[
                              ["symbol","entry_price","exit_price","pct_return"]
                          ].to_dict("records") if not sb_df.empty else [],
        "all_results":    df,
    }


# ══════════════════════════════════════════════════════════════════
# RUN BACKTEST ACROSS MULTIPLE DATES
# ══════════════════════════════════════════════════════════════════
def run_full_backtest(holding_days: int = HOLDING_PERIOD_DAYS) -> list:
    """
    Runs backtest for every date we have score data for.
    Only dates where exit price data is also available will show results.
    Others will show WAITING status.
    """
    # get all dates we have scores for
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT date FROM scores ORDER BY date ASC
        """))
        score_dates = [row[0] for row in result]

    if not score_dates:
        logger.warning("No score history found — run scorer.py first")
        return []

    logger.info(f"Running backtest for {len(score_dates)} dates...")
    all_results = []

    for score_date in score_dates:
        result = backtest_single_date(score_date, holding_days)
        all_results.append(result)
        logger.info(
            f"  {score_date}: {result['status']}"
            + (f" — alpha: {result.get('alpha','')}%" if result.get('alpha') else "")
        )

    return all_results


# ══════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS ACROSS ALL COMPLETE BACKTESTS
# ══════════════════════════════════════════════════════════════════
def backtest_summary(all_results: list) -> dict:
    """
    Aggregates results across all complete backtest dates.
    Returns overall hit rate, average alpha, and system credibility score.
    """
    complete = [r for r in all_results if r["status"] == "COMPLETE"]

    if not complete:
        return {
            "status":  "INSUFFICIENT_DATA",
            "message": (
                f"Need at least {HOLDING_PERIOD_DAYS} days of data "
                f"for backtest to complete. Keep running the pipeline daily!"
            ),
            "days_collected": len(all_results),
            "days_needed":    HOLDING_PERIOD_DAYS,
        }

    # aggregate STRONG BUY performance across all complete dates
    all_sb_returns = []
    all_alphas     = []
    all_hit_rates  = []

    for r in complete:
        sb_stats = r["signal_stats"].get("STRONG BUY", {})
        if sb_stats:
            all_hit_rates.append(sb_stats["hit_rate"])
            all_sb_returns.append(sb_stats["avg_return"])
        if r.get("alpha") is not None:
            all_alphas.append(r["alpha"])

    avg_hit_rate  = np.mean(all_hit_rates)  if all_hit_rates  else None
    avg_return    = np.mean(all_sb_returns) if all_sb_returns else None
    avg_alpha     = np.mean(all_alphas)     if all_alphas     else None

    # credibility rating
    if avg_hit_rate and avg_alpha:
        if avg_hit_rate >= 60 and avg_alpha > 2:
            credibility = "HIGH — system consistently outperforms market"
        elif avg_hit_rate >= 55 or avg_alpha > 0:
            credibility = "MODERATE — system shows positive edge"
        else:
            credibility = "LOW — system needs improvement"
    else:
        credibility = "PENDING — more data needed"

    return {
        "status":         "COMPLETE",
        "backtest_dates": len(complete),
        "avg_hit_rate":   round(avg_hit_rate, 1) if avg_hit_rate else None,
        "avg_return":     round(avg_return, 2)   if avg_return   else None,
        "avg_alpha":      round(avg_alpha, 2)     if avg_alpha    else None,
        "credibility":    credibility,
        "message": (
            f"Based on {len(complete)} backtest period(s): "
            f"STRONG BUY picks had {avg_hit_rate:.0f}% hit rate "
            f"with {avg_alpha:+.1f}% alpha vs NEPSE index"
        ) if avg_hit_rate and avg_alpha else "Calculating..."
    }


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("Running NEPSE Gem Finder backtest...\n")
    all_results = run_full_backtest(holding_days=30)
    summary     = backtest_summary(all_results)

    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")

    if summary["status"] == "INSUFFICIENT_DATA":
        print(f"\n⏳ {summary['message']}")
        print(f"\nProgress: {summary['days_collected']} days collected "
              f"/ {summary['days_needed']} days needed")
        print("\nKeep running: python pipeline.py --once")
        print("Results will appear automatically once enough data is collected.")
    else:
        print(f"\n✅ {summary['message']}")
        print(f"\nBacktest dates analysed: {summary['backtest_dates']}")
        print(f"Average hit rate:        {summary['avg_hit_rate']}%")
        print(f"Average return:          {summary['avg_return']:+.2f}%")
        print(f"Average alpha vs NEPSE:  {summary['avg_alpha']:+.2f}%")
        print(f"System credibility:      {summary['credibility']}")

    print(f"\n{'='*60}")
    print("PER-DATE RESULTS")
    print(f"{'='*60}")
    for r in all_results:
        status_icon = {
            "COMPLETE":     "✅",
            "WAITING":      "⏳",
            "NO_DATA":      "❌",
            "INSUFFICIENT": "⚠️"
        }.get(r["status"], "?")

        print(f"\n{status_icon} Entry: {r['entry_date']} | Status: {r['status']}")
        if r["status"] == "COMPLETE":
            sb = r["signal_stats"].get("STRONG BUY", {})
            print(f"   STRONG BUY: {sb.get('count',0)} stocks, "
                  f"hit rate {sb.get('hit_rate',0)}%, "
                  f"avg return {sb.get('avg_return',0):+.2f}%")
            if r.get("alpha") is not None:
                print(f"   Alpha vs NEPSE index: {r['alpha']:+.2f}%")
        else:
            print(f"   {r.get('message','')}")