# before stock enters the scoring system, it must pass
# 4 quality checks:
# 1. FRESHNESS: was this data collected today?
# 2. RANGE: are the values in realistic ranges?
# 3. COMPLETENESS:  do they have the key fields needed for scoring
# 4. PRICE SANITY: is the price above Rs. 1000

# Each stock gets a data_quality badge:
#   ✅ VERIFIED   — passed all checks, safe to score
#   ⚠️ PARTIAL    — some fields missing, score with lower confidence
#   ❌ INVALID    — failed critical checks, do not score

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL  = "sqlite:///nepse.db"
engine  = create_engine(DB_URL, echo=False)

# check 1: FRESHNESS
def check_freshness(scraped_at: str, max_days_old: int = 2) -> tuple[bool, str]:
    """
    Rejects data older than max_days_old market days.

    Why this matters:
    NEPSE trades Sunday–Thursday. If today is Monday and the data
    says Friday, that's actually fine (weekend gap). But if data
    is from 5 days ago on a trading day, something went wrong with
    the scraper and the prices are stale.

    Returns: (is_fresh: bool, reason: str)
    """
    if not scraped_at:
        return False, "No scrape timestamp found"

    try:
        scraped_date = datetime.fromisoformat(scraped_at).date()
    except ValueError:
        return False, f"Invalid timestamp format: {scraped_at}"

    days_old = (date.today() - scraped_date).days

    if days_old <= max_days_old:
        return True, f"Fresh — scraped {days_old} day(s) ago"
    else:
        return False, f"Stale — scraped {days_old} days ago (max allowed: {max_days_old})"

# check 2: RANGE
def check_ranges(stock: dict) -> tuple[bool, list]:
    """
    Checks that numeric values are within realistic bounds.

    Why this matters:
    If ShareSansar or Merolagani has a formatting error or shows
    a corrupted value like P/E = 99999, our scoring system would
    treat it as a real number and make wrong recommendations.

    We define realistic ranges based on NEPSE market knowledge:
    - Price: Rs 10 to Rs 50,000 (most stocks Rs 100–5000)
    - P/E: 0 to 200 (negative P/E means loss-making, above 200 is speculation)
    - EPS: -500 to 500 (extreme values flag data errors)
    - Volume: 0 to 100 crore units per day

    Returns: (all_valid: bool, list of warning messages)
    """
    warnings = []

    # define what's "realistic" for each field
    # format: (field_name, min_val, max_val, label)
    range_rules = [
        ("ltp",       10,    50000, "Last Traded Price"),
        ("pe_ratio",  0,     200,   "P/E Ratio"),
        ("eps",       -500,  500,   "EPS"),
        ("week52_high", 10,  50000, "52-week High"),
        ("week52_low",  5,   50000, "52-week Low"),
    ]

    all_valid = True
    for field, min_val, max_val, label in range_rules:
        val = stock.get(field)
        if val is None:
            continue  # missing is handled by completeness check, not range check

        if not (min_val <= val <= max_val):
            warnings.append(f"{label} out of range: {val} (expected {min_val}–{max_val})")
            all_valid = False

    # extra check: 52-week high must be >= 52-week low
    high = stock.get("week52_high")
    low  = stock.get("week52_low")
    if high and low and high < low:
        warnings.append(f"52w high ({high}) is less than 52w low ({low}) — data error")
        all_valid = False

    return all_valid, warnings

# check 3: COMPLETENESS
def check_completeness(stock: dict) -> tuple[str, list]:
    """
    Checks which key fields are present.

    Why this matters:
    We can score a stock with ONLY price data (basic scoring).
    But for a full score we need P/E and EPS too.
    If we only have price, we still show the stock but with a
    ⚠️ PARTIAL badge and a lower confidence level.

    Fields needed:
    - CRITICAL (must have): ltp, symbol
    - SCORING (needed for full score): pe_ratio, eps
    - NICE TO HAVE: week52_high, week52_low, volume

    Returns: (quality_level: str, missing_fields: list)
        quality_level is "FULL", "PARTIAL", or "CRITICAL_MISSING"
    """
    critical_fields = ["ltp", "symbol"]
    scoring_fields  = ["pe_ratio", "eps"]
    nice_fields     = ["week52_high", "week52_low", "volume"]

    missing_critical = [f for f in critical_fields if not stock.get(f)]
    missing_scoring  = [f for f in scoring_fields  if not stock.get(f)]
    missing_nice     = [f for f in nice_fields      if not stock.get(f)]

    if missing_critical:
        return "CRITICAL_MISSING", missing_critical

    if missing_scoring:
        return "PARTIAL", missing_scoring

    return "FULL", missing_nice  # nice fields missing is just informational

# check 4: PRICE SANITY
def check_price_sanity(stock: dict) -> tuple[bool, str]:
    """
    Filters out very low-priced stocks (penny stocks).

    Why this matters:
    Stocks trading below Rs 100 in NEPSE are usually:
    - Companies in financial trouble
    - Thinly traded stocks easy to manipulate
    - Companies with structural problems
    These are high-risk and not suitable for a beginner screener.

    We also check that today's price isn't more than 60% below
    the 52-week high — which would suggest a company in serious trouble.

    Returns: (passes: bool, reason: str)
    """
    ltp = stock.get("ltp")

    if not ltp:
        return False, "No price data"

    # penny stock check
    if ltp < 100:
        return False, f"Penny stock — price Rs {ltp} is below Rs 100 minimum"

    # freefall check: price more than 60% below 52-week high
    high_52w = stock.get("week52_high")
    if high_52w and high_52w > 0:
        ratio = ltp / high_52w
        if ratio < 0.40:
            # price has fallen more than 60% from yearly high
            pct_drop = round((1 - ratio) * 100, 1)
            return False, f"Possible freefall — price is {pct_drop}% below 52-week high"

    return True, f"Price Rs {ltp} passes sanity check"

# MASTER VALIDATOR: runs all 4 checks
def validate_stock(stock: dict) -> dict:
    """
    Runs all 4 checks on a single stock dict.
    Returns the original stock dict with added validation fields:
      - data_quality:  "VERIFIED" / "PARTIAL" / "INVALID"
      - quality_score: 0–4 (how many checks passed)
      - quality_notes: human-readable list of issues found
    """
    notes = []
    checks_passed = 0

    # ── Check 1: Freshness ────────────────────────────────────
    is_fresh, freshness_msg = check_freshness(stock.get("scraped_at", ""))
    if is_fresh:
        checks_passed += 1
    else:
        notes.append(f"STALE: {freshness_msg}")

    # ── Check 2: Range validation ────────────────────────────
    ranges_ok, range_warnings = check_ranges(stock)
    if ranges_ok:
        checks_passed += 1
    else:
        notes.extend([f"RANGE: {w}" for w in range_warnings])

    # ── Check 3: Completeness ────────────────────────────────
    completeness, missing = check_completeness(stock)
    if completeness == "FULL":
        checks_passed += 1
        data_completeness = "FULL"
    elif completeness == "PARTIAL":
        checks_passed += 0.5  # partial credit
        data_completeness = "PARTIAL"
        notes.append(f"MISSING: {', '.join(missing)} — will use basic scoring only")
    else:
        data_completeness = "CRITICAL_MISSING"
        notes.append(f"CRITICAL MISSING: {', '.join(missing)}")

    # ── Check 4: Price sanity ────────────────────────────────
    price_ok, price_msg = check_price_sanity(stock)
    if price_ok:
        checks_passed += 1
    else:
        notes.append(f"PRICE: {price_msg}")

    # ── Determine final quality badge ────────────────────────
    # INVALID: failed freshness OR critical fields missing OR price sanity
    if not is_fresh or data_completeness == "CRITICAL_MISSING" or not price_ok:
        quality = "INVALID"

    # VERIFIED: passed all 4 checks with full data
    elif checks_passed >= 3.5 and data_completeness == "FULL":
        quality = "VERIFIED"

    # PARTIAL: passed most checks but missing some scoring data
    else:
        quality = "PARTIAL"

    # add validation results back to the stock dict
    return {
        **stock,
        "data_quality":    quality,
        "quality_score":   round(checks_passed, 1),
        "quality_notes":   " | ".join(notes) if notes else "All checks passed",
        "data_completeness": data_completeness,
    }


def validate_all(stocks: list) -> dict:
    """
    Runs validate_stock() on a list of stocks.
    Returns a dict with three lists:
      - verified: ready for full scoring
      - partial:  can be scored with lower confidence
      - invalid:  excluded from scoring entirely
    Also prints a summary.
    """
    verified = []
    partial  = []
    invalid  = []

    for stock in stocks:
        result = validate_stock(stock)
        if result["data_quality"] == "VERIFIED":
            verified.append(result)
        elif result["data_quality"] == "PARTIAL":
            partial.append(result)
        else:
            invalid.append(result)

    # summary
    total = len(stocks)
    logger.info(f"Validation complete: {total} stocks")
    logger.info(f"  ✅ VERIFIED : {len(verified)} ({round(len(verified)/total*100)}%)")
    logger.info(f"  ⚠️  PARTIAL  : {len(partial)} ({round(len(partial)/total*100)}%)")
    logger.info(f"  ❌ INVALID  : {len(invalid)} ({round(len(invalid)/total*100)}%)")

    # show a sample of why stocks were flagged as invalid
    if invalid:
        logger.info("Sample of invalid stocks:")
        for s in invalid[:5]:
            logger.info(f"  {s['symbol']:10} → {s['quality_notes']}")

    return {
        "verified": verified,
        "partial":  partial,
        "invalid":  invalid,
    }


# ══════════════════════════════════════════════════════════════════
# LOAD FROM DB AND VALIDATE
# ══════════════════════════════════════════════════════════════════
def validate_todays_data() -> dict:

    """

    Loads today's scraped data from the database and runs validation.

    This is what pipeline.py will call after scraping.

    """

    today = date.today().isoformat()



    with engine.connect() as conn:

        result = conn.execute(text("""

            SELECT p.*, s.sector, s.company_name

            FROM daily_prices p

            LEFT JOIN stocks s ON p.symbol = s.symbol

            WHERE p.date = :today

        """), {"today": today})



        rows = [dict(row._mapping) for row in result]



    if not rows:

        logger.warning(f"No data found in database for {today}")

        return {"verified": [], "partial": [], "invalid": []}



    logger.info(f"Loaded {len(rows)} stocks from database for {today}")

    return validate_all(rows)


# TEST
if __name__ == "__main__":
    print("Running validator on today's scraped data...\n")
    results = validate_todays_data()

    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"✅ VERIFIED  (safe to score)  : {len(results['verified'])}")
    print(f"⚠️  PARTIAL   (basic score only): {len(results['partial'])}")
    print(f"❌ INVALID   (excluded)        : {len(results['invalid'])}")

    print(f"\nSample VERIFIED stocks:")
    for s in results["verified"][:3]:
        print(f"  {s['symbol']:10} LTP:{s['ltp']:>8}  Quality: {s['data_quality']}")

    print(f"\nSample PARTIAL stocks (missing P/E or EPS):")
    for s in results["partial"][:3]:
        print(f"  {s['symbol']:10} LTP:{s['ltp']:>8}  Notes: {s['quality_notes']}")

    print(f"\nSample INVALID stocks (excluded from scoring):")
    for s in results["invalid"][:3]:
        print(f"  {s['symbol']:10} LTP:{s['ltp']:>8}  Reason: {s['quality_notes']}")