# Purpose: calculate technical indicators from historical price data

# Technical indicators are mathematical formulas applied to price
# history that help identify patterns. Like "signals"
# that experienced traders use to decide when to buy or sell.

# 5 indicators:
# 1. MOVING AVERAGES (MA7, MA30, MA200)
#    The average closing price over the last N days.
#    If today's price is ABOVE the average → stock is strong.
#    If today's price is BELOW the average → stock may be cheap.
#    MA7  = short-term trend (1 week)
#    MA30 = medium-term trend (1 month)
#    MA200 = long-term trend (year) — the most important one

# 2. RSI (Relative Strength Index)
#    Measures momentum on a scale of 0–100.
#    Under 30 = stock fell too fast, may bounce back (oversold)
#    Above 70 = stock rose too fast, may pull back (overbought)
#    30–70    = normal range
#    Formula: RSI = 100 - (100 / (1 + avg_gain/avg_loss))

# 3. RSI TREND
#    Is RSI rising or falling right now?
#    RSI under 30 AND rising = strong buy signal (oversold + recovering)
#    RSI under 30 AND falling = trap (still getting worse)

# 4. MA200 TREND
#    Is the 200-day MA itself going up or down?
#    MA200 trending up = long-term bull market for this stock
#    MA200 trending down = long-term bear market for this stock

# 5. DAYS BELOW MA200
#    How many consecutive days has the price been below MA200?
#    < 60 days = recent dip (possible buy opportunity)
#    > 90 days = extended downtrend (higher risk)

import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL = "sqlite:///nepse.db"
engine = create_engine(DB_URL, echo=False)

# database setup
def create_indicators_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS indicators (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT NOT NULL,
                date                TEXT NOT NULL,
                ma7                 REAL,
                ma30                REAL,
                ma200               REAL,
                rsi                 REAL,
                rsi_trend           TEXT,
                ma200_trend         TEXT,
                days_below_ma200    INTEGER,
                price_vs_ma7        TEXT,
                price_vs_ma200      TEXT,
                UNIQUE(symbol, date)
            )
        """))
        conn.commit()
    logger.info("Indicators table ready") 

# load price history for a symbol
def load_price_history(symbol: str, days: int = 210) -> pd.DataFrame:
    """
    Loads the last N days of closing prices for a symbol from the database.

    We load 210 days to calculate MA200 properly — you need at least
    200 data points for the 200-day moving average to be meaningful.

    Returns a DataFrame sorted by date (oldest first) with columns:
    date, ltp (last traded price = our "close" price), volume
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date, ltp, volume
            FROM daily_prices
            WHERE symbol = :symbol
              AND date >= :cutoff
              AND ltp IS NOT NULL
            ORDER BY date ASC
        """), {"symbol": symbol, "cutoff": cutoff})

        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "ltp", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["ltp"]  = pd.to_numeric(df["ltp"], errors="coerce")
    return df

# Indicator 1- Moving Averages
def calculate_moving_averages(prices: pd.Series) -> dict:
    """
    Calculates 3 moving averages from a price series.

    A moving average is just the mean of the last N prices.
    pandas makes this easy with .rolling(N).mean()

    Example:
    prices = [100, 102, 98, 105, 103]
    MA3    = [NaN, NaN, 100, 101.67, 102]
    (first 2 are NaN because we don't have 3 prices yet)

    We return the LATEST value (most recent day's MA).
    Returns None if we don't have enough data.
    """
    result = {}

    # need at least 7 days for MA7
    if len(prices) >= 7:
        ma7_series = prices.rolling(7).mean()
        result["ma7"] = round(float(ma7_series.iloc[-1]), 2)
    else:
        result["ma7"] = None

    # need at least 30 days for MA30
    if len(prices) >= 30:
        ma30_series = prices.rolling(30).mean()
        result["ma30"] = round(float(ma30_series.iloc[-1]), 2)
    else:
        result["ma30"] = None

    # need at least 200 days for MA200
    # most of our stocks won't have this yet since we just started scraping
    # we'll collect data over time and this will become available
    if len(prices) >= 200:
        ma200_series = prices.rolling(200).mean()
        result["ma200"] = round(float(ma200_series.iloc[-1]), 2)

        # MA200 TREND: compare today's MA200 to 10 days ago
        # if MA200 now > MA200 10 days ago → it's rising (bullish)
        # if MA200 now < MA200 10 days ago → it's falling (bearish)
        if len(ma200_series.dropna()) >= 11:
            ma200_now  = ma200_series.iloc[-1]
            ma200_past = ma200_series.iloc[-11]  # 10 days ago
            if ma200_now > ma200_past * 1.001:   # 0.1% threshold to avoid noise
                result["ma200_trend"] = "RISING"
            elif ma200_now < ma200_past * 0.999:
                result["ma200_trend"] = "FALLING"
            else:
                result["ma200_trend"] = "FLAT"
        else:
            result["ma200_trend"] = None
    else:
        result["ma200"] = None
        result["ma200_trend"] = None

    return result

# Indicator 2- RSI
def calculate_rsi(prices: pd.Series, period: int = 14) -> float | None:
    """
    Calculates RSI (Relative Strength Index) using the standard 14-day formula.

    HOW RSI WORKS:
    1. Calculate daily price changes: [+2, -1, +3, -2, +1, ...]
    2. Separate into gains and losses
    3. Average gain = mean of positive changes over 14 days
    4. Average loss = mean of negative changes over 14 days (as positive number)
    5. RS = average_gain / average_loss
    6. RSI = 100 - (100 / (1 + RS))

    Example interpretation:
    RSI = 25 → stock has been falling heavily → oversold → possible bounce
    RSI = 75 → stock has been rising heavily → overbought → possible pullback
    RSI = 50 → neutral, no strong signal

    We need at least period+1 (15) data points for a valid RSI.
    """
    if len(prices) < period + 1:
        return None

    # calculate day-over-day price changes
    # e.g. if prices = [100, 102, 99], changes = [+2, -3]
    changes = prices.diff().dropna()

    # separate gains (positive changes) from losses (negative changes)
    gains  = changes.clip(lower=0)   # keep only positive, replace negatives with 0
    losses = (-changes).clip(lower=0) # flip sign, keep only what were negatives

    # calculate rolling average over 'period' days
    # we use ewm (exponential weighted mean) which is the standard Wilder RSI method
    # it gives more weight to recent data than simple average
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean().iloc[-1]

    # avoid division by zero (if there are NO losses, RSI = 100)
    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

# Indicator 3- RSI TREND
def calculate_rsi_trend(prices: pd.Series, period: int = 14) -> str | None:
    """
    Determines if RSI is currently RISING, FALLING, or NEUTRAL.

    This is critical for avoiding the RSI trap:
    RSI under 30 + RISING = oversold AND recovering → strong signal
    RSI under 30 + FALLING = oversold AND still falling → avoid

    We compare today's RSI to RSI from 3 days ago.
    If the difference is more than 2 points → meaningful trend.
    """
    if len(prices) < period + 4:  # need extra days for comparison
        return None

    rsi_now  = calculate_rsi(prices, period)
    rsi_past = calculate_rsi(prices.iloc[:-3], period)  # RSI 3 days ago

    if rsi_now is None or rsi_past is None:
        return None

    diff = rsi_now - rsi_past

    if diff > 2:
        return "RISING"
    elif diff < -2:
        return "FALLING"
    else:
        return "NEUTRAL"

# Indicator 4- DAYS BELOW MA200
def calculate_days_below_ma200(prices: pd.Series) -> int | None:
    """
    Counts how many consecutive days the price has been below MA200.

    This distinguishes two scenarios:
    1. Price dipped below MA200 recently (< 60 days) → possible buy dip
    2. Price has been below MA200 for 90+ days → structural downtrend

    We go backwards from today counting consecutive days below MA200.
    The moment we find a day where price was ABOVE MA200, we stop.
    """
    if len(prices) < 200:
        return None  # can't calculate without enough history

    # calculate full MA200 series
    ma200_series = prices.rolling(200).mean()

    # count backwards from today
    days_below = 0
    for i in range(len(prices) - 1, -1, -1):
        ma200_val = ma200_series.iloc[i]
        price_val = prices.iloc[i]

        # skip if MA200 not yet calculable (early in series)
        if pd.isna(ma200_val):
            break

        if price_val < ma200_val:
            days_below += 1
        else:
            # price was above MA200 on this day → stop counting
            break

    return days_below

# price vs comparison
def price_vs_ma(current_price: float, ma_value: float | None) -> str | None:
    """
    Simple comparison: is current price above or below a moving average?
    Returns "ABOVE", "BELOW", or None if MA not available.
    """
    if ma_value is None or current_price is None:
        return None
    if current_price > ma_value:
        return "ABOVE"
    elif current_price < ma_value:
        return "BELOW"
    return "AT"

# Master function- calculates all indicators for one symbol
def calculate_indicators(symbol: str) -> dict | None:
    """
    Loads price history and calculates all indicators for one stock.
    Returns a dict with all indicator values, or None if not enough data.
    """
    # load historical prices
    df = load_price_history(symbol, days=210)

    if df.empty or len(df) < 7:
        # need at least 7 days of data for any indicator to work
        logger.debug(f"{symbol}: insufficient price history ({len(df)} days)")
        return None

    prices = df["ltp"]
    current_price = float(prices.iloc[-1])
    today = date.today().isoformat()

    # calculate all indicators
    mas          = calculate_moving_averages(prices)
    rsi          = calculate_rsi(prices)
    rsi_trend    = calculate_rsi_trend(prices)
    days_below   = calculate_days_below_ma200(prices)

    return {
        "symbol":            symbol,
        "date":              today,
        "ma7":               mas.get("ma7"),
        "ma30":              mas.get("ma30"),
        "ma200":             mas.get("ma200"),
        "rsi":               rsi,
        "rsi_trend":         rsi_trend,
        "ma200_trend":       mas.get("ma200_trend"),
        "days_below_ma200":  days_below,
        "price_vs_ma7":      price_vs_ma(current_price, mas.get("ma7")),
        "price_vs_ma200":    price_vs_ma(current_price, mas.get("ma200")),
    }

# save indicators to db
def save_indicators(indicators: dict):
    """Saves a single stock's indicators to the database."""
    if not indicators:
        return

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO indicators
                (symbol, date, ma7, ma30, ma200, rsi, rsi_trend,
                 ma200_trend, days_below_ma200, price_vs_ma7, price_vs_ma200)
            VALUES
                (:symbol, :date, :ma7, :ma30, :ma200, :rsi, :rsi_trend,
                 :ma200_trend, :days_below_ma200, :price_vs_ma7, :price_vs_ma200)
        """), indicators)
        conn.commit()

# Run for all stocks
def run_indicators():
    """
    Calculates and saves indicators for all stocks in the database.
    Called by pipeline.py after scraping.
    """
    create_indicators_table()

    # get list of all symbols
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT symbol FROM daily_prices"))
        symbols = [row[0] for row in result]

    logger.info(f"Calculating indicators for {len(symbols)} stocks...")
    saved = 0
    skipped = 0

    for symbol in symbols:
        ind = calculate_indicators(symbol)
        if ind:
            save_indicators(ind)
            saved += 1
        else:
            skipped += 1

    logger.info(f"Indicators done: {saved} calculated, {skipped} skipped (not enough history)")
    return saved

# TEST
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("Calculating indicators for all stocks...\n")
    total = run_indicators()

    # show sample results for a few well-known stocks
    test_symbols = ["NABIL", "ADBL", "AKJCL", "AHPC", "NTC"]

    print(f"\n{'='*65}")
    print(f"SAMPLE INDICATOR RESULTS")
    print(f"{'='*65}")
    print(f"{'Symbol':<10} {'RSI':>6} {'RSI Trend':<12} {'vs MA7':<8} {'vs MA200':<10} {'Days↓MA200'}")
    print(f"{'-'*65}")

    with engine.connect() as conn:
        for sym in test_symbols:
            result = conn.execute(text("""
                SELECT rsi, rsi_trend, price_vs_ma7, price_vs_ma200, days_below_ma200
                FROM indicators
                WHERE symbol = :sym
                ORDER BY date DESC LIMIT 1
            """), {"sym": sym})
            row = result.fetchone()
            if row:
                rsi, rsi_trend, vs_ma7, vs_ma200, days_below = row
                rsi_str   = f"{rsi:.1f}" if rsi else "N/A"
                trend_str = rsi_trend or "N/A"
                ma7_str   = vs_ma7   or "N/A"
                ma200_str = vs_ma200 or "N/A"
                days_str  = str(days_below) if days_below is not None else "N/A"
                print(f"{sym:<10} {rsi_str:>6} {trend_str:<12} {ma7_str:<8} {ma200_str:<10} {days_str}")
            else:
                print(f"{sym:<10} No indicator data yet")

    print(f"\nNote: MA200 needs 200 days of history. Since we just started")
    print(f"scraping, most stocks will show N/A for MA200 — this is correct.")
    print(f"After collecting data daily for a few weeks, more will populate.")
    print(f"\nRSI and MA7 will work immediately with just a few days of data.")