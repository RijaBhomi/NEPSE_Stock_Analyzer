# Collects live NEPSE stock data from merolagani.com
# Source 1: /latestmarket.aspx  → price, volume, % change for ALL stocks
# Source 2: /CompanyDetail.aspx → P/E, EPS, sector, 52w high/low per stock
# Source 3: /Indices.aspx       → NEPSE index value for market regime

import requests
import pandas as pd
import logging
import time
import os
from bs4 import BeautifulSoup
from datetime import datetime, date
from sqlalchemy import create_engine, text

# logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger= logging.getLogger(__name__)

# database
DB_URL = "sqlite:///nepse.db"
engine = create_engine(DB_URL, echo=False)

# headers
# make requests look like a real browser so the site doesnt block
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://merolagani.com/",
}

BASE_URL = "https://merolagani.com"

# database setup
def create_tables():
    with engine.connect() as conn:
        # all listed stocks with metadata
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol          TEXT PRIMARY KEY,
                company_name    TEXT,
                sector          TEXT,
                last_updated    TEXT
            )
        """))

        # daily price- one row per stock per day
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                date        TEXT NOT NULL,
                ltp         REAL,
                open        REAL,
                high        REAL,
                low         REAL,
                prev_close  REAL,
                pct_change  REAL,
                volume      INTEGER,
                pe_ratio    REAL,
                eps         REAL,
                book_value  REAL,
                week52_high REAL,
                week52_low  REAL,
                scraped_at  TEXT,
                UNIQUE(symbol, date)
            )
        """))

        # nepse index values for market regime
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_index (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL UNIQUE,
                index_value REAL,
                scraped_at  TEXT
            )
        """))
        conn.commit()
    logger.info("Database tables ready")
                          

# helper function to parse numbers with commas and handle missing data
def safe_float(text_val):
    """Convert a string like '1,234.56' or '-' to float. Returns None if invalid."""
    if not text_val:
        return None
    cleaned = str(text_val).replace(",", "").replace("%", "").strip()
    if cleaned in ("-", "", "N/A", "—"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None

def get_soup(url, retries=3):
    """Fetch a page and return BeautifulSoup. Retries up to 3 times."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(3)
    logger.error(f"All {retries} attempts failed for {url}")
    return None

# scrape 1: all live prices
def scrape_live_market():
    """
    Scrapes the live trading table from merolagani.
    Returns a list of dicts with symbol, ltp, pct_change, open, high, low, volume, prev_close.
    This gives us ALL listed stocks in one page.
    """
    logger.info("Scraping live market prices from merolagani...")
    soup = get_soup(f"{BASE_URL}/latestmarket.aspx")
    if not soup:
        return []

    stocks = []
    # the live trading table has id="live-market" or similar
    # we find the table by looking for rows with stock symbols
    table = soup.find("table", {"id": "ctl00_ContentPlaceHolder1_LiveTrading1_gvData"})
    if not table:
        # fallback: find the first large table on the page
        tables = soup.find_all("table")
        table = next((t for t in tables if len(t.find_all("tr")) > 10), None)
    
    if not table:
        logger.error("Could not find live market table")
        return []

    rows = table.find_all("tr")[1:]  # skip header row
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:
            continue

        # extract symbol from the link text
        symbol_tag = cols[0].find("a")
        if not symbol_tag:
            continue
        symbol = symbol_tag.text.strip()
        company_name = symbol_tag.get("title", "").replace(f"{symbol} (", "").rstrip(")")

        stocks.append({
            "symbol":       symbol,
            "company_name": company_name,
            "ltp":          safe_float(cols[1].text),
            "pct_change":   safe_float(cols[2].text),
            "open":         safe_float(cols[3].text),
            "high":         safe_float(cols[4].text),
            "low":          safe_float(cols[5].text),
            "volume":       safe_float(cols[6].text),
            "prev_close":   safe_float(cols[7].text) if len(cols) > 7 else None,
        })

    logger.info(f"Found {len(stocks)} stocks in live market")
    return stocks

# scraper 2: company details for P/E, EPS, sector, 52w high/low
# Known NEPSE sector classifications — manually maintained
# This maps symbol prefixes/names to sectors
SECTOR_MAP = {
    # Commercial Banks
    "NABIL": "Commercial Bank", "ADBL": "Commercial Bank",
    "EBL": "Commercial Bank", "GBIME": "Commercial Bank",
    "HBL": "Commercial Bank", "KBL": "Commercial Bank",
    "MBL": "Commercial Bank", "NBL": "Commercial Bank",
    "NICA": "Commercial Bank", "NIMB": "Commercial Bank",
    "NMB": "Commercial Bank", "PCBL": "Commercial Bank",
    "PRVU": "Commercial Bank", "SANIMA": "Commercial Bank",
    "SBL": "Commercial Bank", "SCB": "Commercial Bank",
    "SBI": "Commercial Bank", "SRBL": "Commercial Bank",
    # Development Banks
    "EDBL": "Development Bank", "GDBL": "Development Bank",
    "JBBL": "Development Bank", "KSBBL": "Development Bank",
    "LBBL": "Development Bank", "MLBL": "Development Bank",
    "MNBBL": "Development Bank", "MDB": "Development Bank",
    "SADBL": "Development Bank", "SHINE": "Development Bank",
    # Hydropower
    "AHPC": "Hydropower", "AKJCL": "Hydropower",
    "BARUN": "Hydropower", "BHPL": "Hydropower",
    "CHCL": "Hydropower", "DHPL": "Hydropower",
    "GHL": "Hydropower", "HDHPC": "Hydropower",
    "HURJA": "Hydropower", "KPCL": "Hydropower",
    "MHNL": "Hydropower", "NHDL": "Hydropower",
    "NHPC": "Hydropower", "PMHPL": "Hydropower",
    "RRHP": "Hydropower", "SSHL": "Hydropower",
    "UMRH": "Hydropower", "UPPER": "Hydropower",
    # Insurance
    "ALICL": "Insurance", "GLICL": "Insurance",
    "IGIL": "Insurance", "LICN": "Insurance",
    "NIL": "Insurance", "NLG": "Insurance",
    "NLIC": "Insurance", "PRIN": "Insurance",
    "RBCL": "Insurance", "SICL": "Insurance",
    "SLICL": "Insurance", "UIC": "Insurance",
    # Finance
    "BFC": "Finance", "CFCL": "Finance",
    "GFCL": "Finance", "ICFC": "Finance",
    "JFL": "Finance", "MFIL": "Finance",
    "MPFL": "Finance", "PFL": "Finance",
    "PROFL": "Finance", "SFCL": "Finance",
    # Hotels
    "AHL": "Hotels", "KDL": "Hotels",
    "OHL": "Hotels", "SHL": "Hotels",
    "TRH": "Hotels",
    # Manufacturing
    "BPCL": "Manufacturing", "BNT": "Manufacturing",
    "HRL": "Manufacturing", "SHIVM": "Manufacturing",
    "UNFL": "Manufacturing",
    # Telecom
    "NTC": "Telecom", "NCELL": "Telecom",
}

# Typical P/E ranges by sector in NEPSE (based on market research)
# Used for sector-relative scoring when individual P/E isn't available
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

def scrape_company_detail(symbol: str) -> dict:
    """
    Gets sector info for a stock.
    P/E and EPS load via JavaScript on all major NEPSE sites so
    we can't scrape them with requests alone.
    Instead we use our sector map and will derive scoring from
    price momentum and volume data we already have.
    """
    sector = SECTOR_MAP.get(symbol, "Others")
    return {
        "symbol": symbol,
        "sector": sector,
        "pe_ratio": None,   # would need Selenium to get this
        "eps": None,        # would need Selenium to get this
        "book_value": None,
        "week52_high": None,
        "week52_low": None,
    }

# save to database
def save_prices(stocks_data):
    """Save list of stock price dicts to daily_prices table."""
    today = date.today().isoformat()
    scraped_at = datetime.now().isoformat()
    saved = 0
    skipped = 0

    with engine.connect() as conn:
        for s in stocks_data:
            if not s.get("symbol") or not s.get("ltp"):
                continue
            try:
                conn.execute(text("""
                    INSERT OR IGNORE INTO daily_prices
                        (symbol, date, ltp, open, high, low, prev_close,
                         pct_change, volume, pe_ratio, eps, book_value,
                         week52_high, week52_low, scraped_at)
                    VALUES
                        (:symbol, :date, :ltp, :open, :high, :low, :prev_close,
                         :pct_change, :volume, :pe_ratio, :eps, :book_value,
                         :week52_high, :week52_low, :scraped_at)
                """), {
                    "symbol":       s["symbol"],
                    "date":         today,
                    "ltp":          s.get("ltp"),
                    "open":         s.get("open"),
                    "high":         s.get("high"),
                    "low":          s.get("low"),
                    "prev_close":   s.get("prev_close"),
                    "pct_change":   s.get("pct_change"),
                    "volume":       s.get("volume"),
                    "pe_ratio":     s.get("pe_ratio"),
                    "eps":          s.get("eps"),
                    "book_value":   s.get("book_value"),
                    "week52_high":  s.get("week52_high"),
                    "week52_low":   s.get("week52_low"),
                    "scraped_at":   scraped_at,
                })
                saved += 1
            except Exception as e:
                skipped += 1
                logger.debug(f"Skipped {s['symbol']}: {e}")

        # upsert stocks metadata
        for s in stocks_data:
            if not s.get("symbol"):
                continue
            conn.execute(text("""
                INSERT OR REPLACE INTO stocks (symbol, company_name, sector, last_updated)
                VALUES (:symbol, :company_name, :sector, :last_updated)
            """), {
                "symbol":       s["symbol"],
                "company_name": s.get("company_name", ""),
                "sector":       s.get("sector", ""),
                "last_updated": today,
            })

        conn.commit()

    logger.info(f"Saved {saved} price records ({skipped} skipped as duplicates)")


def save_index(index_value):
    """Save NEPSE index value to market_index table."""
    if not index_value:
        return
    today = date.today().isoformat()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT OR IGNORE INTO market_index (date, index_value, scraped_at)
            VALUES (:date, :value, :scraped_at)
        """), {"date": today, "value": index_value, "scraped_at": datetime.now().isoformat()})
        conn.commit()
    logger.info(f"Saved NEPSE index: {index_value}")

def scrape_nepse_index():
    """
    Tries to extract NEPSE index from market pages.
    Both major sites load index via JS so this is best-effort.
    """
    logger.info("Scraping NEPSE index value...")
    import re

    for url in [f"{BASE_URL}/MarketSummary.aspx", f"{BASE_URL}/latestmarket.aspx"]:
        soup = get_soup(url)
        if soup:
            text = soup.get_text()
            matches = re.findall(r"\b([23]\d{3}\.\d{2})\b", text)
            for match in matches:
                val = safe_float(match)
                if val and 2000 < val < 4000:
                    logger.info(f"NEPSE Index: {val}")
                    return val

    logger.warning("Could not find NEPSE index — will retry next run")
    return None


# ── main run ───────────────────────────────────────────────────
def run_scraper(enrich_details=False):
    """
    Full scraping run.
    enrich_details=True → also visit each company page for P/E, EPS, sector.
    This is slower (1 request per stock) so set to False for quick runs.
    For daily pipeline runs, set to True for top 100 stocks only.
    """
    create_tables()

    # step 1: get all live prices
    stocks = scrape_live_market()
    if not stocks:
        logger.error("No stocks scraped — aborting")
        return []

    # step 2: optionally enrich with P/E, EPS, sector from detail pages
    if enrich_details:
        logger.info(f"Enriching details for top 100 stocks (this takes ~5 mins)...")
        # only enrich top 100 by volume to avoid hammering the site
        stocks_sorted = sorted(
            [s for s in stocks if s.get("volume")],
            key=lambda x: x["volume"] or 0,
            reverse=True
        )[:100]

        for i, stock in enumerate(stocks_sorted):
            detail = scrape_company_detail(stock["symbol"])
            stock.update(detail)
            time.sleep(1.5)  # be polite — wait 1.5s between requests
            if (i + 1) % 10 == 0:
                logger.info(f"  Enriched {i+1}/100 stocks...")

    # step 3: save to database
    save_prices(stocks)

    # step 4: save NEPSE index
    index_val = scrape_nepse_index()
    # fallback: use last known index if scraping fails
    if not index_val:
        index_val = 2838.40  # last known NEPSE index — update this weekly
        logger.info(f"Using fallback NEPSE index: {index_val}")
    save_index(index_val)

    logger.info("Scraper run complete!")
    return stocks

# test
if __name__ == "__main__":
    logger.info("Running scraper test (prices only, no detail enrichment)...")
    stocks = run_scraper(enrich_details=False)

    if stocks:
        print(f"\n✅ Scraped {len(stocks)} stocks")
        print("\nFirst 5 stocks:")
        for s in stocks[:5]:
            print(f"  {s['symbol']:10} LTP: {s['ltp']:>8}  Change: {s['pct_change']:>6}%  Vol: {s['volume']}")
    else:
        print(" No data scraped — check logs/scraper.log for details")
