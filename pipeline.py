# Runs the full data collection pipeline daily at 4 pm
# NEPSE closes at 3pm Nepal time so 4pm gives time for final price update

import schedule
import time
import logging
import sys
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_daily_pipeline():
    """Runs scraper → indicators → market regime → scorer every day."""
    logger.info("=" * 50)
    logger.info(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # step 1: scrape prices
    logger.info("Step 1: Scraping prices...")
    from scraper import run_scraper
    run_scraper(enrich_details=False)

    # step 2: calculate indicators
    logger.info("Step 2: Calculating indicators...")
    from indicators import run_indicators
    run_indicators()

    # step 3: market regime
    logger.info("Step 3: Checking market regime...")
    from market_regime import determine_regime, save_regime
    regime_data = determine_regime()
    save_regime(regime_data)
    logger.info(f"Regime: {regime_data['regime']}")

    # step 4: score all stocks
    logger.info("Step 4: Scoring stocks...")
    from scorer import run_scorer
    results = run_scorer()
    if results:
        top = results[0]
        logger.info(f"Top stock today: {top['symbol']} "
                    f"(score {top['total_score']}/12, {top['signal']})")

    logger.info("Pipeline complete!")
    logger.info("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # run once immediately and exit
        run_daily_pipeline()
    else:
        # run daily at 4pm Nepal time
        # (your laptop needs to be on at 4pm for this to trigger)
        schedule.every().day.at("16:00").do(run_daily_pipeline)
        logger.info("Scheduler started — pipeline runs daily at 4:00 PM")
        logger.info("Running once immediately on startup...")
        run_daily_pipeline()

        while True:
            schedule.run_pending()
            time.sleep(60)