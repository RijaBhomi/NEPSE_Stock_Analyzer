
# 💎 NEPSE Stock Analyzer

[![Daily NEPSE Pipeline](https://github.com/RijaBhomi/NEPSE_Stock_Analyzer/actions/workflows/daily_pipeline.yml/badge.svg)](https://github.com/RijaBhomi/NEPSE_Stock_Analyzer/actions/workflows/daily_pipeline.yml)

A fully automated, data-driven stock screening tool built specifically for the **Nepal Stock Exchange (NEPSE)**. Every day after 4 PM NPT, it scrapes live market data, computes technical indicators, scores all listed stocks on a 0–12 scale, and surfaces them on an interactive Streamlit dashboard — complete with market regime detection, sector filtering, and plain-English explanations for every score.

> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only. It is **not financial advice**. Always do your own research before investing.

---

## 📸 Dashboard Preview

The dashboard is divided into four sections:

| Section | What You Get |
|---|---|
| **Market Overview** | BULL/NEUTRAL/BEAR regime banner, KPI cards, top gainers, losers & most active stocks |
| **Gems Today** | Full ranked table of all scored stocks with signal badges and score progress bars |
| **Stock Deep Dive** | Price history chart (LTP + MA7), score breakdown bar chart, plain-English explanation |
| **How It Works** | Transparent explanation of the scoring rules and disclaimer |

---

## 🏗️ Architecture

The project follows a clean ETL + scoring pipeline pattern:

```
merolagani.com
      │
      ▼
 scraper.py          ← Collects live prices, volume, index for all NEPSE stocks
      │
      ▼
 validator.py        ← Filters out low-quality / unscoreable stocks
      │
      ▼
 indicators.py       ← Computes RSI, MA7, MA200, 52-week position
      │
      ▼
 scorer.py           ← Scores each stock 0–12 across 5 rules, assigns signal
      │
      ▼
 nepse.db (SQLite)   ← Stores everything: prices, indicators, scores, index
      │
      ▼
 dashboard.py        ← Streamlit dashboard reads from DB and renders the UI
```

**GitHub Actions** (`daily_pipeline.yml`) runs the full pipeline automatically every trading day after market close.

---

## 🔬 The Scoring System

Each stock is scored out of **12 points** across 5 rules. Higher score = more signals aligning for a potential opportunity.

### Rule 1 — Price Sanity (Gate Rule)
Enforced by `validator.py` before any stock reaches the scorer. Stocks trading below Rs 100 or in freefall are excluded upstream.

### Rule 2 — 52-Week Position (Value Signal) — *up to 3 pts*
Measures where the current price sits in its 52-week high/low range using the formula:

```
position = (LTP - 52w_low) / (52w_high - 52w_low)
```

| Position | Points | Meaning |
|---|---|---|
| 0 – 20% | **3 pts** | Bottom 20% of yearly range — very cheap relative to history |
| 20 – 35% | **2 pts** | Lower third — cheap |
| 35 – 50% | **1 pt** | Middle of range — neutral |
| Above 50% | **0 pts** | Upper half — no value signal |

### Rule 3 — RSI Signal (Momentum) — *up to 3 pts*
Uses the Relative Strength Index (14-day) with trend confirmation to avoid the "falling knife" trap.

| RSI Condition | Points | Meaning |
|---|---|---|
| RSI < 30 AND rising | **3 pts** | Oversold AND recovering — strong signal |
| RSI < 30, direction unclear | **2 pts** | Oversold, watch closely |
| RSI < 30 AND falling | **0 pts** | Oversold but still declining — avoid |
| RSI 30 – 45 | **1 pt** | Cooling down, potential setup |
| RSI 45+ | **0 pts** | Neutral or overbought |

*Requires 14+ days of price history. Defaults to 0 pts for newer stocks.*

### Rule 4 — Moving Average Trend (Short & Long-Term) — *up to 3 pts*

| Condition | Points |
|---|---|
| Price above MA7 (7-day average) | **+1 pt** |
| Price below MA200 for < 60 days | **+2 pts** — recent dip, possible opportunity |
| Price below MA200 for 60 – 90 days | **+1 pt** — extended dip, caution |
| Price below MA200 for 90+ days | **0 pts** — extended downtrend |

*MA7 requires 7+ days; MA200 requires 200+ days of accumulated price history.*

### Rule 5 — Volume Signal (Buying Interest) — *up to 2 pts*
Compares today's volume against the stock's 30-day average.

| Condition | Points |
|---|---|
| Volume ≥ 1.5× average on an UP day | **2 pts** — strong buying interest |
| Volume 1.0 – 1.5× average on an UP day | **1 pt** — moderate interest |
| Volume ≥ 1.5× average on a DOWN day | **0 pts** — heavy selling, avoid |
| Low/normal volume | **0 pts** — no signal |

### Rule 6 — Market Regime Bonus — *up to 1 pt*
The NEPSE index is compared against its 30-day moving average to classify the broader market.

| Regime | Points | Logic |
|---|---|---|
| BULL (index > avg × 1.02) | **+1 pt** | Rising tide lifts all boats |
| NEUTRAL | **0 pts** | No adjustment |
| BEAR (index < avg × 0.98) | **0 pts** | STRONG BUY signals are downgraded to WATCH |

### Signal Thresholds

| Signal | Score Range | Meaning |
|---|---|---|
| 🟢 **STRONG BUY** | 7 – 12 | Multiple signals align — worth deeper research |
| 🟡 **WATCH** | 5 – 6 | Some positive signals — monitor closely |
| ⚪ **NEUTRAL** | 3 – 4 | Mixed signals — no strong reason to act |
| 🔴 **AVOID** | 0 – 2 | Few or no positive signals |

---

## 📁 Project Structure

```
NEPSE_Stock_Analyzer/
├── scraper.py          # Live price + NEPSE index scraping from merolagani.com
├── validator.py        # Data quality checks and stock filtering
├── indicators.py       # RSI, MA7, MA200 computation from price history
├── scorer.py           # 5-rule scoring engine (0–12 pts) + signal assignment
├── dashboard.py        # Streamlit 4-section dashboard (dark green theme)
├── pipeline.py         # Orchestrator: runs scraper → validator → indicators → scorer
├── backtest.py         # Backtesting module for historical signal performance
├── market_regime.py    # NEPSE index trend classification (BULL/NEUTRAL/BEAR)
├── nepse.db            # SQLite database (prices, indicators, scores, index)
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version (for Streamlit Cloud)
└── .github/
    └── workflows/
        └── daily_pipeline.yml   # GitHub Actions: auto-runs pipeline each trading day
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/RijaBhomi/NEPSE_Stock_Analyzer.git
cd NEPSE_Stock_Analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline (Collect + Score Data)

```bash
# Quick run — prices only (no per-stock detail enrichment)
python pipeline.py --once

# Full run — includes per-stock sector enrichment for top 100 stocks
python pipeline.py --once --enrich
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🤖 Automated Daily Pipeline

The pipeline runs automatically every trading day via **GitHub Actions** (`.github/workflows/daily_pipeline.yml`). It executes after NEPSE market close (around 4 PM NPT / ~10:15 AM UTC) in sequence:

1. `scraper.py` — fetches all live prices and the NEPSE index
2. `validator.py` — filters stocks that lack enough data
3. `indicators.py` — computes RSI, MA7, and MA200 from stored history
4. `scorer.py` — scores all validated stocks and saves results to `nepse.db`

No manual intervention needed once the Actions workflow is active.

---

## 🗄️ Database Schema

The SQLite database (`nepse.db`) stores four main tables:

**`stocks`** — master list of all NEPSE-listed companies with sector classification

**`daily_prices`** — one row per stock per trading day: LTP, open, high, low, volume, % change, 52-week high/low, scraped timestamp

**`indicators`** — computed technical indicators per stock per day: RSI, RSI trend, price vs MA7, price vs MA200, days below MA200

**`scores`** — final scored output per stock per day: total score, signal, per-rule points breakdown, 52-week position, plain-English explanation

**`market_index`** — NEPSE composite index values for regime detection

---

## 📊 Key Technical Decisions

**Why SQLite?** NEPSE has ~200–300 listed stocks. SQLite is zero-infrastructure, sufficient for this data volume, and portable — the entire historical dataset travels as a single `.db` file.

**Why no P/E or EPS scoring?** Both merolagani.com and ShareSansar load P/E and EPS via JavaScript, making them inaccessible to `requests`-based scraping without a headless browser. The scoring system is therefore designed entirely around data that *can* be scraped: price, volume, and computed technicals. This is documented transparently in the `scorer.py` header.

**Why RSI trend confirmation?** A falling RSI below 30 is a "falling knife" — the stock looks oversold but is still declining. The scorer checks whether RSI is *rising* before awarding the full 3 points, avoiding this trap.

**Why 52-week data from local DB?** The live merolagani table doesn't always expose 52-week high/low fields reliably. The scraper therefore computes these from accumulated `daily_prices` history once 7+ days of data exist (`compute_52week_from_db`).

---

## 🔧 Configuration

The sector classification map in `scraper.py` (`SECTOR_MAP`) covers the major NEPSE sectors:

- Commercial Banks, Development Banks, Finance Companies
- Hydropower
- Insurance
- Hotels & Tourism
- Manufacturing & Processing
- Telecom

Unlisted or unrecognised symbols default to `"Others"`. You can extend the map with additional symbols as needed.

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Interactive charts (candlestick, bar, line) |
| **Pandas** | Data manipulation and indicator computation |
| **SQLAlchemy + SQLite** | Database ORM and storage |
| **Requests + BeautifulSoup** | Web scraping (merolagani.com) |
| **GitHub Actions** | Daily automated pipeline |

---

## ⚙️ Requirements

```
streamlit
pandas
plotly
sqlalchemy
requests
beautifulsoup4
lxml
```

See `requirements.txt` for exact pinned versions.

---

## 🔮 Known Limitations & Roadmap

**Current limitations:**
- P/E and EPS are not yet available (JavaScript-rendered on all major NEPSE sites — would require Selenium or Playwright)
- MA200 requires 200 days of accumulated data to become meaningful — improves automatically over time
- NEPSE index scraping is best-effort; a fallback hardcoded value is used if the live scrape fails
- Sector classification is manually maintained via `SECTOR_MAP`

**Potential future improvements:**
- Add Selenium/Playwright scraping for P/E and EPS data
- Email or Telegram alerts for new STRONG BUY signals
- Portfolio tracking — mark stocks as held and monitor score changes
- Candlestick charts with volume overlay in the Deep Dive section
- Sector-relative scoring (compare a stock's P/E to sector median)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Rija Bhomi**
- GitHub: [@RijaBhomi](https://github.com/RijaBhomi)

---

*Built with Python, Streamlit & Plotly · Data source: merolagani.com · Updated daily after 4 PM NPT*
