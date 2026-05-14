# dashboard.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: 4-section Streamlit dashboard for NEPSE Gem Finder
#
# SECTION 1: Market Overview
#   - NEPSE regime banner (BULL/NEUTRAL/BEAR)
#   - Today's market summary (gainers, losers, volume leaders)
#
# SECTION 2: Gems Today
#   - Ranked table of all scored stocks
#   - Filter by signal, sector, score range
#
# SECTION 3: Stock Deep Dive
#   - Select any stock → see price chart + score breakdown
#   - Plain English explanation of why it scored how it did
#
# SECTION 4: How It Works
#   - Explains the scoring system so users understand it
#   - Disclaimer that this is not financial advice
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="NEPSE Gem Finder",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── dark green theme matching ETL project ──────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1a14; }

    .navbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:14px 32px; background:#0f1a14;
        border-bottom:1px solid #1e3a28; margin-bottom:0;
    }
    .nav-brand { font-size:22px; font-weight:700; color:#4ade80; }
    .nav-sub   { font-size:13px; color:#6b9e7e; margin-top:2px; }

    .regime-bull   { background:#14532d; border:1px solid #4ade80;
                     border-radius:10px; padding:12px 20px; color:#4ade80; }
    .regime-bear   { background:#450a0a; border:1px solid #f87171;
                     border-radius:10px; padding:12px 20px; color:#f87171; }
    .regime-neutral{ background:#162419; border:1px solid #6b9e7e;
                     border-radius:10px; padding:12px 20px; color:#a7c4b0; }

    .kpi-card {
        background:#162419; border-radius:14px;
        padding:18px 22px; border:1px solid #1e3a28; height:110px;
    }
    .kpi-label { font-size:11px; font-weight:600; color:#6b9e7e;
                 text-transform:uppercase; letter-spacing:.8px; }
    .kpi-value { font-size:26px; font-weight:700; color:#e2f5e9; line-height:1.2; }
    .kpi-sub   { font-size:12px; color:#6b9e7e; margin-top:2px; }

    .signal-strong { background:#14532d; color:#4ade80;
                     padding:3px 10px; border-radius:20px;
                     font-size:12px; font-weight:600; }
    .signal-watch  { background:#451a03; color:#fb923c;
                     padding:3px 10px; border-radius:20px;
                     font-size:12px; font-weight:600; }
    .signal-neutral{ background:#1e2a1e; color:#a7c4b0;
                     padding:3px 10px; border-radius:20px;
                     font-size:12px; font-weight:600; }
    .signal-avoid  { background:#450a0a; color:#f87171;
                     padding:3px 10px; border-radius:20px;
                     font-size:12px; font-weight:600; }

    .card {
        background:#162419; border-radius:14px;
        padding:18px 22px; border:1px solid #1e3a28;
        margin-bottom:12px;
    }
    .card-title { font-size:15px; font-weight:600;
                  color:#e2f5e9; margin-bottom:10px;
                  border-bottom:1px solid #1e3a28; padding-bottom:8px; }

    .score-bar-bg  { background:#1e3a28; border-radius:6px;
                     height:10px; width:100%; }
    .score-bar-fill{ background:#4ade80; border-radius:6px; height:10px; }

    #MainMenu{visibility:hidden} footer{visibility:hidden} header{visibility:hidden}
    .block-container { padding:0 !important; max-width:100% !important; }
    div[data-testid="stHorizontalBlock"] { padding:0 32px; gap:14px; }

    section[data-testid="stSidebar"] {
        background:#0d1710; border-right:1px solid #1e3a28;
    }
</style>
""", unsafe_allow_html=True)

import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nepse.db")
DB_URL  = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL, echo=False)

CHART_BG   = "rgba(0,0,0,0)"
PLOT_BG    = "#162419"
GRID_COLOR = "#1e3a28"
TICK_COLOR = "#a7c4b0"


# ══════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_scores(target_date: str) -> pd.DataFrame:
    """Loads today's scored stocks from the database."""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT s.*, p.volume, p.week52_high, p.week52_low,
                   p.open, p.high, p.low
            FROM scores s
            LEFT JOIN daily_prices p
                ON s.symbol = p.symbol AND s.date = p.date
            WHERE s.date = :d
            ORDER BY s.total_score DESC, s.ltp DESC
        """), conn, params={"d": target_date})
    return df


@st.cache_data(ttl=300)
def load_price_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """Loads price history for a stock."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, ltp, high, low, open, volume, pct_change
            FROM daily_prices
            WHERE symbol = :sym AND date >= :cutoff
            ORDER BY date ASC
        """), conn, params={"sym": symbol, "cutoff": cutoff})
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_market_index(days: int = 60) -> pd.DataFrame:
    """Loads NEPSE index history."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, index_value FROM market_index
            WHERE date >= :cutoff ORDER BY date ASC
        """), conn, params={"cutoff": cutoff})
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_all_scores_history() -> pd.DataFrame:
    """Loads score history for backtesting section."""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, signal, COUNT(*) as count
            FROM scores
            GROUP BY date, signal
            ORDER BY date DESC
        """), conn)
    return df


def get_regime() -> tuple[str, str]:
    """Returns (regime, description) based on index data."""
    idx_df = load_market_index(30)
    if idx_df.empty or len(idx_df) < 5:
        return "NEUTRAL", "Insufficient index data — defaulting to neutral"

    current = idx_df["index_value"].iloc[-1]
    avg30   = idx_df["index_value"].mean()

    if current > avg30 * 1.02:
        return "BULL", f"NEPSE index ({current:.0f}) is above 30-day average ({avg30:.0f}) — market trending up"
    elif current < avg30 * 0.98:
        return "BEAR", f"NEPSE index ({current:.0f}) is below 30-day average ({avg30:.0f}) — market in downtrend"
    return "NEUTRAL", f"NEPSE index ({current:.0f}) near 30-day average ({avg30:.0f}) — market neutral"


def signal_badge(signal: str) -> str:
    cls = {
        "STRONG BUY": "signal-strong",
        "WATCH":      "signal-watch",
        "NEUTRAL":    "signal-neutral",
        "AVOID":      "signal-avoid",
    }.get(signal, "signal-neutral")
    return f'<span class="{cls}">{signal}</span>'


def score_bar(score: float, max_score: float = 12) -> str:
    pct = min(100, int(score / max_score * 100))
    colour = "#4ade80" if pct >= 58 else "#fb923c" if pct >= 40 else "#f87171"
    return f"""
    <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%; background:{colour}"></div>
    </div>
    <div style="font-size:11px; color:#6b9e7e; margin-top:2px">{score:.1f}/12</div>
    """


# ══════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════
def render_navbar(today: str, total_stocks: int):
    st.markdown(f"""
    <div class="navbar">
        <div>
            <div class="nav-brand">💎 NEPSE Gem Finder</div>
            <div class="nav-sub">
                Data-driven stock screening for Nepali investors
            </div>
        </div>
        <div style="text-align:right; color:#6b9e7e; font-size:13px">
            📅 {today} &nbsp;|&nbsp; {total_stocks} stocks tracked
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame) -> tuple:
    with st.sidebar:
        st.markdown("### Filters")
        st.markdown("---")

        # signal filter
        signals = ["All"] + ["STRONG BUY", "WATCH", "NEUTRAL", "AVOID"]
        selected_signal = st.selectbox("Signal", signals)

        # sector filter
        sectors = ["All"] + sorted(
            [s for s in df["sector"].dropna().unique() if s and s != "Others"]
        ) + ["Others"]
        selected_sector = st.selectbox("Sector", sectors)

        # score range
        score_range = st.slider("Min score", 0, 12, 0)

        # refresh
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("**Score legend**")
        for sig, col, pts in [
            ("STRONG BUY", "#4ade80", "7–12"),
            ("WATCH",      "#fb923c", "5–6"),
            ("NEUTRAL",    "#a7c4b0", "3–4"),
            ("AVOID",      "#f87171", "0–2"),
        ]:
            st.markdown(
                f'<span style="color:{col}">●</span> {sig} ({pts} pts)',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("**Data status**")
        st.caption("📡 Prices: merolagani.com")
        st.caption("🕐 Updated daily after 4pm NPT")
        st.caption("⚠️ NOT financial advice")

    return selected_signal, selected_sector, score_range


# ══════════════════════════════════════════════════════════════════
# SECTION 1: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════
def render_market_overview(df: pd.DataFrame, regime: str, regime_desc: str):
    st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)

    # regime banner
    regime_class = {
        "BULL":    "regime-bull",
        "BEAR":    "regime-bear",
        "NEUTRAL": "regime-neutral"
    }.get(regime, "regime-neutral")

    regime_icon = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(regime, "🟡")
    st.markdown(f"""
    <div class="{regime_class}" style="margin-bottom:16px">
        <b>{regime_icon} Market Regime: {regime}</b> &nbsp;—&nbsp; {regime_desc}
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    total  = len(df)
    gainers = len(df[df["pct_change"] > 0]) if "pct_change" in df.columns else 0
    losers  = len(df[df["pct_change"] < 0]) if "pct_change" in df.columns else 0
    strong_buys = len(df[df["signal"] == "STRONG BUY"])
    watches     = len(df[df["signal"] == "WATCH"])
    avg_score   = df["total_score"].mean() if not df.empty else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "Total Stocks",   str(total),       "tracked today"),
        (k2, "Gainers",        str(gainers),      f"{round(gainers/total*100)}% of market"),
        (k3, "Losers",         str(losers),       f"{round(losers/total*100)}% of market"),
        (k4, "Strong Buys",    str(strong_buys),  f"+ {watches} on Watch"),
        (k5, "Avg Score",      f"{avg_score:.1f}", "out of 12"),
    ]
    for col, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # top movers row
    col_gain, col_lose, col_vol = st.columns(3)

    with col_gain:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Top Gainers Today</div>',
                    unsafe_allow_html=True)
        gainers_df = df[df["pct_change"] > 0].nlargest(5, "pct_change")
        for _, row in gainers_df.iterrows():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:0.5px solid #1e3a28">'
                f'<span style="color:#e2f5e9;font-weight:500">{row["symbol"]}</span>'
                f'<span style="color:#4ade80">+{row["pct_change"]:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_lose:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 Top Losers Today</div>',
                    unsafe_allow_html=True)
        losers_df = df[df["pct_change"] < 0].nsmallest(5, "pct_change")
        for _, row in losers_df.iterrows():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:0.5px solid #1e3a28">'
                f'<span style="color:#e2f5e9;font-weight:500">{row["symbol"]}</span>'
                f'<span style="color:#f87171">{row["pct_change"]:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vol:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔥 Most Active (Volume)</div>',
                    unsafe_allow_html=True)
        vol_df = df[df["volume"] > 0].nlargest(5, "volume") \
            if "volume" in df.columns else pd.DataFrame()
        if not vol_df.empty:
            for _, row in vol_df.iterrows():
                vol_fmt = f"{int(row['volume']):,}"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 0;border-bottom:0.5px solid #1e3a28">'
                    f'<span style="color:#e2f5e9;font-weight:500">{row["symbol"]}</span>'
                    f'<span style="color:#a7c4b0">{vol_fmt}</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2: GEMS TABLE
# ══════════════════════════════════════════════════════════════════
def render_gems_table(df: pd.DataFrame, sig_filter: str,
                      sec_filter: str, min_score: float):
    st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 💎 Stocks Ranked by Score")

    # apply filters
    filtered = df.copy()
    if sig_filter != "All":
        filtered = filtered[filtered["signal"] == sig_filter]
    if sec_filter != "All":
        filtered = filtered[filtered["sector"] == sec_filter]
    filtered = filtered[filtered["total_score"] >= min_score]

    if filtered.empty:
        st.info("No stocks match your filters. Try adjusting the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.caption(f"Showing {len(filtered)} stocks (out of {len(df)} scored today)")

    # render table header
    h1, h2, h3, h4, h5, h6, h7 = st.columns([1, 2, 2, 2, 2, 3, 3])
    for col, label in zip([h1,h2,h3,h4,h5,h6,h7],
                          ["#","Symbol","LTP","Change","Score","Signal","Sector"]):
        col.markdown(f"**{label}**")

    st.markdown(
        '<hr style="border:0.5px solid #1e3a28; margin:4px 0 8px">',
        unsafe_allow_html=True
    )

    for i, (_, row) in enumerate(filtered.iterrows(), 1):
        c1,c2,c3,c4,c5,c6,c7 = st.columns([1,2,2,2,2,3,3])
        chg     = row.get("pct_change")
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        chg_col = "#4ade80" if (chg or 0) > 0 else "#f87171" if (chg or 0) < 0 else "#a7c4b0"

        c1.markdown(f"<span style='color:#6b9e7e'>{i}</span>",
                    unsafe_allow_html=True)
        c2.markdown(f"**{row['symbol']}**")
        c3.markdown(f"Rs {row['ltp']:,.1f}")
        c4.markdown(f"<span style='color:{chg_col}'>{chg_str}</span>",
                    unsafe_allow_html=True)
        c5.markdown(score_bar(row["total_score"]), unsafe_allow_html=True)
        c6.markdown(signal_badge(row["signal"]), unsafe_allow_html=True)
        c7.markdown(f"<span style='color:#6b9e7e'>{row.get('sector','Unknown')}</span>",
                    unsafe_allow_html=True)

        if i < len(filtered):
            st.markdown(
                '<hr style="border:0.5px solid #1e3a28; margin:2px 0">',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 3: STOCK DEEP DIVE
# ══════════════════════════════════════════════════════════════════
def render_deep_dive(df: pd.DataFrame):
    st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🔍 Stock Deep Dive")

    symbols = df["symbol"].tolist()
    selected = st.selectbox(
        "Select a stock to analyse",
        options=symbols,
        format_func=lambda s: f"{s} — Rs {df[df['symbol']==s]['ltp'].values[0]:,.1f}"
        if len(df[df["symbol"]==s]) > 0 else s,
        key="deep_dive_select"
    )

    if not selected:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    stock = df[df["symbol"] == selected].iloc[0]
    hist  = load_price_history(selected, days=90)

    # top metrics
    m1,m2,m3,m4,m5 = st.columns(5)
    metrics = [
        ("Price",    f"Rs {stock['ltp']:,.1f}", None),
        ("Score",    f"{stock['total_score']:.1f}/12", None),
        ("Signal",   stock["signal"], None),
        ("Sector",   stock.get("sector","Unknown"), None),
        ("Change",   f"{stock['pct_change']:+.2f}%" if stock.get('pct_change') else "N/A", None),
    ]
    for col, (label, val, _) in zip([m1,m2,m3,m4,m5], metrics):
        col.metric(label, val)

    left_col, right_col = st.columns([6, 4], gap="medium")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">Price History — {selected}</div>',
                    unsafe_allow_html=True)

        if hist.empty or len(hist) < 2:
            st.info(
                f"Only {len(hist)} day(s) of price data available for {selected}. "
                f"Keep running the pipeline daily — charts will fill in automatically."
            )
        else:
            fig = go.Figure()

            # price line
            fig.add_trace(go.Scatter(
                x=hist["date"], y=hist["ltp"],
                mode="lines+markers",
                name="LTP",
                line=dict(color="#4ade80", width=2),
                marker=dict(size=5)
            ))

            # MA7 if enough data
            if len(hist) >= 7:
                hist["ma7"] = hist["ltp"].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=hist["date"], y=hist["ma7"],
                    mode="lines", name="MA7",
                    line=dict(color="#fb923c", width=1.5, dash="dash")
                ))

            fig.update_layout(
                paper_bgcolor=CHART_BG,
                plot_bgcolor=PLOT_BG,
                font_color=TICK_COLOR,
                height=280,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR)),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                           title="Price (Rs)"),
                legend=dict(bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2f5e9")),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Score Breakdown</div>',
                    unsafe_allow_html=True)

        # score breakdown bar chart
        rules = [
            ("52-week position",  stock.get("rule2_points", 0), 3),
            ("RSI momentum",      stock.get("rule3_points", 0), 3),
            ("MA trend",          stock.get("rule4_points", 0), 3),
            ("Volume signal",     stock.get("rule5_points", 0), 2),
            ("Market regime",     stock.get("rule6_points", 0), 1),
        ]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=[r[0] for r in rules],
            x=[r[1] for r in rules],
            orientation="h",
            marker_color=["#4ade80" if r[1]>0 else "#1e3a28" for r in rules],
            text=[f"{r[1]}/{r[2]}" for r in rules],
            textposition="outside",
            textfont=dict(color="#e2f5e9", size=11),
        ))
        fig_bar.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=PLOT_BG,
            font_color=TICK_COLOR,
            height=220,
            margin=dict(l=10,r=50,t=10,b=10),
            xaxis=dict(range=[0,4], gridcolor=GRID_COLOR,
                       tickfont=dict(color=TICK_COLOR)),
            yaxis=dict(tickfont=dict(color="#e2f5e9", size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # plain english explanation
        explanation = stock.get("plain_english", "")
        if explanation:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Plain English Explanation</div>',
                        unsafe_allow_html=True)
            for line in explanation.split("\n"):
                if line.strip():
                    colour = "#a7c4b0" if not line.startswith("⚠️") else "#fb923c"
                    st.markdown(
                        f'<div style="font-size:13px;color:{colour};'
                        f'line-height:1.6;margin-bottom:4px">{line}</div>',
                        unsafe_allow_html=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════
def render_how_it_works():
    st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📖 How the scoring system works", expanded=False):
        st.markdown("""
        ### How NEPSE Gem Finder scores stocks

        Every stock is scored **0–12 points** across 5 rules.
        Higher score = more signals pointing to a potential opportunity.

        | Rule | What it checks | Max points |
        |------|---------------|-----------|
        | **52-week position** | Is the price near its yearly low? (cheap relative to history) | 3 pts |
        | **RSI momentum** | Is the stock oversold AND recovering? (not falling further) | 3 pts |
        | **MA trend** | Is price stabilising above its 7-day average? | 3 pts |
        | **Volume signal** | Is there buying interest (high volume on up days)? | 2 pts |
        | **Market regime** | Is the overall NEPSE market healthy? | 1 pt |

        ### Signal thresholds
        - 🟢 **STRONG BUY** (7–12 pts): Multiple signals align — worth researching further
        - 🟡 **WATCH** (5–6 pts): Some positive signals — keep an eye on it
        - ⚪ **NEUTRAL** (3–4 pts): Mixed signals — no strong reason to act
        - 🔴 **AVOID** (0–2 pts): Few or no positive signals

        ### Important limitations
        - RSI requires **14+ days** of price data — new for stocks just added
        - MA200 requires **200 days** of data — will improve over time
        - P/E and EPS are not yet available (NEPSE sites load these via JavaScript)
        - **This tool screens stocks, it does NOT predict prices**

        ### ⚠️ Disclaimer
        This is a **data-driven educational tool only**.
        It is NOT financial advice. Stock markets involve real risk of loss.
        Always do your own research, consult a qualified financial advisor,
        and never invest money you cannot afford to lose.
        """)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
    <div style="text-align:center;padding:20px;font-size:12px;color:#4a7a5a;
                border-top:1px solid #1e3a28;margin-top:16px">
        © 2026 NEPSE GEM FINDER &nbsp;·&nbsp;
        BUILT WITH PYTHON, STREAMLIT & PLOTLY &nbsp;·&nbsp;
        DATA: MEROLAGANI.COM &nbsp;·&nbsp;
        ⚠️ NOT FINANCIAL ADVICE
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    # use the most recent date that has data
    # this avoids timezone issues between Streamlit Cloud (UTC)
    # and Nepal time (UTC+5:45)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date) FROM scores"))
        latest_date = result.fetchone()[0]

    today = latest_date if latest_date else date.today().isoformat()
    df    = load_scores(today)
    
    regime, regime_desc = get_regime()

    render_navbar(today, len(df))

    if df.empty:
        st.markdown("<div style='padding:40px 32px'>", unsafe_allow_html=True)
        st.warning(
            f"No scored data found for today ({today}). "
            f"Run the pipeline first: `python pipeline.py --once` "
            f"then `python scorer.py`"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # sidebar filters
    sig_filter, sec_filter, min_score = render_sidebar(df)

    # sections
    render_market_overview(df, regime, regime_desc)
    render_gems_table(df, sig_filter, sec_filter, min_score)
    if not df.empty:
        render_deep_dive(df)
    render_how_it_works()
    render_footer()


if __name__ == "__main__":
    main()