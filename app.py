import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import time
import engine as eng
from tickers import MASTER_MAP

# 1. PAGE CONFIG
st.set_page_config(page_title="Market Monitor v0.5.6", layout="wide")

# 2. UI STYLE & BENCHMARK CSS
st.markdown("""
    <style>
    .block-container { max-width: 98% !important; padding: 3.5rem 1rem 1rem 1rem !important; }
    section[data-testid="stSidebar"] { background-color: #f1f3f6 !important; }
    
    /* Sidebar Headers */
    .sidebar-header {
        font-size: 0.85rem !important; font-weight: 800 !important; color: #1e3a8a !important;
        text-transform: uppercase; letter-spacing: 1px; padding: 10px 5px 5px 0px;
        margin-top: 1.2rem !important; border-bottom: 2px solid #3b82f6; display: block;
    }
    
    /* Benchmark Alerts & Labels */
    .snapshot-label {
        font-size: 0.75rem; font-weight: 700; color: #b91c1c; background-color: #fee2e2;
        padding: 4px 10px; border-radius: 6px; margin-top: 10px; display: inline-block; border: 1px solid #fecaca;
    }
    
    /* Loading Sequence Design */
    .loading-box { background-color: #fff3e0; border: 1px solid #ffe0b2; padding: 20px; border-radius: 10px; text-align: center; }
    .load-timer { font-size: 2.2rem; font-weight: 800; color: #e65100; font-family: monospace; }
    .phase-active { color: #d35400; font-weight: 700; font-size: 0.9rem; }
    .phase-done { color: #27ae60; font-weight: 700; font-size: 0.9rem; }

    /* Segmented Control Styling */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #1e3a8a !important; color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. BENCHMARK STATE INITIALIZATION
if "market_df" not in st.session_state: st.session_state.market_df = pd.DataFrame()
if "watchlist" not in st.session_state: st.session_state.watchlist = eng.load_watchlist()

TOTAL_MASTER_COUNT = len(MASTER_MAP)
MASTER_TICKERS = list(MASTER_MAP.keys())

# 4. SIDEBAR RENDERING
st.sidebar.markdown('<p class="sidebar-header">📊 MARKET STATUS</p>', unsafe_allow_html=True)
with st.sidebar.container():
    stat_col1, stat_col2 = st.sidebar.columns(2)
    metric_stocks, metric_nifty = stat_col1.empty(), stat_col2.empty()
    if "fundamentals_time" in st.session_state:
        st.sidebar.markdown(f'<div class="snapshot-label">🕒 Snapshot: {st.session_state.fundamentals_time}</div>', unsafe_allow_html=True)

# NIFTY REAL-TIME CARD
try:
    n_df = yf.download("^NSEI", period="5d", progress=False)
    if not n_df.empty:
        cn, pn = n_df["Close"].iloc[-1], n_df["Close"].iloc[-2]
        chg_pct = ((cn - pn) / pn) * 100
        border = "#27ae60" if chg_pct >= 0 else "#e74c3c"
        n_html = f'<div style="background-color:#fff; padding:8px 10px; border-radius:5px; border-left:4px solid {border};"><div style="font-size:0.7rem; color:#555; font-weight:bold;">NIFTY 50</div><div style="display:flex; justify-content:space-between;"><span style="font-size:0.95rem; font-weight:800;">{cn:.0f}</span><span style="font-size:0.85rem; font-weight:700; color:{border};">{chg_pct:+.2f}%</span></div></div>'
    else: n_html = "Offline"
except: n_html = "Offline"
metric_nifty.markdown(n_html, unsafe_allow_html=True)

active_count = len(st.session_state.market_df) if not st.session_state.market_df.empty else 0
metric_stocks.markdown(f'<div style="background-color:#ebf5fb; padding:8px 10px; border-radius:5px; border-left:4px solid #3498db;"><div style="font-size:0.7rem; color:#555; font-weight:bold;">Active Scripts</div><div style="font-size:0.95rem; font-weight:800; color:#2980b9;">{active_count} / {TOTAL_MASTER_COUNT}</div></div>', unsafe_allow_html=True)

# PERSISTENT WATCHLIST CONTROLS
st.sidebar.markdown('<p class="sidebar-header">⭐ WATCHLIST</p>', unsafe_allow_html=True)
show_favs = st.sidebar.toggle("Show Favorites Only", value=False)
target_to_star = st.sidebar.selectbox("Star/Unstar Stock", options=[""] + sorted(MASTER_TICKERS))
if st.sidebar.button("Update Star Status", use_container_width=True) and target_to_star:
    is_adding = target_to_star not in st.session_state.watchlist
    eng.save_to_watchlist(target_to_star, add=is_adding)
    st.session_state.watchlist = eng.load_watchlist()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-header">🖥️ VIEW CONFIG</p>', unsafe_allow_html=True)
view_mode = st.sidebar.segmented_control("Mode", options=["Mobile", "Desktop"], default="Desktop", label_visibility="collapsed")
mode_key = view_mode.lower() if view_mode else "desktop"

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-header">🔍 DATA FILTERS</p>', unsafe_allow_html=True)
ref_date = st.sidebar.date_input("Ref Date", value=datetime.date(2023, 12, 31))
search_q = st.sidebar.text_input("Live Search", placeholder="Ticker / Name / Sector...")
trend_view = st.sidebar.radio("Trend", ["All", "Green", "Red"], index=0, horizontal=True)
view_filter = st.sidebar.selectbox("View", ["All", "Vol Breakout", "Near 52W High (<= 5%)", "Near 52W Low (>= -5%)"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-header">⚙️ CONTROLS</p>', unsafe_allow_html=True)
status_footer_placeholder = st.sidebar.empty()
c1, c2 = st.sidebar.columns(2)
if c1.button("Refresh", use_container_width=True): st.session_state.market_df = pd.DataFrame(); st.rerun()
if c2.button("Reset", use_container_width=True): st.cache_data.clear(); st.rerun()

# 5. INITIAL FETCH (Phases 1-3)
if st.session_state.market_df.empty:
    container = st.empty()
    start_time = time.time()
    with container.container():
        st.markdown('<div class="loading-box">', unsafe_allow_html=True)
        clock, status = st.empty(), st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        try:
            status.markdown('<p class="phase-active">⏳ Phase 1: Downloading 2y History...</p>', unsafe_allow_html=True)
            raw = st.cache_data(eng.download_bulk_history)(MASTER_TICKERS)
            t1 = time.time() - start_time
            status.markdown(f'<p class="phase-done">✅ Phase 1 ({t1:.1f}s)</p><p class="phase-active">⏳ Phase 2: Analysis...</p>', unsafe_allow_html=True)
            clock.markdown(f'<div class="load-timer">{t1:.1f}s</div>', unsafe_allow_html=True)
            base = eng.calculate_baselines(MASTER_TICKERS, raw, ref_date)
            t2 = time.time() - start_time
            status.markdown(f'<p class="phase-done">✅ Phase 2 ({t2:.1f}s)</p><p class="phase-active">⏳ Phase 3: Live Prices...</p>', unsafe_allow_html=True)
            clock.markdown(f'<div class="load-timer">{t2:.1f}s</div>', unsafe_allow_html=True)
            
            # --- FIXED MOBILE BUTTON LOGIC ---
            df, _, _ = eng.get_live_data(MASTER_TICKERS, base, set(), mode=mode_key)
            
            st.session_state.market_df = df
            st.session_state.total_load_time = f"{time.time() - start_time:.2f}s"
            container.empty(); st.rerun()
        except Exception as e: st.error(f"Error: {e}")

# 6. MAIN TABLE (Benchmark Display & Sorting)
if not st.session_state.market_df.empty:
    # --- SAFE MISSING TICKER DETECTOR ---
    master_set = set(MASTER_TICKERS)
    loaded_set = set(st.session_state.market_df['TickerID'].unique())
    missing = list(master_set - loaded_set)
    if missing:
        st.sidebar.warning(f"⚠️ {len(missing)} Stocks Failed")
        with st.sidebar.expander("View Missing"): st.write(missing)

    active = st.session_state.market_df.copy()
    active["⭐"] = active["TickerID"].apply(lambda x: "⭐" if x in st.session_state.watchlist else "")
    
    # Apply Basic Filters
    if show_favs: active = active[active["⭐"] == "⭐"]
    if search_q: active = active[active["Name"].str.contains(search_q, case=False) | active["Sector"].str.contains(search_q, case=False)]
    if trend_view == "Green": active = active[active["Change %"] > 0]
    elif trend_view == "Red": active = active[active["Change %"] < 0]
    
    # Apply View Filters
    if view_filter == "Vol Breakout": active = active[active["Vol Breakout"] >= 1.5]
    elif view_filter == "Near 52W High (<= 5%)": active = active[active["vs 52W High (%)"] >= -5]
    elif view_filter == "Near 52W Low (>= -5%)": active = active[active["vs 52W Low (%)"] <= 5]

    # --- BENCHMARK SORTING: FLOATING STARS ---
    # Priorities: 1. Starred, 2. Name
    active["sort_order"] = active["⭐"].apply(lambda x: 0 if x == "⭐" else 1)
    active = active.sort_values(by=["sort_order", "Name"], ascending=[True, True])
    active = active.drop(columns=["sort_order"])
    active.insert(0, "#", range(1, len(active) + 1))
    
    # Formatting Logic
    order = ["⭐", "#", "Name", "Sector", "LTP", "Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 52W High (%)", "vs 52W Low (%)", "Up/Low since", "RSI (14)", "Vol Breakout", "vs 100DMA (%)", "Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]
    color_cols = ["Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 52W High (%)", "vs 52W Low (%)", "Up/Low since", "vs 100DMA (%)"]

    def apply_color(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return "color: black;"
        return f"color: {'#27ae60' if val > 0 else '#e74c3c'}; font-weight: bold;"

    styled_df = (active[order].style
                 .applymap(apply_color, subset=color_cols)
                 .format(precision=1, subset=color_cols + ["Vol Breakout", "RSI (14)", "Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]))

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=850,
        column_config={
            "⭐": st.column_config.TextColumn("⭐", width=35, pinned=True),
            "#": st.column_config.NumberColumn("#", width=35, pinned=True),
            "Name": st.column_config.TextColumn("Name", width=180, pinned=True),
            "LTP": st.column_config.NumberColumn("LTP", format="%.1f")
        })

    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    status_footer_placeholder.markdown(f'<div style="font-size:0.65rem; color:#888; margin-bottom:10px;">⏱️ Load: {st.session_state.get("total_load_time", "N/A")} | 🔄 Sync: {now_str}</div>', unsafe_allow_html=True)

    # 7. BACKGROUND FETCH (Benchmark Persistence)
    if st.session_state.market_df["Market Cap ($M)"].isnull().all():
        with st.status("Fetching Fundamentals...", expanded=False):
            f_map = eng.fetch_fundamentals_map(MASTER_TICKERS, eng.get_usd_rate())
            st.session_state.fundamentals_time = datetime.datetime.now().strftime("%H:%M")
            for t, v in f_map.items():
                for col, val in v.items(): st.session_state.market_df.loc[st.session_state.market_df['TickerID'] == t, col] = val
            st.rerun()
