import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import time
from streamlit_js_eval import streamlit_js_eval
import engine as eng
from tickers import MASTER_MAP

# 1. PAGE CONFIG
st.set_page_config(page_title="Market Monitor v0.5.6", layout="wide")

# 2. UI STYLE & CSS
st.markdown("""
    <style>
    .block-container { max-width: 98% !important; padding: 3.5rem 1rem 1rem 1rem !important; }
    section[data-testid="stSidebar"] { background-color: #f1f3f6 !important; }
    .sidebar-header { font-size: 0.85rem !important; font-weight: 800 !important; color: #1e3a8a !important; text-transform: uppercase; border-bottom: 2px solid #3b82f6; display: block; margin-top: 1.2rem !important; }
    .snapshot-label { font-size: 0.75rem; font-weight: 700; color: #b91c1c; background-color: #fee2e2; padding: 4px 10px; border-radius: 6px; margin-top: 10px; display: inline-block; border: 1px solid #fecaca; }
    [data-testid="stDataEditor"] div[role="columnheader"], [data-testid="stDataFrame"] div[role="columnheader"] { height: 105px !important; min-height: 105px !important; }
    [data-testid="stDataEditor"] div[role="columnheader"] p, [data-testid="stDataFrame"] div[role="columnheader"] p { white-space: normal !important; word-wrap: break-word !important; line-height: 1.1 !important; font-size: 0.72rem !important; text-align: center !important; overflow: visible !important; }
    </style>
""", unsafe_allow_html=True)

# 3. INITIALIZATION
if "market_df" not in st.session_state: st.session_state.market_df = pd.DataFrame()
if "watchlist" not in st.session_state: st.session_state.watchlist = eng.load_watchlist()
MASTER_TICKERS = list(MASTER_MAP.keys())
TOTAL_MASTER_COUNT = len(MASTER_MAP)

# 4. DEVICE DETECTION
ua = streamlit_js_eval(js_expressions='window.navigator.userAgent', key='UA')
is_mobile_detected = any(x in str(ua).lower() for x in ["mobile", "android", "iphone"]) if ua else False

st.sidebar.markdown('<p class="sidebar-header">🖥️ VIEW CONFIG</p>', unsafe_allow_html=True)
manual_desktop = st.sidebar.toggle("Force Desktop View on Mobile", value=False)
lite_mode = is_mobile_detected and not manual_desktop

if lite_mode:
    FETCH_PERIOD = "1mo"
    order = ["#", "Name", "Sector", "LTP", "Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 30D High (%)", "vs 30D Low (%)"]
    color_cols = ["Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 30D High (%)", "vs 30D Low (%)"]
    st.sidebar.info("📱 Mobile Lite Mode (30-Day Fetch)")
else:
    FETCH_PERIOD = "2y"
    order = ["⭐", "#", "Name", "Sector", "LTP", "Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 52W High (%)", "vs 52W Low (%)", "Up/Low since", "RSI (14)", "Vol Breakout", "vs 100DMA (%)", "Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]
    color_cols = ["Change %", "vs 7D High (%)", "vs 7D Low (%)", "vs 15D High (%)", "vs 15D Low (%)", "vs 52W High (%)", "vs 52W Low (%)", "Up/Low since", "vs 100DMA (%)"]

# 5. SIDEBAR RENDERING
st.sidebar.markdown('<p class="sidebar-header">📊 MARKET STATUS</p>', unsafe_allow_html=True)
stat_col1, stat_col2 = st.sidebar.columns(2)
metric_stocks, metric_nifty = stat_col1.empty(), stat_col2.empty()

if "fundamentals_time" in st.session_state:
    st.sidebar.markdown(f'<div class="snapshot-label">🕒 Snapshot: {st.session_state.fundamentals_time}</div>', unsafe_allow_html=True)

try:
    n_df = yf.download("^NSEI", period="5d", progress=False)
    if not n_df.empty:
        cn, pn = n_df["Close"].iloc[-1], n_df["Close"].iloc[-2]
        chg_pct = ((cn - pn) / pn) * 100
        clr = "#27ae60" if chg_pct >= 0 else "#e74c3c"
        metric_nifty.markdown(f'<div style="background-color:#fff; padding:8px 10px; border-radius:5px; border-left:4px solid {clr};"><div style="font-size:0.7rem; color:#555; font-weight:bold;">NIFTY 50</div><div style="display:flex; justify-content:space-between;"><span style="font-size:0.95rem; font-weight:800;">{cn:.0f}</span><span style="font-size:0.85rem; font-weight:700; color:{clr};">{chg_pct:+.2f}%</span></div></div>', unsafe_allow_html=True)
except: metric_nifty.write("Nifty Offline")

active_count = len(st.session_state.market_df) if not st.session_state.market_df.empty else 0
metric_stocks.markdown(f'<div style="background-color:#ebf5fb; padding:8px 10px; border-radius:5px; border-left:4px solid #3498db;"><div style="font-size:0.7rem; color:#555; font-weight:bold;">Active Scripts</div><div style="font-size:0.95rem; font-weight:800; color:#2980b9;">{active_count} / {TOTAL_MASTER_COUNT}</div></div>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-header">⭐ WATCHLIST</p>', unsafe_allow_html=True)
show_favs = st.sidebar.toggle("Show Favorites Only", value=False)
target_to_star = st.sidebar.selectbox("Star/Unstar Stock", options=[""] + sorted(MASTER_TICKERS))
if st.sidebar.button("Update Star Status", use_container_width=True) and target_to_star:
    is_adding = target_to_star not in st.session_state.watchlist
    eng.save_to_watchlist(target_to_star, add=is_adding)
    st.session_state.watchlist = eng.load_watchlist()
    st.rerun()

st.sidebar.markdown('<p class="sidebar-header">🔍 DATA FILTERS</p>', unsafe_allow_html=True)
ref_date = st.sidebar.date_input("Ref Date", value=datetime.date(2023, 12, 31))
search_q = st.sidebar.text_input("Live Search", placeholder="Ticker / Sector...")
trend_view = st.sidebar.radio("Trend", ["All", "Green", "Red"], horizontal=True)
view_filter = st.sidebar.selectbox("View", ["All", "Vol Breakout", "Near 52W High (<= 5%)", "Near 52W Low (>= -5%)"], index=0)

st.sidebar.markdown('<p class="sidebar-header">⚙️ CONTROLS</p>', unsafe_allow_html=True)
status_footer_placeholder = st.sidebar.empty()
c1, c2 = st.sidebar.columns(2)
if c1.button("Refresh", use_container_width=True): st.session_state.market_df = pd.DataFrame(); st.rerun()
if c2.button("Reset", use_container_width=True): st.cache_data.clear(); st.rerun()

# 6. INITIAL FETCH (With Granular Phase Tracking)
if st.session_state.market_df.empty:
    start_time = time.time()
    # Using st.status to provide a persistent heartbeat to the browser
    with st.status(f"🚀 Phase 1: Fetching {FETCH_PERIOD} History...", expanded=True) as status:
        try:
            # TRACKER 1: Connection Phase
            st.write("📡 Step 1: Connecting to Market Data Gateway...")
            raw = st.cache_data(eng.download_bulk_history)(MASTER_TICKERS, period=FETCH_PERIOD)
            
            if raw.empty:
                st.error("⚠️ Phase 1 Failed: No data returned from Gateway. Check for Rate Limits.")
                st.stop()
            
            # TRACKER 2: Baseline Calculation
            st.write(f"🔢 Step 2: Calculating Baselines for {len(MASTER_TICKERS)} scripts...")
            base = eng.calculate_baselines(MASTER_TICKERS, raw, ref_date)
            
            # TRACKER 3: Live Snapshot Phase
            st.write("📊 Step 3: Finalizing Live Snapshot View...")
            df, _, _ = eng.get_live_data(MASTER_TICKERS, base, set())
            
            # TRACKER 4: Mobile Lite mode Post-Processing
            if lite_mode:
                st.write("📱 Step 4: Trimming View for Mobile Lite Mode...")
                for x in MASTER_TICKERS:
                    if x in base and not df.loc[df['TickerID']==x].empty:
                        ltp = df.loc[df['TickerID']==x, 'LTP'].values[0]
                        h30, l30 = base[x].get('30H'), base[x].get('30L')
                        df.loc[df['TickerID']==x, "vs 30D High (%)"] = ((ltp - h30)/h30)*100 if h30 else 0
                        df.loc[df['TickerID']==x, "vs 30D Low (%)"] = ((ltp - l30)/l30)*100 if l30 else 0
            
            st.session_state.market_df = df
            st.session_state.total_load_time = f"{time.time() - start_time:.2f}s"
            
            # TRACKER 5: Completion
            status.update(label=f"✅ Data Synced in {st.session_state.total_load_time}!", state="complete", expanded=False)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Phase Tracker Error: {type(e).__name__} at {e}")
            # This logs the specific failure point for our next debug iteration
            print(f"DEBUG: App hung at Step {status}")

# 7. MAIN TABLE (Strictly Limited for Mobile/Lite Mode)
if not st.session_state.market_df.empty:
    
    # --- 7.1 DATA PREP & TIMEZONE NEUTRALITY ---
    active = st.session_state.market_df.copy()
    if isinstance(active.index, pd.DatetimeIndex):
        active.index = active.index.tz_localize(None) # Prevents hosted link crash
    active["⭐"] = active["TickerID"].apply(lambda x: "⭐" if x in st.session_state.watchlist else "")
    
    # --- 7.2 APPLY FILTERS ---
    if show_favs: active = active[active["⭐"] == "⭐"]
    if search_q: 
        active = active[active["Name"].str.contains(search_q, case=False) | 
                        active["Sector"].str.contains(search_q, case=False)]
    if trend_view == "Green": active = active[active["Change %"] > 0]
    elif trend_view == "Red": active = active[active["Change %"] < 0]
    
    # Only apply extended filters if NOT in Lite Mode
    if not lite_mode:
        if "Vol Breakout" in active.columns and view_filter == "Vol Breakout": 
            active = active[active["Vol Breakout"] >= 1.5]
        elif "vs 52W High (%)" in active.columns and view_filter == "Near 52W High (<= 5%)": 
            active = active[active["vs 52W High (%)"] >= -5]

    # --- 7.3 SORTING & INDEXING ---
    active["sort_order"] = active["⭐"].apply(lambda x: 0 if x == "⭐" else 1)
    active = active.sort_values(by=["sort_order", "Name"], ascending=[True, True])
    active = active.reset_index(drop=True)
    active.insert(0, "#", range(1, len(active) + 1))

    # --- 7.4 DEFENSIVE STYLING (Strict Column Filtering) ---
    # KEY FIX: Dynamically identify existing columns to prevent KeyError
    existing_order = [c for c in order if c in active.columns]
    existing_color_cols = [c for c in color_cols if c in active.columns]
    
    # Only format numeric columns that exist in the current dataframe
    potential_fmt = color_cols + ["Vol Breakout", "RSI (14)", "Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]
    existing_fmt_cols = [c for c in potential_fmt if c in active.columns]

    def apply_color(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return "color: black;"
        return f"color: {'#27ae60' if val > 0 else '#e74c3c'}; font-weight: bold;"

    # Switched to .map() as required by Streamlit 2026 logs
    styled_df = (active[existing_order].style
                 .map(apply_color, subset=existing_color_cols)
                 .format(precision=1, subset=existing_fmt_cols))

    # --- 7.5 RENDER TABLE (Removed 'pinned' to fix TypeError) ---
    # Streamlit 1.19.0 on hosted server does not support 'pinned'
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=850,
        column_config={
            "⭐": st.column_config.TextColumn("⭐"),
            "#": st.column_config.NumberColumn("#"),
            "Name": st.column_config.TextColumn("Name"),
            "LTP": st.column_config.NumberColumn("LTP", format="%.1f")
        })

    # --- 7.6 FOOTER & SNAPSHOT ---
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    status_footer_placeholder.markdown(f'<div style="font-size:0.65rem; color:#888;">⏱️ Load: {st.session_state.get("total_load_time", "N/A")} | 🔄 Sync: {now_str}</div>', unsafe_allow_html=True)

    # Background Fundamentals: Only for Full Desktop Mode
    if not lite_mode and st.session_state.market_df["Market Cap ($M)"].isnull().all():
        with st.status("Fetching Fundamentals...", expanded=False) as fundamental_status:
            try:
                f_map = eng.fetch_fundamentals_map(MASTER_TICKERS, eng.get_usd_rate())
                st.session_state.fundamentals_time = datetime.datetime.now().strftime("%H:%M")
                for t, v in f_map.items():
                    for col in v:
                        st.session_state.market_df.loc[st.session_state.market_df['TickerID'] == t, col] = v[col]
                fundamental_status.update(label="Fundamentals Updated", state="complete")
                st.rerun()
            except: fundamental_status.update(label="Sync Interrupted", state="error")
