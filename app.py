import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from tickers import MASTER_MAP

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(page_title="Market Monitor", layout="wide", page_icon="📈")

# CSS: Fix layout for Phone & Desktop
st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        [data-testid="stDataFrame"], [data-testid="stMarkdownContainer"] p {
            color: #000000 !important; /* Force Black Text */
        }
        h1 {
            font-size: 1.6rem !important;
            margin-bottom: 0.5rem !important;
        }
        /* Ensure Sidebar is on top on mobile */
        section[data-testid="stSidebar"] {
            z-index: 99999;
        }
    </style>
""", unsafe_allow_html=True)

DEFAULT_TICKERS = list(MASTER_MAP.keys())

# ----------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def get_usd_rate():
    try:
        df = yf.download("USDINR=X", period="1d", progress=False)
        return float(df['Close'].iloc[-1])
    except: return 84.0

def clean_ticker_list(tickers):
    cleaned = []
    for t in tickers:
        t = t.upper().strip()
        if not (t.endswith('.NS') or t.endswith('.BO')): t += '.NS'
        cleaned.append(t)
    return list(set(cleaned))

def fetch_fundamental_single(ticker, usd_rate):
    try:
        info = yf.Ticker(ticker).info
        mcap = info.get('marketCap', np.nan)
        curr = info.get('currency', 'INR')
        if not pd.isna(mcap):
            mcap = (mcap / usd_rate / 1000000) if curr == 'INR' else (mcap / 1000000)
            mcap = round(mcap, 2)
        return mcap, info.get('trailingPE', np.nan), info.get('trailingEps', np.nan), \
               info.get('priceToBook', np.nan), (info.get('dividendYield', np.nan) * 100) if info.get('dividendYield') else np.nan, \
               info.get('sector', None)
    except: return np.nan, np.nan, np.nan, np.nan, np.nan, None

# ----------------------------------------------------
# 3. DATA FETCHING (MODE-AWARE)
# ----------------------------------------------------
@st.cache_data(ttl=43200) 
def get_baselines_bulk(tickers, mode="mobile"):
    # Mobile Mode = Slower but safer batching
    chunk_size = 20 if mode == "mobile" else 100
    delay = 0.5 if mode == "mobile" else 0.1
    
    tickers = clean_ticker_list(tickers)
    baselines = {}
    if not tickers: return {}
    
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        try:
            time.sleep(delay)
            data = yf.download(batch, period="2y", group_by='ticker', progress=False, threads=True)
            for t in batch:
                try:
                    df = data[t] if len(batch) > 1 else data
                    if df.empty: continue
                    rec = df.tail(252)
                    baselines[t] = {'52H': float(rec['High'].max()), '52L': float(rec['Low'].min()), 'Jan20': np.nan}
                except: pass
        except: pass
    return baselines

def get_live_dashboard_optimized(tickers, baselines, mode="mobile"):
    clean_tickers = clean_ticker_list(tickers)
    rows = []
    start_time = time.time()
    
    # Configure Fetching Parameters based on Mode
    if mode == "mobile":
        chunk_size = 20   # Tiny chunks to prevent rate limits
        delay = 0.5       # Longer delay to be polite
        label = "Fetching (Mobile Safe Mode)..."
    else:
        chunk_size = 100  # Big chunks for speed
        delay = 0.1       # Minimal delay
        label = "Fetching (Desktop Fast Mode)..."

    progress_bar = st.sidebar.progress(0, text=label)
    failed_phase_1 = [] 
    
    # --- PHASE 1 ---
    for i in range(0, len(clean_tickers), chunk_size):
        batch = clean_tickers[i:i+chunk_size]
        progress_bar.progress(min((i / len(clean_tickers)), 1.0))
        try:
            time.sleep(delay)
            data = yf.download(batch, period="5d", group_by='ticker', progress=False, threads=True)
            for t in batch:
                try:
                    loaded = False
                    if len(batch) > 1:
                        if t in data.columns.levels[0]: df = data[t]; loaded = True
                    else: df = data; loaded = True
                    
                    if loaded and not (df.empty or df['Close'].isna().all()):
                        # Only append if valid data is returned
                        res = process_ticker_data(t, df, baselines, t)
                        if res: rows.append(res)
                    else: failed_phase_1.append(t)
                except: failed_phase_1.append(t)
        except: failed_phase_1.extend(batch)
            
    # --- PHASE 2: FALLBACK (Same logic for both) ---
    retry_batch = []
    fallback_map = {} 
    for t in failed_phase_1:
        if '.NS' in t:
            bse_t = t.replace('.NS', '.BO')
            retry_batch.append(bse_t)
            fallback_map[bse_t] = t

    if retry_batch:
        progress_bar.progress(1.0, text="Retrying failed stocks...")
        for i in range(0, len(retry_batch), chunk_size):
            batch = retry_batch[i:i+chunk_size]
            try:
                time.sleep(delay)
                bse_data = yf.download(batch, period="5d", group_by='ticker', progress=False, threads=True)
                for t in batch:
                    try:
                        if len(batch) > 1:
                            if t in bse_data.columns.levels[0]:
                                res = process_ticker_data(t, bse_data[t], baselines, fallback_map.get(t, t))
                                if res: rows.append(res)
                    except: pass
            except: pass

    progress_bar.empty()
    return pd.DataFrame(rows), (time.time() - start_time)

def process_ticker_data(ticker_to_use, df, baselines, original_id):
    try:
        curr = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2])
        stats = baselines.get(original_id, {'52H': curr, '52L': curr, 'Jan20': np.nan})
        h52 = max(stats['52H'], curr)
        l52 = min(stats['52L'], curr)
        
        name, sector = original_id, "Other"
        lookup = original_id
        if lookup in MASTER_MAP:
            name = MASTER_MAP[lookup]['Name']
            sector = MASTER_MAP[lookup]['Sector']
        else:
            base = lookup.replace('.NS','').replace('.BO','')
            for k, v in MASTER_MAP.items():
                if base in k: name = v['Name']; sector = v['Sector']; break

        return {
            "Name": name, "LTP": int(round(curr,0)), "Change %": ((curr-prev)/prev)*100,
            "52W High": int(round(h52,0)), "Down from High (%)": ((h52-curr)/h52)*100 if h52 else 0,
            "52W Low": int(round(l52,0)), "Up from Low (%)": ((curr-l52)/l52)*100 if l52 else 0,
            "Since Jan20 (%)": np.nan, "Sector": sector, "TickerID": ticker_to_use,
            "Get Info?": False, "Market Cap ($M)": None, "PE Ratio": None, "PB Ratio": None, "Div Yield (%)": None, "EPS": None
        }
    except: return None

# ----------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------
st.sidebar.title("Controls")

# --- MODE SELECTOR (The Key Feature) ---
# Defaults to "Mobile" for safety. Change to "Desktop" on PC.
app_mode = st.sidebar.radio("🚀 App Mode:", ["Mobile (Safe)", "Desktop (Fast)"], index=0)
mode_key = "mobile" if "Mobile" in app_mode else "desktop"

if 'market_df' not in st.session_state: st.session_state.market_df = pd.DataFrame()
if 'load_time' not in st.session_state: st.session_state.load_time = 0.0
if 'usd_rate' not in st.session_state: st.session_state.usd_rate = get_usd_rate()
if 'missing_stocks' not in st.session_state: st.session_state.missing_stocks = []

if st.sidebar.button("🔄 Update Prices"):
    st.session_state.market_df = pd.DataFrame()
    st.session_state.missing_stocks = []
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Filters")
search_query = st.sidebar.text_input("🔍 Search Stock/Sector", "")
mobile_view = st.sidebar.checkbox("📱 Simplified Columns", value=(mode_key == "mobile")) # Auto-check if mobile
view_option = st.sidebar.radio("Trend:", ["All", "Green Only", "Red Only"], index=0)
near_high = st.sidebar.checkbox("Near 52W High (<5%)")
near_low = st.sidebar.checkbox("Near 52W Low (<10%)")

# --- LOAD DATA ---
if st.session_state.market_df.empty:
    with st.spinner(f"Loading in {mode_key} mode..."):
        baselines = get_baselines_bulk(DEFAULT_TICKERS, mode=mode_key)
        df, duration = get_live_dashboard_optimized(DEFAULT_TICKERS, baselines, mode=mode_key)
        st.session_state.market_df = df
        st.session_state.load_time = duration
        
        if not df.empty:
            loaded_ids = set()
            for t in df['TickerID'].tolist():
                loaded_ids.add(t); 
                if '.BO' in t: loaded_ids.add(t.replace('.BO', '.NS'))
            cleaned_defaults = clean_ticker_list(DEFAULT_TICKERS)
            missing = [t for t in cleaned_defaults if t not in loaded_ids]
            st.session_state.missing_stocks = missing

df = st.session_state.market_df.copy()

# --- SIDEBAR STATUS ---
if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    st.sidebar.metric("Stocks Updated", f"{len(df)} / {len(DEFAULT_TICKERS)}")
    st.sidebar.metric("Load Time", f"{st.session_state.load_time:.2f}s")
    
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="2d")
        if not hist.empty:
            curr = hist['Close'].iloc[-1]; prev = hist['Close'].iloc[-2]
            st.sidebar.metric("Nifty 50", f"{curr:.0f}", f"{((curr-prev)/prev)*100:+.2f}%")
        else: st.sidebar.metric("Nifty 50", "N/A")
    except: st.sidebar.metric("Nifty 50", "Err")

if st.session_state.missing_stocks:
    with st.sidebar.expander(f"⚠️ Failed ({len(st.session_state.missing_stocks)})", expanded=False):
        st.text("\n".join(st.session_state.missing_stocks))

# --- MAIN PAGE ---
st.title("Market Monitor")

if not df.empty:
    if search_query:
        df = df[df['Name'].str.contains(search_query, case=False, na=False) | df['Sector'].str.contains(search_query, case=False, na=False)]
    if view_option == "Green Only": df = df[df['Change %'] > 0]
    elif view_option == "Red Only": df = df[df['Change %'] < 0]
    if near_high: df = df[df['Down from High (%)'] < 5]
    if near_low: df = df[df['Up from Low (%)'] < 10]

    df = df.sort_values(by="Name", ignore_index=True)
    df.insert(0, "#", range(1, len(df) + 1))

    # --- ROW LIMIT LOGIC ---
    # Mobile Mode: Hard limit to 100 rows to prevent crash
    # Desktop Mode: Show all by default
    default_limit = 100 if mode_key == "mobile" else len(df)
    limit_rows = st.sidebar.slider("Rows to Display", 10, len(df), default_limit)
    df_display = df.head(limit_rows)

    def color_change(val):
        if pd.isna(val): return ''
        color = '#2ecc71' if val > 0 else '#ff4b4b' if val < 0 else ''
        return f'color: {color}'

    styled_df = df_display.style.map(color_change, subset=['Change %', 'Since Jan20 (%)'])

    num_cols = ["LTP", "Change %", "52W High", "Down from High (%)", "52W Low", "Up from Low (%)", "Since Jan20 (%)"]
    fund_cols = ["Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]
    
    col_config = {
        "#": st.column_config.NumberColumn("#", format="%d"),
        "Name": st.column_config.TextColumn("Name", width="large"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Get Info?": st.column_config.CheckboxColumn("Get Info?", width="small"),
        "TickerID": None
    }
    for col in num_cols: col_config[col] = st.column_config.NumberColumn(col, width="small", format="%d" if "High" in col or "Low" in col or "LTP" in col else "%.1f")
    for col in fund_cols: col_config[col] = st.column_config.NumberColumn(col, width="small", format="%.2f")

    active_cols = ["#", "Name", "LTP", "Change %"] if mobile_view else ["#", "Name", "Sector"] + num_cols + ["Get Info?"] + fund_cols
    
    # Render
    edited_df = st.data_editor(
        styled_df, 
        use_container_width=True, hide_index=True, height=(len(df_display)+1)*35+3,
        column_order=active_cols, column_config=col_config,
        disabled=["#", "Name", "Sector"] + num_cols + fund_cols, key="data_editor"
    )

    # Trigger
    changed_rows = edited_df[edited_df.get("Get Info?", pd.Series([False]*len(edited_df))) == True]
    if not changed_rows.empty:
        with st.spinner("Fetching fundamentals..."):
            for index, row in changed_rows.iterrows():
                t = row['TickerID']
                mcap, pe, eps, pb, div, sec = fetch_fundamental_single(t, st.session_state.usd_rate)
                mask = st.session_state.market_df['TickerID'] == t
                st.session_state.market_df.loc[mask, 'Market Cap ($M)'] = mcap
                st.session_state.market_df.loc[mask, 'PE Ratio'] = pe
                st.session_state.market_df.loc[mask, 'PB Ratio'] = pb
                st.session_state.market_df.loc[mask, 'Div Yield (%)'] = div
                st.session_state.market_df.loc[mask, 'EPS'] = eps
                if sec: st.session_state.market_df.loc[mask, 'Sector'] = sec
                st.session_state.market_df.loc[mask, 'Get Info?'] = False
        st.rerun()
else:
    st.warning("No data found (or Rate Limit hit). Wait 60s and try again.")
