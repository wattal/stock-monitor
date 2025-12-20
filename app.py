import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from tickers import MASTER_MAP

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CSS
# ----------------------------------------------------
st.set_page_config(page_title="Market Monitor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
        /* Fix the top padding so title doesn't hide behind the navbar on mobile */
        .block-container {
            padding-top: 3rem !important; 
            padding-bottom: 2rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        /* Make the first column (index) tiny */
        [data-testid="stDataFrame"] div[role="grid"] div[role="row"] > div:nth-child(1) {
            min-width: 30px !important; max-width: 40px !important; width: 40px !important; flex: 0 0 40px !important;
        }
        [data-testid="stDataFrame"] div[role="grid"] div[role="rowgroup"] > div[role="row"] > div:nth-child(1) {
            min-width: 30px !important; max-width: 40px !important; width: 40px !important; flex: 0 0 40px !important;
        }
        /* Fix Header Text */
        h1 {
            font-size: 1.8rem !important;
            padding-top: 0rem !important;
        }
        /* Force Table Text color if needed - usually not required if Theme is set correctly */
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
    except:
        return 84.0

def clean_ticker_list(tickers):
    FIX_MAP = {
        "M&M": "M&M.NS", "ARE&M": "ARE&M.NS", "M&MFIN": "M&MFIN.NS",
        "MCDOWELL-N": "UNITDSPR.NS", "WAASOLAR": "WAAREERTL.NS",
        "WEBSOL": "WEBELSOLAR.NS", "WELLIVING": "WELSPUNLIV.NS",
        "RATTANINDIA": "RTNINDIA.NS", "GUJFLUORO": "FLUOROCHEM.NS",
        "BERGERPAINT": "BERGEPAINT.NS", "KILBURN": "KILBUNENG.NS",
        "FINOTEX": "FCL.NS", "SOMDIST": "SDBL.NS", "VALOR": "DBREALTY.NS",
        "EQUINOX": "IBREALEST.NS", "WARDWIZARD": "WARDFIN.BO", 
        "BLACKBOX": "BBOX.NS", "GALAXYSURF": "GALXBRG.BO",
        "AUTOLINE": "AUTOIND.NS", "ANDHRACEMT": "ACL.BO", "TRIL": "TARIL.BO"
    }
    cleaned = []
    for t in tickers:
        base = t.replace('.NS', '').replace('.BO', '')
        if base in FIX_MAP: t = FIX_MAP[base]
        elif not (t.endswith('.NS') or t.endswith('.BO')): t += '.NS'
        cleaned.append(t)
    return list(set(cleaned))

def fetch_fundamental_single(ticker, usd_rate):
    try:
        info = yf.Ticker(ticker).info
        mcap = info.get('marketCap', np.nan)
        curr = info.get('currency', 'INR')
        
        if not pd.isna(mcap):
            if curr == 'INR': mcap = (mcap / usd_rate) / 1000000 
            else: mcap = mcap / 1000000
            mcap = round(mcap, 2)
        
        pe = info.get('trailingPE', np.nan)
        eps = info.get('trailingEps', np.nan)
        pb = info.get('priceToBook', np.nan)
        div = info.get('dividendYield', np.nan)
        if not pd.isna(div): div = div * 100 
        
        fetched_sector = info.get('sector', None)
        return mcap, pe, eps, pb, div, fetched_sector
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None

# ----------------------------------------------------
# 3. DATA FETCHING
# ----------------------------------------------------
@st.cache_data(ttl=43200) 
def get_baselines_bulk(tickers):
    tickers = clean_ticker_list(tickers)
    baselines = {}
    if not tickers: return {}
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', progress=False, threads=True)
        valid = [t for t in tickers if t in data.columns.levels[0]]
        for t in valid:
            try:
                df = data[t]
                if df.empty: continue
                rec = df.tail(252)
                baselines[t] = {'52H': float(rec['High'].max()), '52L': float(rec['Low'].min()), 'Jan20': np.nan}
            except: pass
    except: pass
    return baselines

def get_live_dashboard_optimized(tickers, baselines):
    clean_tickers = clean_ticker_list(tickers)
    rows = []
    
    start_time = time.time()
    
    # --- PHASE 1: NSE FETCH ---
    chunk_size = 100
    progress_bar = st.sidebar.progress(0, text="Fetching Stocks (NSE)...")
    failed_phase_1 = [] 
    
    for i in range(0, len(clean_tickers), chunk_size):
        batch = clean_tickers[i:i+chunk_size]
        progress_bar.progress((i/len(clean_tickers)))
        try:
            data = yf.download(batch, period="5d", group_by='ticker', progress=False, threads=True)
            for t in batch:
                loaded = False
                try:
                    if len(batch) > 1:
                        if t in data.columns.levels[0]: 
                            df = data[t]
                            loaded = True
                    else:
                        df = data
                        loaded = True
                    
                    if loaded and not (df.empty or df['Close'].isna().all()):
                        rows.append(process_ticker_data(t, df, baselines, t))
                    else:
                        failed_phase_1.append(t)
                except:
                    failed_phase_1.append(t)
        except:
            failed_phase_1.extend(batch)
            
    # --- PHASE 2: BSE FALLBACK ---
    fallback_map = {} 
    retry_batch = []
    
    for t in failed_phase_1:
        if '.NS' in t:
            bse_t = t.replace('.NS', '.BO')
            retry_batch.append(bse_t)
            fallback_map[bse_t] = t

    if retry_batch:
        progress_bar.progress(1.0, text=f"Retrying {len(retry_batch)} stocks on BSE...")
        try:
            bse_data = yf.download(retry_batch, period="5d", group_by='ticker', progress=False, threads=True)
            for t in retry_batch:
                try:
                    loaded = False
                    if len(retry_batch) > 1:
                        if t in bse_data.columns.levels[0]:
                            df = bse_data[t]
                            loaded = True
                    else:
                        df = bse_data
                        loaded = True
                    
                    if loaded and not (df.empty or df['Close'].isna().all()):
                        original_id = fallback_map.get(t, t)
                        rows.append(process_ticker_data(t, df, baselines, original_id))
                except: pass
        except: pass

    progress_bar.empty()
    end_time = time.time()
    return pd.DataFrame(rows), (end_time - start_time)

def process_ticker_data(ticker_to_use, df, baselines, original_id):
    curr = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    
    stats = baselines.get(original_id, {'52H': curr, '52L': curr, 'Jan20': np.nan})
    h52 = max(stats['52H'], curr)
    l52 = min(stats['52L'], curr)
    
    name, sector = original_id, "Other"
    lookup = original_id
    if lookup not in MASTER_MAP:
        if lookup.replace('.NS', '') in MASTER_MAP: lookup = lookup.replace('.NS', '')
        elif lookup.replace('.BO', '') in MASTER_MAP: lookup = lookup.replace('.BO', '')

    if lookup in MASTER_MAP:
        name = MASTER_MAP[lookup]['Name']
        sector = MASTER_MAP[lookup]['Sector']

    return {
        "Name": name,
        "LTP": int(round(curr,0)),
        "Change %": ((curr-prev)/prev)*100,
        "52W High": int(round(h52,0)),
        "Down from High (%)": ((h52-curr)/h52)*100 if h52 else 0,
        "52W Low": int(round(l52,0)),
        "Up from Low (%)": ((curr-l52)/l52)*100 if l52 else 0,
        "Since Jan20 (%)": np.nan,
        "Sector": sector,
        "TickerID": ticker_to_use,
        "Get Info?": False,
        "Market Cap ($M)": None,
        "PE Ratio": None,
        "PB Ratio": None,
        "Div Yield (%)": None,
        "EPS": None
    }

# ----------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Controls")

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

# 📱 Mobile View Checkbox Added Here
mobile_view = st.sidebar.checkbox("📱 Mobile Simplified View", value=False)

view_option = st.sidebar.radio("Trend:", ["All", "Green Only", "Red Only"], index=0)
near_high = st.sidebar.checkbox("Near 52W High (<5%)")
near_low = st.sidebar.checkbox("Near 52W Low (<10%)")

# --- LOAD DATA ---
if st.session_state.market_df.empty:
    with st.spinner("Loading Market Data..."):
        baselines = get_baselines_bulk(DEFAULT_TICKERS)
        df, duration = get_live_dashboard_optimized(DEFAULT_TICKERS, baselines)
        st.session_state.market_df = df
        st.session_state.load_time = duration
        
        loaded_ids = set()
        for t in df['TickerID'].tolist():
            loaded_ids.add(t) 
            if '.BO' in t: loaded_ids.add(t.replace('.BO', '.NS'))
        cleaned_defaults = clean_ticker_list(DEFAULT_TICKERS)
        missing = [t for t in cleaned_defaults if t not in loaded_ids]
        st.session_state.missing_stocks = missing

df = st.session_state.market_df.copy()

# --- SIDEBAR STATUS ---
if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    total = len(DEFAULT_TICKERS)
    updated = len(df)
    st.sidebar.metric("Stocks Updated", f"{updated} / {total}")
    st.sidebar.metric("Load Time", f"{st.session_state.load_time:.2f}s")
    
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="2d")
        if not hist.empty:
            curr_nifty = hist['Close'].iloc[-1]
            prev_nifty = hist['Close'].iloc[-2]
            nifty_chg = ((curr_nifty - prev_nifty)/prev_nifty)*100
            st.sidebar.metric("Nifty 50", f"{curr_nifty:.0f}", f"{nifty_chg:+.2f}%")
        else:
            indices_df = yf.download("^NSEI", period="2d", progress=False)
            if not indices_df.empty:
                 curr_nifty = indices_df['Close'].iloc[-1]
                 prev_nifty = indices_df['Close'].iloc[-2]
                 nifty_chg = ((curr_nifty - prev_nifty)/prev_nifty)*100
                 st.sidebar.metric("Nifty 50", f"{curr_nifty:.0f}", f"{nifty_chg:+.2f}%")
            else:
                st.sidebar.metric("Nifty 50", "N/A")
    except:
        st.sidebar.metric("Nifty 50", "Err")

if st.session_state.missing_stocks:
    with st.sidebar.expander(f"⚠️ Failed ({len(st.session_state.missing_stocks)})", expanded=False):
        st.text("\n".join(st.session_state.missing_stocks))

# --- MAIN PAGE ---
st.title("Market Monitor")

if not df.empty:
    # Filter Logic
    if search_query:
        df = df[df['Name'].str.contains(search_query, case=False, na=False) | 
                df['Sector'].str.contains(search_query, case=False, na=False)]

    if view_option == "Green Only": df = df[df['Change %'] > 0]
    elif view_option == "Red Only": df = df[df['Change %'] < 0]
    if near_high: df = df[df['Down from High (%)'] < 5]
    if near_low: df = df[df['Up from Low (%)'] < 10]

    df = df.sort_values(by="Name", ignore_index=True)
    df.insert(0, "#", range(1, len(df) + 1))

    # Styling
    def color_change(val):
        if pd.isna(val): return ''
        color = '#2ecc71' if val > 0 else '#ff4b4b' if val < 0 else ''
        return f'color: {color}'

    styled_df = df.style.map(color_change, subset=['Change %', 'Since Jan20 (%)'])

    # Column Config
    num_cols = ["LTP", "Change %", "52W High", "Down from High (%)", "52W Low", "Up from Low (%)", "Since Jan20 (%)"]
    fund_cols = ["Market Cap ($M)", "PE Ratio", "PB Ratio", "Div Yield (%)", "EPS"]
    
    col_config = {
        "#": st.column_config.NumberColumn("#", format="%d"),
        "Name": st.column_config.TextColumn("Name", width="large"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Get Info?": st.column_config.CheckboxColumn("Get Info?", width="small"),
        "TickerID": None
    }
    for col in num_cols:
        fmt = "%d" if "High" in col or "Low" in col or "LTP" in col else "%.1f"
        col_config[col] = st.column_config.NumberColumn(col, width="small", format=fmt)
    for col in fund_cols:
        col_config[col] = st.column_config.NumberColumn(col, width="small", format="%.2f")

    # Dynamic Column Selection Logic
    if mobile_view:
        # Simplified View
        active_cols = ["#", "Name", "LTP", "Change %"]
    else:
        # Full Desktop View
        active_cols = ["#", "Name", "Sector"] + num_cols + ["Get Info?"] + fund_cols

    # Table Height Calc
    table_height = (len(df) + 1) * 35 + 3

    # Render Table
    edited_df = st.data_editor(
        styled_df, 
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_order=active_cols, # <--- Uses the active list logic
        column_config=col_config,
        disabled=["#", "Name", "Sector"] + num_cols + fund_cols,
        key="data_editor"
    )

    # Fundamental Trigger
    changed_rows = edited_df[edited_df["Get Info?"] == True]
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
    st.warning("No data found.")
