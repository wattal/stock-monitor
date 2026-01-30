import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
import os

# --- BLOCK E1: TECHNICAL CALCULATIONS ---

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index."""
    if series is None or len(series) < period:
        return pd.Series([50] * len(series)) if series is not None else 50
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def get_usd_rate():
    """Fetches USDINR exchange rate for market cap conversion."""
    try:
        df = yf.download("USDINR=X", period="5d", progress=False)
        return float(df["Close"].iloc[-1])
    except:
        return 84.5


# --- BLOCK E2: HISTORY DATA ---

@st.cache_data(ttl=600)
def download_bulk_history(tickers, period="1mo"):
    """Downloads history while bypassing known delisted/failing tickers."""
    blacklist = ["^NSEI", "WARDIN.BO", "JAGATJITIND.BO", "WARDFIN.BO", "HBLPOWER.NS", "BMW.NS"]
    cleaned = [t.upper().strip() + (".NS" if not (t.endswith(".NS") or t.endswith(".BO")) else "") 
               for t in tickers if t not in blacklist]
    ticker_list = list(set(cleaned))
    
    all_chunks, chunk_size = [], 25 
    for i in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[i : i + chunk_size]
        try:
            # threads=False ensures stability on shared hosted IPs 
            data = yf.download(chunk, period=period, group_by="ticker", progress=False, threads=False)
            if not data.empty: 
                all_chunks.append(data)
            time.sleep(0.8) 
        except: 
            continue

    if not all_chunks: 
        return pd.DataFrame()
    full_df = pd.concat(all_chunks, axis=1)
    if not full_df.empty: 
        full_df.index = full_df.index.tz_localize(None)
    return full_df


# --- BLOCK E3: LIVE DATA & BASELINES ---

def calculate_baselines(tickers, raw_data, ref_date=None):
    """Calculates high/low metrics and reference date performance."""
    baselines = {}
    cut = pd.to_datetime(ref_date).tz_localize(None) if ref_date else None
    for t in tickers:
        try:
            df = raw_data[t].dropna(subset=["Close"]) if t in raw_data else pd.DataFrame()
            if df.empty: 
                continue
            data_len = len(df)
            baselines[t] = {
                "30H": float(df.iloc[-min(22, data_len):]["High"].max()), 
                "30L": float(df.iloc[-min(22, data_len):]["Low"].min()),
                "15H": float(df.iloc[-min(11, data_len):]["High"].max()), 
                "15L": float(df.iloc[-min(11, data_len):]["Low"].min()),
                "7H": float(df.iloc[-min(5, data_len):]["High"].max()), 
                "7L": float(df.iloc[-min(5, data_len):]["Low"].min()),
                "52H": float(df["High"].max()) if data_len > 200 else np.nan,
                "52L": float(df["Low"].min()) if data_len > 200 else np.nan,
                "MA100": float(df["Close"].iloc[-100:].mean()) if data_len > 100 else np.nan,
                "RSI": calculate_rsi(df["Close"]).iloc[-1] if data_len > 14 else 50,
                "AvgVol": float(df["Volume"].iloc[-min(10, data_len):].mean()), 
                "RefLow": float(df[df.index >= cut]["Low"].min()) if cut and not df[df.index >= cut].empty else np.nan,
            }
        except: 
            continue 
    return baselines

def get_live_data(tickers, baselines, dormant_set):
    """Restored function: Fetches real-time prices for the table."""
    from tickers import MASTER_MAP
    active = [t for t in tickers if t not in dormant_set]
    rows, failed = [], []
    # threads=True is safe for the smaller live fetch 
    data = yf.download(active, period="5d", group_by="ticker", progress=False, threads=True)
    for t in active:
        try:
            df_t = data[t] if len(active) > 1 else data
            if not df_t.empty:
                valid = df_t.dropna(subset=["Close"])
                p = float(valid["Close"].iloc[-1])
                prev = float(valid["Close"].iloc[-2]) if len(valid) > 1 else p
                b = baselines.get(t, {})
                get_pct = lambda val, base: (((val - base) / base) * 100 if base and not pd.isna(base) else np.nan)
                rows.append({
                    "Name": MASTER_MAP[t]["Name"], 
                    "LTP": p, 
                    "Change %": ((p - prev) / prev) * 100,
                    "vs 7D High (%)": get_pct(p, b.get("7H")), 
                    "vs 7D Low (%)": get_pct(p, b.get("7L")),
                    "vs 15D High (%)": get_pct(p, b.get("15H")), 
                    "vs 15D Low (%)": get_pct(p, b.get("15L")),
                    "vs 52W High (%)": get_pct(p, b.get("52H")), 
                    "vs 52W Low (%)": get_pct(p, b.get("52L")),
                    "Up/Low since": get_pct(p, b.get("RefLow")), 
                    "RSI (14)": b.get("RSI", 50),
                    "Vol Breakout": float(valid["Volume"].iloc[-1]) / b.get("AvgVol", 1) if b.get("AvgVol", 1) > 0 else 1.0,
                    "vs 100DMA (%)": get_pct(p, b.get("MA100")), 
                    "Sector": MASTER_MAP[t]["Sector"],
                    "TickerID": t, 
                    "Market Cap ($M)": np.nan, "PE Ratio": np.nan, "PB Ratio": np.nan, "Div Yield (%)": np.nan, "EPS": np.nan,
                })
        except: 
            failed.append(t)
    return pd.DataFrame(rows), 0, list(set(failed))


# --- BLOCK E4: FUNDAMENTALS & SERIALIZATION FIX ---

@st.cache_data(ttl=86400)
def fetch_fundamentals_map(tickers, usd_rate):
    results = {}
    # Use a small batch to test if the connection is working
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            # ... (rest of your existing clean_val logic)
            results[t] = {
                "Market Cap ($M)": round(float(info.get("marketCap", 0) / usd_rate / 1_000_000), 2),
                "PE Ratio": clean_val(info.get("trailingPE")),
                "PB Ratio": clean_val(info.get("priceToBook")),
                "Div Yield (%)": clean_val(info.get("dividendYield", 0) * 100),
                "EPS": clean_val(info.get("trailingEps"))
            }
        except Exception as e:
            continue
    return results


# --- BLOCK E5: PERSISTENCE ---

def load_watchlist():
    """Restored function: Loads your starred tickers."""
    if not os.path.exists("watchlist.txt"): 
        return []
    with open("watchlist.txt", "r") as f: 
        return [line.strip() for line in f.readlines() if line.strip()]

def save_to_watchlist(ticker, add=True):
    """Restored function: Updates your starred tickers."""
    current = set(load_watchlist())
    if add: 
        current.add(ticker)
    else: 
        current.discard(ticker)
    with open("watchlist.txt", "w") as f:
        for t in sorted(current): 
            f.write(f"{t}\n")
