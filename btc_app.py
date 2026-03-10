import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import os
import joblib
import math
import json
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --- Timezone helpers (display in Eastern Time; keep internal logic & Sheets in UTC-naive) ---
ET_TZ = ZoneInfo("America/New_York")


def _utc_naive_to_et_ts(dt):
    """Interpret a tz-naive datetime/timestamp as UTC and convert to US/Eastern (tz-aware)."""
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return None
    try:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_convert(ET_TZ)
    except Exception:
        return None


def fmt_et(dt, fmt: str) -> str:
    ts = _utc_naive_to_et_ts(dt)
    if ts is None:
        return ""
    try:
        return ts.strftime(fmt)
    except Exception:
        return ""


def et_naive(dt):
    """UTC-naive -> Eastern-naive (for Plotly display)."""
    ts = _utc_naive_to_et_ts(dt)
    if ts is None:
        return None
    try:
        return ts.tz_localize(None).to_pydatetime()
    except Exception:
        try:
            py = ts.to_pydatetime()
            return py.replace(tzinfo=None)
        except Exception:
            return None


# --- Polymarket helpers ---
def get_polymarket_url():
    """Return the Polymarket market URL from secrets, or a placeholder if not configured."""
    try:
        return st.secrets["polymarket"]["url"]
    except Exception:
        return None


@st.cache_data(ttl=10)
def fetch_polymarket_odds(target_time):
    """Fetch real-time UP/DOWN buy prices from the Polymarket CLOB API.
    Falls back to Gamma API outcomePrices if CLOB is unavailable.
    Returns {"up": float, "down": float, "slug": str, "price_to_beat": float|None, "source": str} or None.
    """
    import json, re
    import requests as _requests
    ts = int(target_time.replace(tzinfo=timezone.utc).timestamp()) - 300  # Polymarket slugs use window open time
    slug = f"btc-updown-5m-{ts}"
    try:
        resp = _requests.get(
            f"https://gamma-api.polymarket.com/events?slug={slug}",
            timeout=5,
        )
        data = resp.json()
        if not data:
            return None
        event = data[0] if isinstance(data, list) else data
        markets = event.get("markets", [])
        if not markets:
            return None

        result = {}
        price_to_beat = None
        gamma_up = None
        gamma_down = None
        clob_token_map = {}  # outcome_key -> token_id

        for market in markets:
            # Parse outcomes and gamma prices (fallback)
            raw_outcomes = market.get("outcomes", "[]")
            raw_prices = market.get("outcomePrices", "[]")
            raw_clob_ids = market.get("clobTokenIds", "[]")

            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            clob_ids = json.loads(raw_clob_ids) if isinstance(raw_clob_ids, str) else raw_clob_ids

            for i, (outcome, price) in enumerate(zip(outcomes, prices)):
                key = str(outcome).strip().lower()
                token_id = clob_ids[i] if i < len(clob_ids) else None
                if "up" in key:
                    gamma_up = float(price)
                    if token_id:
                        clob_token_map["up"] = token_id
                elif "down" in key:
                    gamma_down = float(price)
                    if token_id:
                        clob_token_map["down"] = token_id

            # Extract price_to_beat: direct field first, then regex on question
            if price_to_beat is None:
                for field in ("startPrice", "startBTCPrice", "strikePrice"):
                    raw = market.get(field)
                    if raw is not None:
                        try:
                            price_to_beat = float(raw)
                        except Exception:
                            pass
                        break
            if price_to_beat is None:
                question = market.get("question", "")
                m = re.search(r'(?:above|below)\s+\$?([\d,]+(?:\.\d+)?)', question, re.IGNORECASE)
                if not m:
                    m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', question)
                if m:
                    try:
                        price_to_beat = float(m.group(1).replace(",", ""))
                    except Exception:
                        pass

        if gamma_up is None or gamma_down is None:
            return None

        # Attempt CLOB real-time prices (Buy price = ask = what Polymarket UI shows)
        clob_source = False
        if clob_token_map.get("up") and clob_token_map.get("down"):
            try:
                up_resp = _requests.get(
                    f"https://clob.polymarket.com/price?token_id={clob_token_map['up']}&side=BUY",
                    timeout=3,
                )
                down_resp = _requests.get(
                    f"https://clob.polymarket.com/price?token_id={clob_token_map['down']}&side=BUY",
                    timeout=3,
                )
                clob_up = float(up_resp.json().get("price", 0))
                clob_down = float(down_resp.json().get("price", 0))
                if 0 < clob_up < 1 and 0 < clob_down < 1:
                    result["up"] = clob_up
                    result["down"] = clob_down
                    clob_source = True
            except Exception:
                pass  # fall through to gamma prices

        if not clob_source:
            result["up"] = gamma_up
            result["down"] = gamma_down

        result["slug"] = slug
        result["price_to_beat"] = price_to_beat
        result["source"] = "clob" if clob_source else "gamma"
        return result
    except Exception:
        return None


def fetch_polymarket_resolution(target_time, require_closed: bool = True) -> str | None:
    """
    Query Gamma API for a settled 5-minute BTC window.
    target_time: the window close time (UTC-naive datetime).
    Returns 'UP', 'DOWN', or None (not yet settled / not found).

    Note: Polymarket never sets resolved=True for these markets — settlement is
    signalled purely by outcomePrices reaching 1.0/0.0 once the market is closed.
    """
    try:
        import requests as _r
        import json as _json
        window_open = target_time - timedelta(minutes=5)
        ts = int(window_open.replace(tzinfo=timezone.utc).timestamp())
        slug = f"btc-updown-5m-{ts}"
        r = _r.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        event = data[0] if isinstance(data, list) else data

        # Only attempt to read resolution from closed markets
        if require_closed and not event.get("closed"):
            return None

        markets = event.get("markets", [])
        for mkt in markets:
            raw_outcomes = mkt.get("outcomes", "[]")
            raw_prices = mkt.get("outcomePrices", "[]")
            outcomes = _json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
            prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            for outcome, price in zip(outcomes, prices):
                try:
                    if float(price) >= 0.99:
                        return str(outcome).strip().upper()  # "UP" or "DOWN"
                except (ValueError, TypeError):
                    continue
        return None
    except Exception:
        return None


def snap_to_polymarket_window(dt):
    """Snap a UTC-naive datetime forward to the next :00/:05/:10... boundary.
    If dt is already on a boundary, returns the NEXT boundary (next window close semantics).
    """
    # If exactly on a 5-min boundary, advance to the next one
    if dt.second == 0 and dt.microsecond == 0 and dt.minute % 5 == 0:
        return dt + timedelta(minutes=5)
    # Otherwise snap forward to the next 5-min boundary
    minutes_to_add = 5 - (dt.minute % 5)
    return dt.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)


def is_at_polymarket_boundary(dt):
    """Return True if dt falls exactly on a :00/:05/:10... boundary."""
    return dt.second == 0 and dt.minute % 5 == 0


# --- 1. Page Setup & Live Sync ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin AI Trading Terminal")

st.sidebar.markdown("### ⚙️ Terminal Settings")
auto_pilot = st.sidebar.toggle("Enable Live Sync (5m interval)")
prediction_mode = st.sidebar.radio("Prediction Mode", ["Manual", "Auto"], index=0)

if auto_pilot:
    # Full-app rerun every 5 minutes (kept for your existing "Live Sync" behavior)
    st_autorefresh(interval=300000, key="data_refresh")
    st.sidebar.success("Live Sync Active: Fetching latest data every 5 minutes.")
# Refresh every 10s so CLOB odds stay near-real-time regardless of auto_pilot
st_autorefresh(interval=10000, key="odds_refresh")
if prediction_mode == "Auto" and not auto_pilot:
    st.sidebar.info("Auto mode requires Live Sync to be enabled.")


# --- 2. Exchange (cached) ---
@st.cache_resource
def get_exchange():
    """Create a single CCXT exchange client (shared across reruns)."""
    return ccxt.kraken({"enableRateLimit": True})


# --- 3. Load the Brain ---
@st.cache_resource
def load_model(mtime=None):   # mtime as cache key forces reload when file changes on disk
    return joblib.load("btc_5m_rf_model.joblib")


try:
    _model_mtime = os.path.getmtime("btc_5m_rf_model.joblib")
    model = load_model(_model_mtime)
except Exception as e:
    st.error(f"Could not load the model. Error: {e}")
    st.stop()


# --- 4. Data Fetchers (KRAKEN) ---
def get_live_prediction_data():
    exchange = get_exchange()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1m", limit=100)

    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)

    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"], window_slow=26, window_fast=12)
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["EMA_9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA_21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
    df["Volume_ROC"] = df["Volume"].pct_change(periods=5)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def get_24h_chart_data():
    exchange = get_exchange()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", limit=288)

    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df


PYTH_BTC_URL = (
    "https://hermes.pyth.network/v2/updates/price/latest"
    "?ids[]=0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
)


def get_polymarket_btc_price() -> float | None:
    """Fetch live BTC/USD from Pyth (Polymarket's oracle). Returns None on failure."""
    try:
        import requests as _r
        r = _r.get(PYTH_BTC_URL, timeout=5)
        r.raise_for_status()
        parsed = r.json()["parsed"][0]["price"]
        price = float(parsed["price"]) * 10 ** int(parsed["expo"])
        return round(price, 2)
    except Exception:
        return None


PYTH_BTC_FEED_ID = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"

def fetch_pyth_price_at(target_time) -> float | None:
    """
    Fetch the Pyth oracle BTC/USD price at a specific historical UTC timestamp.
    Uses Hermes v2 publish_time endpoint — returns the price update at or just
    before target_time, which is the same source Polymarket uses to settle.
    target_time: UTC-naive datetime.
    """
    try:
        import requests as _r
        ts = int(target_time.replace(tzinfo=timezone.utc).timestamp())
        url = (
            f"https://hermes.pyth.network/v2/updates/price/{ts}"
            f"?ids[]={PYTH_BTC_FEED_ID}"
        )
        r = _r.get(url, timeout=8)
        r.raise_for_status()
        parsed = r.json()["parsed"][0]["price"]
        price = float(parsed["price"]) * 10 ** int(parsed["expo"])
        return round(price, 2)
    except Exception:
        return None


def get_live_ticker_price():
    """Fetch live BTC/USD — Pyth oracle first (Polymarket source), Kraken fallback."""
    price = get_polymarket_btc_price()
    if price:
        return price
    try:
        exchange = get_exchange()
        ticker = exchange.fetch_ticker("BTC/USDT")
        return ticker.get("last")
    except Exception:
        return None


# --- 5. Google Sheets Connection & Grader ---
def get_gspread_client():
    creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(ttl=120)
def _fetch_sheet_records():
    """Cached sheet read — 2-minute TTL to reduce Google Sheets API quota usage.
    Uses get_all_values() instead of get_all_records() to tolerate blank/duplicate header cells
    that arise when new columns are appended without updating the header row first.
    """
    client = get_gspread_client()
    sheet = client.open("BTC_AI_Tracker").sheet1
    rows = sheet.get_all_values()
    if not rows:
        return []
    headers = rows[0]
    # Trim trailing empty headers; named columns always precede blanks
    while headers and headers[-1] == "":
        headers = headers[:-1]
    records = []
    for row in rows[1:]:
        row = list(row)
        # Pad short rows, trim extra columns beyond known headers
        row = row + [""] * max(0, len(headers) - len(row))
        row = row[: len(headers)]
        records.append(dict(zip(headers, row)))
    return records


_EXPECTED_HEADERS = [
    "Prediction_Time", "Entry_Price", "Window_Start_Price", "Prediction",
    "Confidence", "Target_Time", "Close_Price", "Outcome",
    "Polymarket_Odds", "PM_Resolution", "Seconds_Left", "Model", "Model_Version",
]

def _ensure_sheet_headers(sheet):
    """Write any missing column headers to row 1 without touching data rows."""
    current = sheet.row_values(1)
    updates = []
    for i, expected in enumerate(_EXPECTED_HEADERS, start=1):
        if i > len(current) or current[i - 1].strip() == "":
            updates.append((1, i, expected))
    for r, c, v in updates:
        sheet.update_cell(r, c, v)


def _get_cached_sheet():
    """Open the worksheet once per session and reuse — avoids repeated open() API reads."""
    if "_sheet_obj" not in st.session_state:
        client = get_gspread_client()
        st.session_state["_sheet_obj"] = client.open("BTC_AI_Tracker").sheet1
    return st.session_state["_sheet_obj"]


@st.cache_resource
def get_autotrader_sheet():
    """Returns the AutoTrader worksheet, creating it if needed."""
    try:
        ws_main = get_gspread_client().open("BTC_AI_Tracker")
        try:
            ws = ws_main.worksheet("AutoTrader")
        except gspread.WorksheetNotFound:
            ws = ws_main.add_worksheet("AutoTrader", rows=1000, cols=10)
            ws.append_row(["Trade_Time", "Direction", "Confidence", "Price",
                           "BTC_Change", "Cash_Change", "BTC_Balance", "Cash_Balance",
                           "Portfolio_Value", "Model_Used"])
        return ws
    except Exception:
        return None


def load_history_from_sheets():
    try:
        sheet = _get_cached_sheet()
        if not st.session_state.get("_headers_ensured"):
            _ensure_sheet_headers(sheet)
            st.session_state["_headers_ensured"] = True
        records = _fetch_sheet_records()

        if not records:
            return pd.DataFrame(), sheet

        df = pd.DataFrame(records)
        df["Prediction_Time"] = pd.to_datetime(df["Prediction_Time"], errors='coerce')
        df["Target_Time"] = pd.to_datetime(df["Target_Time"], errors='coerce')
        # Silently discard any rows with unparseable timestamps (corrupted sheet cells)
        if df["Prediction_Time"].isna().any():
            df.dropna(subset=["Prediction_Time"], inplace=True)

        # Coerce all numeric columns — get_all_values() returns everything as strings
        for _col in ["Entry_Price", "Window_Start_Price", "Close_Price",
                     "Confidence", "Polymarket_Odds", "Seconds_Left"]:
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")
            else:
                df[_col] = np.nan

        # String columns — ensure they exist with a blank default
        for _col in ["Model", "PM_Resolution", "Model_Version"]:
            if _col not in df.columns:
                df[_col] = ""

        result = (df, sheet)
        st.session_state["_last_sheet_data"] = result
        return result
    except gspread.exceptions.APIError as e:
        if "429" in str(e):
            last = st.session_state.get("_last_sheet_data")
            if last is not None:
                if not st.session_state.get("_429_warned"):
                    st.warning("⚠️ Google Sheets rate limit reached — showing last cached data.")
                    st.session_state["_429_warned"] = True
                return last
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None


def resolve_pending_trades_in_sheets(history_df, sheet, live_data=None):
    """Resolve pending trades.  live_data (Kraken OHLCV) is optional — only needed
    as a last resort when both the Gamma API and Pyth historical endpoint fail."""
    if history_df.empty or sheet is None:
        return history_df

    pending_mask = history_df["Outcome"] == "Pending"

    for idx, row in history_df[pending_mask].iterrows():
        target_time = row["Target_Time"]
        if pd.isna(target_time):
            continue

        # Only attempt resolution once the window has closed (add 30s grace for Polymarket to settle)
        if target_time > datetime.utcnow() - timedelta(seconds=30):
            continue

        prediction = row["Prediction"]
        outcome = None
        close_price = None

        pm_resolution = None

        # --- Primary: Polymarket Gamma API (authoritative resolution) ---
        pm_resolved = fetch_polymarket_resolution(target_time)
        if pm_resolved in ("UP", "DOWN"):
            pm_resolution = pm_resolved
            outcome = "Win" if pm_resolved == prediction else "Loss"
            pyth_price = fetch_pyth_price_at(target_time)
            close_price = pyth_price if pyth_price else None

        # --- Fallback: Pyth price comparison (Polymarket not yet resolved) ---
        if outcome is None:
            # Both prices from Pyth so the comparison uses the same oracle
            close_price = fetch_pyth_price_at(target_time)
            if close_price is None and live_data is not None:
                # Last resort: Kraken candle at or after target
                valid_candles = live_data[live_data.index >= target_time]
                if not valid_candles.empty:
                    close_price = float(valid_candles["Close"].iloc[0])

            if close_price is None:
                continue   # can't resolve yet — leave Pending

            # NaN-safe reference price: prefer stored window_start_price, then fetch
            # it fresh from Pyth at window open, then fall back to entry price
            def _first_valid_price(*cols):
                for c in cols:
                    v = row.get(c)
                    try:
                        f = float(v)
                        if pd.notna(f) and f > 0:
                            return f
                    except (TypeError, ValueError):
                        pass
                return None

            ref_price = _first_valid_price("Window_Start_Price", "Entry_Price")
            if ref_price is None:
                # Row predates Window_Start_Price column — fetch from Pyth now
                ref_price = fetch_pyth_price_at(target_time - timedelta(minutes=5))

            if ref_price is None:
                continue

            outcome = (
                "Win"
                if (prediction == "UP" and close_price > ref_price)
                or (prediction == "DOWN" and close_price < ref_price)
                else "Loss"
            )

        history_df.at[idx, "Close_Price"] = round(close_price, 2) if close_price else ""
        history_df.at[idx, "Outcome"] = outcome
        if pm_resolution:
            history_df.at[idx, "PM_Resolution"] = pm_resolution
        sheet.update_cell(idx + 2, 7, round(close_price, 2) if close_price else "")  # col G = Close_Price
        sheet.update_cell(idx + 2, 8, outcome)                                         # col H = Outcome
        if pm_resolution:
            sheet.update_cell(idx + 2, 10, pm_resolution)                              # col J = PM_Resolution
        _fetch_sheet_records.clear()
        st.session_state.pop("_sheet_obj", None)
        st.session_state.pop("_headers_ensured", None)

    return history_df


# --- 6. Live Market Fragment (2s auto-refresh, no full app rerun) ---
# Streamlit renamed experimental_fragment -> fragment. Support both.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

if _fragment is None:
    # Fallback: app will still run, but the live ticker won't auto-refresh without a full rerun.
    def _fragment(*args, **kwargs):  # noqa: D401
        def _wrap(func):
            return func

        return _wrap


@_fragment()
def live_market_and_advanced_stats_fragment(
    overall_wr,
    p90_wr,
    p90_threshold,
    up_wr,
    down_wr,
    avg_conf_win,
    avg_conf_loss,
    never_lost_above=None,
):
    """Only this section reruns every 2 seconds while the session is active."""
    st.markdown("#### 📊 Advanced Stats")

    col_stats1, col_stats2 = st.columns(2)

    with col_stats1:
        st.markdown("**Core Win Rates:**")
        st.write(f"- Overall Win Rate: **{overall_wr:.1f}%**")

        if p90_wr is not None and p90_threshold is not None:
            st.write(f"- Top 10% Conf Win Rate: **{p90_wr:.1f}%** *(>={p90_threshold:.1f}% Conf)*")
        else:
            st.write("- Top 10% Conf Win Rate: **N/A**")

        if never_lost_above is not None:
            _nla_thresh, _nla_n, _nla_perfect = never_lost_above
            if _nla_perfect:
                st.write(f"- 🏆 **Never Lost:** Perfect record across all {_nla_n} trade{'s' if _nla_n != 1 else ''}!")
            else:
                st.write(
                    f"- 🛡️ **Never Lost Above {_nla_thresh:.0f}% Conf** "
                    f"*({_nla_n} win{'s' if _nla_n != 1 else ''}, 0 losses above that level)*"
                )

    with col_stats2:
        st.markdown("**Direction & Conviction:**")
        if up_wr is not None and down_wr is not None:
            st.write(f"- Long (UP) Win Rate: **{up_wr:.1f}%**")
            st.write(f"- Short (DOWN) Win Rate: **{down_wr:.1f}%**")
        else:
            st.write("- Long (UP) Win Rate: **N/A**")
            st.write("- Short (DOWN) Win Rate: **N/A**")

        if avg_conf_win is not None and avg_conf_loss is not None:
            st.caption(f"Avg Conf on Wins: {avg_conf_win:.1f}% | On Losses: {avg_conf_loss:.1f}%")

    st.divider()


@st.cache_data(ttl=300)
def _load_model_metadata():
    """Load model_metadata.json — cached for 5 min so all tabs share the same value."""
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.json")
        with open(_p) as _f:
            return json.load(_f)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def _load_training_stats():
    """Read BTCUSDT_1m_processed.csv and return per-horizon row counts and date ranges."""
    try:
        _csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BTCUSDT_1m_processed.csv")
        df = pd.read_csv(_csv_path)
        ts_col = "Timestamp" if "Timestamp" in df.columns else next(
            (c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None
        )
        result = {}
        for h in range(1, 6):
            sub = df[df["minutes_to_window_end"] == h] if "minutes_to_window_end" in df.columns else df
            earliest_str = latest_str = None
            if ts_col and len(sub) > 0:
                parsed = pd.to_datetime(sub[ts_col], errors="coerce").dropna()
                if len(parsed) > 0:
                    earliest_str = str(parsed.min())
                    latest_str = str(parsed.max())
            result[h] = {"rows": int(len(sub)), "earliest": earliest_str, "latest": latest_str}
        return result
    except Exception as e:
        return {"_error": str(e)}


# --- 7. UI Layout (Tabs) ---
# Reset per-cycle warning flag so the 429 warning shows once per render cycle, not 4×
st.session_state.pop("_429_warned", None)
if "at_enabled" not in st.session_state:
    st.session_state.at_enabled = False
if "at_btc" not in st.session_state:
    st.session_state.at_btc = None        # BTC held (qty), seeded on first price fetch
if "at_cash" not in st.session_state:
    st.session_state.at_cash = 500.0      # USD cash
if "at_btc_seeded" not in st.session_state:
    st.session_state.at_btc_seeded = False
if "at_trade_log" not in st.session_state:
    st.session_state.at_trade_log = []

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Live Predictor",
    "📊 Analytics & 24h Visualizer",
    "💰 P&L Simulator",
    "📈 Odds vs Performance",
    "🧠 Model Health",
    "🤖 Auto Trade (Beta)",
])

with tab1:
    st.markdown("### Generate Next Move")

    # Session state for prediction tracking
    if "last_auto_target" not in st.session_state:
        st.session_state.last_auto_target = None
    if "last_pred_candle_ts" not in st.session_state:
        st.session_state.last_pred_candle_ts = None   # candle timestamp of last logged prediction
    if "last_pred_odds_up" not in st.session_state:
        st.session_state.last_pred_odds_up = None     # UP odds used in last logged prediction
    if "last_pred_target" not in st.session_state:
        st.session_state.last_pred_target = None      # window the last prediction was for
    if "last_pred_wall_time" not in st.session_state:
        st.session_state.last_pred_wall_time = None   # wall-clock time of last logged prediction

    now_utc = datetime.utcnow().replace(microsecond=0)
    _current_window = snap_to_polymarket_window(now_utc)

    # Reset per-window data tracking when a new window starts
    if st.session_state.last_pred_target != _current_window:
        st.session_state.last_pred_candle_ts = None
        st.session_state.last_pred_odds_up = None

    # First trigger: fires once at each 5-minute window boundary
    _first_trigger = (
        prediction_mode == "Auto"
        and auto_pilot
        and is_at_polymarket_boundary(now_utc)
        and st.session_state.last_auto_target != _current_window
    )

    # Refresh trigger: same window, at most once per ~60s (1-minute candle cadence)
    _same_window = (st.session_state.last_pred_target == _current_window)
    _last_wall = st.session_state.last_pred_wall_time
    _enough_time = (_last_wall is None) or ((datetime.utcnow() - _last_wall).total_seconds() >= 58)
    _refresh_trigger = (
        prediction_mode == "Auto"
        and auto_pilot
        and _same_window
        and _enough_time
        and not _first_trigger
    )

    auto_trigger = _first_trigger or _refresh_trigger

    if _first_trigger:
        st.info("Auto-prediction firing at Polymarket window boundary...")

    if st.button("Generate Live Prediction", type="primary") or auto_trigger:
        with st.spinner("Fetching live data & consulting AI..."):
            live_data = get_live_prediction_data()
            history_df, sheet = load_history_from_sheets()

            history_df = resolve_pending_trades_in_sheets(history_df, sheet, live_data=live_data)

            current_state = live_data.iloc[-1:]
            current_price = float(current_state["Close"].values[0])
            current_time = current_state.index[0]  # UTC-naive candle timestamp
            _live_price = get_live_ticker_price()
            entry_price = _live_price if _live_price else current_price

            # Use wall-clock time for target_time — candle timestamp lags by up to 1 candle
            # at the boundary (last candle = 14:45 when we're actually at 14:50).
            target_time = snap_to_polymarket_window(datetime.utcnow())

            # Window start = 5 minutes before target_time
            window_start_time = target_time - timedelta(minutes=5)

            # Pyth oracle price at window open (same source Polymarket uses)
            window_start_price = fetch_pyth_price_at(window_start_time)
            if window_start_price is None:
                # Fall back to Kraken candle at window open, then current price
                window_start_candles = live_data[live_data.index == window_start_time]
                window_start_price = (
                    float(window_start_candles["Close"].values[0])
                    if not window_start_candles.empty
                    else current_price
                )

            # Fetch Polymarket odds for this window (non-blocking — fails gracefully)
            fetch_polymarket_odds.clear()          # bypass cache — always fetch fresh at prediction time
            pm_odds = fetch_polymarket_odds(target_time)

            # Check whether data has meaningfully changed since the last prediction this window
            _is_manual = not auto_trigger
            _candle_changed = (current_time != st.session_state.last_pred_candle_ts)
            _odds_changed = (
                pm_odds is not None
                and st.session_state.last_pred_odds_up is not None
                and abs(pm_odds["up"] - st.session_state.last_pred_odds_up) >= 0.01
            )
            _data_changed = _candle_changed or _odds_changed

            # For auto refresh: silently skip if data hasn't changed
            if _refresh_trigger and not _data_changed and not _is_manual:
                st.caption("📊 Auto-checked: data unchanged since last prediction.")
                st.session_state.last_pred_wall_time = datetime.utcnow()
            else:
                if _refresh_trigger and _data_changed:
                    _reasons = []
                    if _candle_changed:
                        _reasons.append("new candle")
                    if _odds_changed:
                        _reasons.append(
                            f"odds Δ {abs(pm_odds['up'] - st.session_state.last_pred_odds_up)*100:.1f}%"
                        )
                    st.info(f"Auto-prediction: data refreshed ({', '.join(_reasons)})")

                # Seconds-precise model selection — derived from wall-clock seconds left,
                # NOT from current_time.minute (candle timestamp can lag by 1–3 min).
                _now = datetime.utcnow()
                _seconds_left = max(0, int((target_time - _now).total_seconds()))

                if _seconds_left < 90:          # < 1.5 min → force 1-min model
                    _minutes_to_end = 1
                else:
                    _minutes_to_end = max(1, min(5, round(_seconds_left / 60)))

                horizon_model = model[_minutes_to_end]

                # Build the 9-feature input for the selected model
                _price_chg = (
                    (current_price - window_start_price) / window_start_price
                    if window_start_price and window_start_price != 0
                    else 0.0
                )
                _FEATURE_COLS = [
                    'RSI_14', 'MACD', 'MACD_Signal', 'EMA_9', 'EMA_21',
                    'BB_Upper', 'BB_Lower', 'Volume_ROC',
                    'price_change_since_window_start', 'price_change_abs'
                ]
                current_state = current_state.copy()
                current_state["price_change_since_window_start"] = _price_chg
                current_state["price_change_abs"] = abs(_price_chg)
                current_state_pred = current_state[_FEATURE_COLS]
                prediction_val = horizon_model.predict(current_state_pred)[0]
                probabilities = horizon_model.predict_proba(current_state_pred)[0]

                direction = "UP" if prediction_val == 1 else "DOWN"
                confidence = probabilities[1] if prediction_val == 1 else probabilities[0]
                confidence_pct = confidence * 100

                if confidence_pct < 55:
                    signal_strength = "⚠️ WEAK SIGNAL (Coin Flip - Do not trade)"
                    color = "orange"
                elif confidence_pct < 60:
                    signal_strength = "✅ MODERATE SIGNAL (Standard Edge)"
                    color = "blue"
                else:
                    signal_strength = "🔥 STRONG SIGNAL (High Probability)"
                    color = "green"

                st.markdown(f"### Target Window: {fmt_et(target_time, '%H:%M %Z')}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Current BTC Price", f"${current_price:,.2f}")
                col2.metric("AI Prediction", f"{direction}", delta="Long" if direction == "UP" else "-Short")
                col3.metric("AI Confidence", f"{confidence_pct:.1f}%")

                st.markdown(f"**Signal Strength:** :{color}[{signal_strength}] &nbsp; `{_minutes_to_end}-min model`")

                # Window countdown
                _mins, _secs = divmod(_seconds_left, 60)
                _countdown_color = "green" if _seconds_left > 180 else "orange" if _seconds_left > 60 else "red"
                st.markdown(f"**⏱️ Window closes in:** :{_countdown_color}[{_mins}m {_secs:02d}s]")

                _window_secs_left = _seconds_left   # already computed above

                if auto_trigger and pm_odds is None and _window_secs_left > 45:
                    st.warning(
                        "⏳ Polymarket odds not yet available — retrying on next refresh. "
                        f"Window closes in {_window_secs_left}s."
                    )
                    # Don't set last_auto_target → allows retry on next rerun
                elif _window_secs_left < 60:
                    st.error(
                        "⛔ Window closes in under 1 minute — prediction blocked. "
                        "Wait for the next window to open."
                    )
                    st.session_state.last_auto_target = target_time   # prevent auto-retry spam
                else:
                    if sheet:
                        odds_val = round(pm_odds[direction.lower()], 4) if pm_odds else ""
                        _pred_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        _append_meta = _load_model_metadata()
                        _append_ver = str(_append_meta.get("model_version", "")) if _append_meta.get("model_version") else ""
                        new_row = [
                            _pred_timestamp,             # Col 1:  Prediction_Time (second-precision UTC)
                            entry_price,                 # Col 2:  Entry_Price
                            window_start_price,          # Col 3:  Window_Start_Price
                            direction,                   # Col 4:  Prediction
                            round(confidence_pct, 2),    # Col 5:  Confidence
                            str(target_time),            # Col 6:  Target_Time
                            "",                          # Col 7:  Close_Price (resolver)
                            "Pending",                   # Col 8:  Outcome (resolver)
                            odds_val,                    # Col 9:  Polymarket_Odds
                            "",                          # Col 10: PM_Resolution (filled by resolver)
                            _seconds_left,               # Col 11: Seconds_Left
                            f"{_minutes_to_end}min",     # Col 12: Model
                            _append_ver,                 # Col 13: Model_Version
                        ]
                        sheet.append_row(new_row)
                        _fetch_sheet_records.clear()   # invalidate read cache so other tabs see new row
                        st.session_state.pop("_sheet_obj", None)
                        st.session_state.pop("_headers_ensured", None)
                        st.success("✅ Prediction successfully logged to Google Sheets!")

                    st.session_state.last_auto_target = target_time
                    st.session_state.last_pred_target = target_time
                    st.session_state.last_pred_candle_ts = current_time
                    st.session_state.last_pred_wall_time = datetime.utcnow()
                    if pm_odds:
                        st.session_state.last_pred_odds_up = pm_odds["up"]

    st.divider()
    # ── Live Market Context ───────────────────────────────────────────────────
    st.markdown("#### ⚡ Live Market Context")
    _t1_lc_col, _t1_odds_col = st.columns([1, 1.5])

    with _t1_lc_col:
        # ── Live ticker ───────────────────────────────────────────────────────
        if "last_live_price" not in st.session_state:
            st.session_state.last_live_price = None
            st.session_state.last_live_price_ts = None
        _t1_live_price_val = get_live_ticker_price()
        if _t1_live_price_val is not None:
            st.session_state.last_live_price = float(_t1_live_price_val)
            st.session_state.last_live_price_ts = datetime.utcnow()
        else:
            _t1_live_price_val = st.session_state.last_live_price
        if _t1_live_price_val is not None:
            st.metric("Live BTC/USDT Price", f"${float(_t1_live_price_val):,.2f}")
            _t1_tick_ts = st.session_state.get("last_live_price_ts")
            if _t1_tick_ts:
                st.caption(f"Last tick: {fmt_et(_t1_tick_ts, '%H:%M:%S %Z')} • Source: Kraken (CCXT)")
            else:
                st.caption("Source: Kraken (CCXT)")
        else:
            st.warning("Could not fetch live price (Kraken).")

        st.markdown("---")
        # ── Last AI Prediction ────────────────────────────────────────────────
        st.markdown("**Last AI Prediction**")
        _t1_last_history, _ = load_history_from_sheets()
        if not _t1_last_history.empty:
            _t1_last = _t1_last_history.iloc[-1]
            _t1_dir_icon = "⬆️" if _t1_last.get("Prediction") == "UP" else "⬇️"
            st.metric("Direction", f"{_t1_dir_icon} {_t1_last.get('Prediction', 'N/A')}")
            st.metric("AI Confidence", f"{_t1_last.get('Confidence', 'N/A')}%")
            _t1_target_display = fmt_et(_t1_last.get("Target_Time"), "%H:%M %Z") or str(_t1_last.get("Target_Time", "N/A"))
            st.metric("Target Window", _t1_target_display)
            _t1_outcome = _t1_last.get("Outcome", "Pending")
            _t1_outcome_color = "green" if _t1_outcome == "Win" else "red" if _t1_outcome == "Loss" else "orange"
            st.markdown(f"**Outcome:** :{_t1_outcome_color}[{_t1_outcome}]")
            _t1_last_odds = _t1_last.get("Polymarket_Odds")
            if pd.notna(_t1_last_odds) and float(_t1_last_odds) > 0:
                st.metric("Odds at Prediction", f"{float(_t1_last_odds)*100:.1f}%",
                          delta=f"Payout {1/float(_t1_last_odds):.2f}x")
        else:
            st.info("No predictions logged yet.")

    with _t1_odds_col:
        st.markdown("**Live Odds — Current Window**")
        if st.button("🔄 Refresh Odds", key="t1_refresh_polymarket_odds"):
            fetch_polymarket_odds.clear()
            st.rerun()
        _t1_current_window = snap_to_polymarket_window(datetime.utcnow().replace(microsecond=0))

        with st.expander("🔍 Odds fetch diagnostics", expanded=False):
            _t1_diag_ts = int(_t1_current_window.replace(tzinfo=timezone.utc).timestamp()) - 300
            _t1_diag_slug = f"btc-updown-5m-{_t1_diag_ts}"
            _t1_diag_url = f"https://gamma-api.polymarket.com/events?slug={_t1_diag_slug}"
            st.write(f"**Window target (UTC):** `{_t1_current_window}`")
            st.write(f"**Slug tried:** `{_t1_diag_slug}`")
            st.write(f"**API URL:** `{_t1_diag_url}`")
            try:
                import requests as _req
                _t1_raw = _req.get(_t1_diag_url, timeout=5).json()
                st.json(_t1_raw)
            except Exception as _t1_e:
                st.error(f"Request failed: {_t1_e}")

        _t1_live_odds = fetch_polymarket_odds(_t1_current_window)

        if _t1_live_odds:
            _t1_odds_col_up, _t1_odds_col_down = st.columns(2)
            _t1_odds_col_up.metric(
                "UP probability",
                f"{_t1_live_odds['up'] * 100:.1f}%",
                delta=f"Implied payout {1 / _t1_live_odds['up']:.2f}x",
            )
            _t1_odds_col_down.metric(
                "DOWN probability",
                f"{_t1_live_odds['down'] * 100:.1f}%",
                delta=f"Implied payout {1 / _t1_live_odds['down']:.2f}x",
            )
            _t1_src_label = "🟢 CLOB (real-time)" if _t1_live_odds.get("source") == "clob" else "🟡 Gamma (cached)"
            st.caption(
                f"Market: `{_t1_live_odds['slug']}` · Source: {_t1_src_label} · Window closes at "
                f"{fmt_et(_t1_current_window, '%H:%M %Z')} · Auto-refreshes every 10s"
            )
        else:
            st.warning(
                "Could not fetch live odds from Polymarket. "
                "The market for this window may not be open yet, or the API is temporarily unavailable."
            )

        _t1_pm_url = get_polymarket_url()
        if _t1_pm_url and not _t1_pm_url.startswith("https://polymarket.com/event/PLACEHOLDER"):
            st.link_button("Open on Polymarket ↗", _t1_pm_url)

        # Accumulate odds readings for trend chart
        if "pm_odds_history" not in st.session_state:
            st.session_state.pm_odds_history = []
        if "pm_odds_window" not in st.session_state:
            st.session_state.pm_odds_window = None
        if st.session_state.pm_odds_window != _t1_current_window:
            st.session_state.pm_odds_history = []
            st.session_state.pm_odds_window = _t1_current_window
        if _t1_live_odds:
            st.session_state.pm_odds_history.append({
                "ts": datetime.utcnow(),
                "up": _t1_live_odds["up"] * 100,
                "down": _t1_live_odds["down"] * 100,
            })

        if len(st.session_state.pm_odds_history) >= 2:
            import plotly.graph_objects as go
            _t1_hist = st.session_state.pm_odds_history
            _t1_times = [h["ts"] for h in _t1_hist]
            _t1_ups = [h["up"] for h in _t1_hist]
            _t1_downs = [h["down"] for h in _t1_hist]
            _t1_fig = go.Figure()
            _t1_fig.add_trace(go.Scatter(x=_t1_times, y=_t1_ups, mode="lines+markers",
                                         name="UP %", line=dict(color="lime", width=2)))
            _t1_fig.add_trace(go.Scatter(x=_t1_times, y=_t1_downs, mode="lines+markers",
                                         name="DOWN %", line=dict(color="tomato", width=2)))
            _t1_fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            _t1_chart_title = ""
            if _t1_live_odds and _t1_live_odds.get("price_to_beat"):
                _t1_chart_title = f"Strike price: ${_t1_live_odds['price_to_beat']:,.2f}"
            _t1_fig.update_layout(
                title=_t1_chart_title,
                yaxis=dict(title="Probability %", range=[0, 100]),
                xaxis_title="Time (UTC)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(_t1_fig, use_container_width=True)
        elif _t1_live_odds:
            st.caption("Collecting readings… chart appears after the second tick (~10s).")
        else:
            st.caption("No odds data available for chart.")

        if _t1_live_odds and _t1_live_odds.get("price_to_beat"):
            st.metric(
                "🎯 Polymarket Strike Price",
                f"${_t1_live_odds['price_to_beat']:,.2f}",
                help="BTC price to beat — parsed from the Polymarket market question for this window.",
            )

    # ── Tick-by-tick BTC price chart (full-width, resets each window) ─────────
    st.markdown("##### 📈 BTC Price — Current Window")

    if "btc_tick_history" not in st.session_state:
        st.session_state.btc_tick_history = []
    if "btc_tick_window" not in st.session_state:
        st.session_state.btc_tick_window = None

    # Reset history when the window rolls over
    if st.session_state.btc_tick_window != _t1_current_window:
        st.session_state.btc_tick_history = []
        st.session_state.btc_tick_window = _t1_current_window

    # Append current tick (de-duplicate: only add if price or timestamp changed)
    if _t1_live_price_val is not None:
        _last_tick = st.session_state.btc_tick_history[-1] if st.session_state.btc_tick_history else None
        if _last_tick is None or _last_tick["price"] != float(_t1_live_price_val):
            st.session_state.btc_tick_history.append({
                "ts": datetime.utcnow(),
                "price": float(_t1_live_price_val),
            })

    # Determine the strike / reference price for the horizontal line:
    # Prefer Polymarket's price_to_beat (exact oracle strike); fall back to first tick (window open proxy).
    _btc_strike = None
    if _t1_live_odds and _t1_live_odds.get("price_to_beat"):
        _btc_strike = float(_t1_live_odds["price_to_beat"])
    elif st.session_state.btc_tick_history:
        _btc_strike = st.session_state.btc_tick_history[0]["price"]

    # Load open bets for the current window (Pending outcome, matching target time)
    _t1_all_history, _ = load_history_from_sheets()
    _win_start_dt = _t1_current_window - timedelta(minutes=5)
    _open_bets = pd.DataFrame()
    if not _t1_all_history.empty and "Outcome" in _t1_all_history.columns:
        _ob_mask = _t1_all_history["Outcome"] == "Pending"
        if "Target_Time" in _t1_all_history.columns:
            # Match bets whose target window is the current window (within 60s tolerance)
            def _target_matches_window(tt):
                try:
                    tt_dt = pd.to_datetime(tt)
                    if tt_dt.tzinfo is not None:
                        tt_dt = tt_dt.tz_localize(None)
                    return abs((tt_dt - _t1_current_window).total_seconds()) < 60
                except Exception:
                    return False
            _ob_mask = _ob_mask & _t1_all_history["Target_Time"].apply(_target_matches_window)
        _open_bets = _t1_all_history[_ob_mask].copy()

    if st.session_state.btc_tick_history:
        import plotly.graph_objects as _go_btc
        _btc_hist = st.session_state.btc_tick_history
        _btc_times = [h["ts"] for h in _btc_hist]
        _btc_prices = [h["price"] for h in _btc_hist]

        # Tight y-axis: include strike + bet entry prices in range
        _btc_all_vals = _btc_prices + ([_btc_strike] if _btc_strike else [])
        if not _open_bets.empty and "Entry_Price" in _open_bets.columns:
            _btc_all_vals += [float(p) for p in _open_bets["Entry_Price"].dropna()]
        _btc_lo = min(_btc_all_vals)
        _btc_hi = max(_btc_all_vals)
        _btc_pad = max((_btc_hi - _btc_lo) * 0.35, 15)

        # Colour: green if current price is above strike, red if below
        _btc_above = (_btc_strike is None) or (_btc_prices[-1] >= _btc_strike)
        _line_col = "rgba(0,255,120,0.9)" if _btc_above else "rgba(255,80,80,0.9)"
        _fill_col = "rgba(0,255,120,0.07)" if _btc_above else "rgba(255,80,80,0.07)"

        _fig_btc = _go_btc.Figure()
        _fig_btc.add_trace(_go_btc.Scatter(
            x=_btc_times,
            y=_btc_prices,
            mode="lines",
            name="BTC/USD",
            line=dict(color=_line_col, width=2),
            fill="tozeroy",
            fillcolor=_fill_col,
        ))

        if _btc_strike:
            _fig_btc.add_hline(
                y=_btc_strike,
                line_dash="dash",
                line_color="gold",
                line_width=1.5,
                annotation_text=f"Strike  ${_btc_strike:,.2f}",
                annotation_position="top right",
                annotation_font_color="gold",
                annotation_font_size=12,
            )

        # ── Overlay open bet entry points ────────────────────────────────────
        if not _open_bets.empty:
            for _, _ob_row in _open_bets.iterrows():
                try:
                    _ob_pred_time = pd.to_datetime(_ob_row.get("Prediction_Time"))
                    if _ob_pred_time.tzinfo is not None:
                        _ob_pred_time = _ob_pred_time.tz_localize(None)
                    _ob_entry = float(_ob_row.get("Entry_Price", 0) or 0)
                    _ob_ref = float(_ob_row.get("Window_Start_Price") or _ob_entry or 0)
                    _ob_dir = _ob_row.get("Prediction", "UP")
                    _ob_conf = _ob_row.get("Confidence", "")
                    _ob_odds = _ob_row.get("Polymarket_Odds", "")

                    # Current win/loss based on live price vs reference
                    if _t1_live_price_val and _ob_ref:
                        _ob_winning = (
                            float(_t1_live_price_val) > _ob_ref if _ob_dir == "UP"
                            else float(_t1_live_price_val) < _ob_ref
                        )
                    else:
                        _ob_winning = None

                    _ob_marker_color = (
                        "lime" if _ob_winning is True
                        else "tomato" if _ob_winning is False
                        else "gold"
                    )
                    _ob_symbol = "triangle-up" if _ob_dir == "UP" else "triangle-down"

                    _ob_hover = (
                        f"{_ob_dir} @ ${_ob_entry:,.2f}<br>"
                        f"Conf: {_ob_conf}%  Odds: {float(_ob_odds)*100:.1f}%"
                        if _ob_odds else f"{_ob_dir} @ ${_ob_entry:,.2f}<br>Conf: {_ob_conf}%"
                    )

                    _fig_btc.add_trace(_go_btc.Scatter(
                        x=[_ob_pred_time],
                        y=[_ob_entry],
                        mode="markers",
                        marker=dict(
                            symbol=_ob_symbol,
                            size=16,
                            color=_ob_marker_color,
                            line=dict(width=1.5, color="white"),
                        ),
                        name=f"{_ob_dir} bet",
                        hovertext=_ob_hover,
                        hoverinfo="text",
                        showlegend=False,
                    ))
                except Exception:
                    pass

        _fig_btc.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            xaxis=dict(
                range=[_win_start_dt, _t1_current_window],
                gridcolor="rgba(255,255,255,0.07)",
                tickformat="%H:%M:%S",
            ),
            yaxis=dict(
                range=[_btc_lo - _btc_pad, _btc_hi + _btc_pad],
                gridcolor="rgba(255,255,255,0.07)",
                tickprefix="$",
                tickformat=",.0f",
            ),
        )
        st.plotly_chart(_fig_btc, use_container_width=True)

        _btc_n = len(_btc_hist)
        _btc_elapsed = int((datetime.utcnow() - _win_start_dt).total_seconds())
        _btc_remaining = max(0, 300 - _btc_elapsed)
        _btc_rm, _btc_rs = divmod(_btc_remaining, 60)
        st.caption(f"{_btc_n} tick{'s' if _btc_n != 1 else ''} this window · {_btc_rm}m {_btc_rs:02d}s remaining · refreshes every 10s")
    else:
        st.caption("Collecting price ticks… chart appears after the first reading.")

    # ── Current Open Bets ────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎰 Current Open Bets")

    if _open_bets.empty:
        st.info("No open bets for the current window.")
    else:
        for _, _ob_row in _open_bets.iterrows():
            try:
                _ob_dir = _ob_row.get("Prediction", "?")
                _ob_dir_icon = "⬆️" if _ob_dir == "UP" else "⬇️"
                _ob_conf = _ob_row.get("Confidence", "")
                _ob_entry = float(_ob_row.get("Entry_Price", 0) or 0)
                _ob_ref = float(_ob_row.get("Window_Start_Price") or _ob_entry or 0)
                _ob_target = fmt_et(_ob_row.get("Target_Time"), "%H:%M %Z") or "N/A"
                _ob_pred_t = fmt_et(_ob_row.get("Prediction_Time"), "%H:%M:%S %Z") or "N/A"
                _ob_model = _ob_row.get("Model", "")

                # Odds at time of prediction
                _ob_odds_raw = _ob_row.get("Polymarket_Odds")
                try:
                    _ob_odds_f = float(_ob_odds_raw)
                    _ob_odds_str = f"{_ob_odds_f*100:.1f}%  (payout {1/_ob_odds_f:.2f}x)"
                except (TypeError, ValueError, ZeroDivisionError):
                    _ob_odds_str = "—"

                # Live P&L vs reference price
                if _t1_live_price_val and _ob_ref:
                    _ob_live = float(_t1_live_price_val)
                    _ob_diff = _ob_live - _ob_ref if _ob_dir == "UP" else _ob_ref - _ob_live
                    _ob_diff_pct = (_ob_diff / _ob_ref) * 100
                    _ob_winning = _ob_diff > 0
                    _ob_status_icon = "🟢" if _ob_winning else "🔴"
                    _ob_pnl_str = f"{_ob_status_icon} ${abs(_ob_diff):,.2f} ({_ob_diff_pct:+.3f}%)"
                else:
                    _ob_pnl_str = "—"
                    _ob_winning = None

                # Payout string
                try:
                    _ob_payout_str = f"{1/float(_ob_odds_raw):.2f}x"
                except (TypeError, ValueError, ZeroDivisionError):
                    _ob_payout_str = "—"

                with st.container(border=True):
                    _bc1, _bc2, _bc3, _bc4, _bc5, _bc6 = st.columns([1.2, 1.3, 1.4, 1.3, 1.3, 2])
                    _bc1.metric("Direction", f"{_ob_dir_icon} {_ob_dir}")
                    _bc2.metric("Confidence", f"{_ob_conf}%" if _ob_conf else "—")
                    _bc3.metric("Entry Price", f"${_ob_entry:,.2f}")
                    _bc4.metric("Odds at Bet", _ob_odds_str.split("  ")[0] if "  " in _ob_odds_str else _ob_odds_str)
                    _bc5.metric("Payout", _ob_payout_str)
                    _bc6.metric("Live P&L", _ob_pnl_str)
                    st.caption(
                        f"⏰ Placed: {_ob_pred_t} · Closes: {_ob_target}"
                        + (f" · Model: {_ob_model}" if _ob_model else "")
                    )
            except Exception:
                pass

with tab2:
    st.markdown("### Model Performance Analytics")

    history, _sheet2 = load_history_from_sheets()

    # Auto-resolve on every page render — Gamma API + Pyth only, no Kraken fetch needed
    _pending_count = int((history["Outcome"] == "Pending").sum()) if not history.empty else 0
    if _pending_count > 0:
        history = resolve_pending_trades_in_sheets(history, _sheet2)

    _btn_col1, _btn_col2, _btn_col3 = st.columns(3)

    with _btn_col1:
        if st.button("🔄 Resolve Pending Trades (+ Kraken fallback)", help="Force-resolve using Kraken candles as last resort if Gamma/Pyth unavailable"):
            with st.spinner("Resolving pending trades…"):
                _live2 = get_live_prediction_data()
                history = resolve_pending_trades_in_sheets(history, _sheet2, live_data=_live2)
            st.success("Done.")
            st.rerun()

    with _btn_col2:
        if st.button("🔁 Backfill PM_Resolution", help="Fetch Polymarket UP/DOWN outcome for all rows that are missing it — including already-resolved Win/Loss rows"):
            with st.spinner("Backfilling PM_Resolution from Gamma API…"):
                _filled = 0
                _attempted = 0
                _missing_mask = (
                    history["PM_Resolution"].isna() | (history["PM_Resolution"].astype(str).str.strip() == "")
                ) if "PM_Resolution" in history.columns else pd.Series(True, index=history.index)

                for idx, row in history[_missing_mask].iterrows():
                    target_time = row.get("Target_Time")
                    if pd.isna(target_time) or target_time > datetime.utcnow():
                        continue
                    _attempted += 1
                    pm_res = fetch_polymarket_resolution(target_time, require_closed=False)
                    if pm_res in ("UP", "DOWN"):
                        history.at[idx, "PM_Resolution"] = pm_res
                        if _sheet2:
                            _sheet2.update_cell(idx + 2, 10, pm_res)
                        _filled += 1
                _fetch_sheet_records.clear()
                st.session_state.pop("_sheet_obj", None)
                st.session_state.pop("_headers_ensured", None)
            st.success(f"Backfilled PM_Resolution for {_filled} of {_attempted} row(s) attempted.")
            st.rerun()

    with _btn_col3:
        if st.button("🏷️ Backfill Model Versions", help="Stamp v0/v1/v2/… on all rows missing a Model_Version, based on when each model was retrained."):
            with st.spinner("Backfilling Model_Version…"):
                # Build version timeline from model_history.json
                _ver_timeline = []  # list of (version_int, cutoff_datetime)
                try:
                    _hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_history.json")
                    with open(_hist_path) as _hf:
                        _hist_data = json.load(_hf)
                    for _he in sorted(_hist_data, key=lambda x: x.get("retrained_at_utc", "")):
                        if _he.get("model_version") and _he.get("retrained_at_utc"):
                            _ver_timeline.append((
                                int(_he["model_version"]),
                                pd.to_datetime(_he["retrained_at_utc"]),
                            ))
                except Exception:
                    _ver_timeline = []

                def _version_for_time(pred_time):
                    """Return the version string (v0, v1, …) for a given prediction timestamp."""
                    if not _ver_timeline:
                        return ""
                    ver = 0  # pre-history default
                    for _v, _cutoff in _ver_timeline:
                        if pred_time >= _cutoff:
                            ver = _v
                    return str(ver)

                _bf_filled = 0
                _missing_ver = (
                    history["Model_Version"].isna() | (history["Model_Version"].astype(str).str.strip() == "")
                ) if "Model_Version" in history.columns else pd.Series(True, index=history.index)

                # Collect all updates first, then write in one batch API call
                _batch_updates = []
                for idx, row in history[_missing_ver].iterrows():
                    pred_time = row.get("Prediction_Time")
                    if pd.isna(pred_time):
                        continue
                    _ver_str = _version_for_time(pd.to_datetime(pred_time))
                    if _ver_str:
                        _batch_updates.append((idx, _ver_str))
                        history.at[idx, "Model_Version"] = _ver_str
                        _bf_filled += 1

                if _batch_updates and _sheet2:
                    _sheet2.batch_update([
                        {"range": f"M{idx + 2}", "values": [[_ver_str]]}
                        for idx, _ver_str in _batch_updates
                    ])

                _fetch_sheet_records.clear()
                st.session_state.pop("_sheet_obj", None)
                st.session_state.pop("_headers_ensured", None)
            st.success(f"Stamped Model_Version on {_bf_filled} row(s).")
            st.rerun()

    _tab2_meta = _load_model_metadata()
    _current_model_ver = str(_tab2_meta.get("model_version", "")) if _tab2_meta.get("model_version") else ""

    _exclude_prev_model = st.toggle(
        "Exclude previous model rows from stats & log",
        value=True,
        help="Hides rows generated by older model versions. Rows with no version tag (logged before versioning) are also excluded. Does not delete data from the sheet.",
    )
    if _exclude_prev_model and not history.empty and _current_model_ver:
        history = history[history["Model_Version"].astype(str) == _current_model_ver]

    _available_models = (
        ["All"] + sorted(history["Model"].dropna().unique().tolist())
        if "Model" in history.columns and not history.empty
        else ["All"]
    )
    _model_filter = st.selectbox(
        "Filter by model horizon",
        options=_available_models,
        index=0,
        help="Restrict all stats and charts to a specific model horizon (e.g. '3min'). "
             "Select 'All' to see combined stats.",
    )
    if _model_filter != "All" and "Model" in history.columns:
        history = history[history["Model"] == _model_filter]

    if not history.empty:
        completed_trades = history[history["Outcome"].isin(["Win", "Loss"])]
        total_completed = len(completed_trades)

        # --- Thermal Streaks ---
        def calculate_streak(history_df, hours):
            cutoff = datetime.utcnow() - timedelta(hours=hours)  # UTC-naive cutoff (matches sheet + live_data)
            window = history_df[
                (history_df["Prediction_Time"] >= cutoff) & (history_df["Outcome"].isin(["Win", "Loss"]))
            ]
            total = len(window)
            if total == 0:
                return "Not Enough Data", "gray"
            wins = len(window[window["Outcome"] == "Win"])
            win_rate = (wins / total) * 100

            if total < 3:
                return f"{win_rate:.0f}% ({wins}/{total}) - Neutral", "gray"
            elif win_rate >= 60:
                return f"{win_rate:.0f}% ({wins}/{total}) - 🔥 HOT", "green"
            elif win_rate <= 45:
                return f"{win_rate:.0f}% ({wins}/{total}) - ❄️ COLD", "red"
            else:
                return f"{win_rate:.0f}% ({wins}/{total}) - ⚖️ CHOP", "orange"

        st.markdown("#### Model Temperature (Rolling Windows)")
        s1, s2, s3 = st.columns(3)
        rate_1h, color_1h = calculate_streak(history, 1)
        rate_12h, color_12h = calculate_streak(history, 12)
        rate_24h, color_24h = calculate_streak(history, 24)

        s1.markdown(f"**Last 1 Hour:** :{color_1h}[{rate_1h}]")
        s2.markdown(f"**Last 12 Hours:** :{color_12h}[{rate_12h}]")
        s3.markdown(f"**Last 24 Hours:** :{color_24h}[{rate_24h}]")
        st.divider()

        # --- The 24h Visualizer ---
        st.markdown("#### 24-Hour Trade Overlay (Times shown in Eastern Time)")
        with st.spinner("Rendering 24-hour chart..."):
            chart_data = get_24h_chart_data()

            # Convert chart x-axis to Eastern time for display (interpret index as UTC-naive).
            chart_data_et = chart_data.copy()
            chart_data_et.index = pd.to_datetime(chart_data_et.index).tz_localize("UTC").tz_convert(ET_TZ).tz_localize(None)

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=chart_data_et.index,
                        open=chart_data_et["Open"],
                        high=chart_data_et["High"],
                        low=chart_data_et["Low"],
                        close=chart_data_et["Close"],
                        name="BTC/USDT",
                    )
                ]
            )

            # Filter in UTC (original data), then convert points to ET for plotting.
            recent_history = history[history["Prediction_Time"] >= chart_data.index[0]]

            for _, row in recent_history.iterrows():
                symbol = "triangle-up" if row["Prediction"] == "UP" else "triangle-down"
                color = "green" if row["Outcome"] == "Win" else "red" if row["Outcome"] == "Loss" else "yellow"
                size = 14 if row["Outcome"] != "Pending" else 10

                x_pt = et_naive(row["Prediction_Time"])
                if x_pt is None:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=[x_pt],
                        y=[row["Entry_Price"]],
                        mode="markers",
                        marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color="white")),
                        name=f"{row['Outcome']} ({row['Prediction']})",
                        hoverinfo="text",
                        hovertext=f"{row['Prediction']} | Conf: {row['Confidence']}% | {row['Outcome']}",
                        showlegend=False,
                    )
                )

            fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- Stats panel filters (independent of the global model filter above) ---
        _scol1, _scol2, _scol3 = st.columns(3)
        _stats_time = _scol1.radio(
            "Stats window", ["All Time", "Past 12h", "Past 1h"],
            horizontal=True, key="stats_time_filter",
        )
        _stats_model = _scol2.selectbox(
            "Stats model", options=_available_models, key="stats_model_filter",
        )
        _stats_odds_bucket_opts = ["All", ">50%", "<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"]
        _stats_odds_bucket = _scol3.selectbox(
            "Odds bucket", options=_stats_odds_bucket_opts, key="stats_odds_bucket_filter",
            help="Filter stats to trades in this odds range. '>50%' = market agrees (aligned); '<50%' = contrarian bets only.",
        )

        # Build filtered slice for the stats display
        _stats_trades = completed_trades.copy()
        if _stats_time == "Past 12h":
            _stats_cutoff = datetime.utcnow() - timedelta(hours=12)
            _stats_trades = _stats_trades[_stats_trades["Prediction_Time"] >= _stats_cutoff]
        elif _stats_time == "Past 1h":
            _stats_cutoff = datetime.utcnow() - timedelta(hours=1)
            _stats_trades = _stats_trades[_stats_trades["Prediction_Time"] >= _stats_cutoff]
        if _stats_model != "All" and "Model" in _stats_trades.columns:
            _stats_trades = _stats_trades[_stats_trades["Model"] == _stats_model]
        if _stats_odds_bucket != "All" and "Polymarket_Odds" in _stats_trades.columns:
            _stats_bucket_map = {
                ">50%": (0.50, 1.01),
                "<50%": (0.0, 0.50),
                "50–60%": (0.50, 0.60), "60–70%": (0.60, 0.70),
                "70–80%": (0.70, 0.80), "80–90%": (0.80, 0.90), "90%+": (0.90, 1.01),
            }
            _sb_lo, _sb_hi = _stats_bucket_map[_stats_odds_bucket]
            _stats_trades = _stats_trades[
                (_stats_trades["Polymarket_Odds"] >= _sb_lo) &
                (_stats_trades["Polymarket_Odds"] < _sb_hi)
            ]
        _stats_total = len(_stats_trades)

        # --- Precompute stats for the Advanced Stats fragment ---
        wins = len(_stats_trades[_stats_trades["Outcome"] == "Win"])
        overall_wr = (wins / _stats_total * 100) if _stats_total > 0 else 0.0

        p90_threshold = None
        p90_wr = None
        if _stats_total > 0 and "Confidence" in _stats_trades.columns:
            try:
                p90_threshold = float(_stats_trades["Confidence"].quantile(0.90))
                p90_trades = _stats_trades[_stats_trades["Confidence"] >= p90_threshold]
                if len(p90_trades) > 0:
                    p90_wins = len(p90_trades[p90_trades["Outcome"] == "Win"])
                    p90_wr = (p90_wins / len(p90_trades) * 100)
            except Exception:
                p90_threshold, p90_wr = None, None

        up_wr = None
        down_wr = None
        avg_conf_win = None
        avg_conf_loss = None
        if _stats_total > 0:
            up_trades = _stats_trades[_stats_trades["Prediction"] == "UP"]
            down_trades = _stats_trades[_stats_trades["Prediction"] == "DOWN"]

            up_wr = (
                (len(up_trades[up_trades["Outcome"] == "Win"]) / len(up_trades) * 100) if len(up_trades) > 0 else 0.0
            )
            down_wr = (
                (len(down_trades[down_trades["Outcome"] == "Win"]) / len(down_trades) * 100)
                if len(down_trades) > 0
                else 0.0
            )

            if "Confidence" in _stats_trades.columns:
                avg_conf_win = _stats_trades[_stats_trades["Outcome"] == "Win"]["Confidence"].mean()
                avg_conf_loss = _stats_trades[_stats_trades["Outcome"] == "Loss"]["Confidence"].mean()
                avg_conf_win = float(avg_conf_win) if pd.notna(avg_conf_win) else 0.0
                avg_conf_loss = float(avg_conf_loss) if pd.notna(avg_conf_loss) else 0.0

        # "Never lost above X%" — highest confidence at which a loss was ever recorded,
        # so all trades above that level are wins.
        never_lost_above = None   # (threshold_pct, n_wins_above) or None
        if "Confidence" in _stats_trades.columns and _stats_total > 0:
            _nla_losses = _stats_trades[_stats_trades["Outcome"] == "Loss"]
            _nla_wins   = _stats_trades[_stats_trades["Outcome"] == "Win"]
            if _nla_losses.empty and not _nla_wins.empty:
                # Perfect record — use the minimum winning confidence as the threshold
                never_lost_above = (float(_nla_wins["Confidence"].min()), len(_nla_wins), True)
            elif not _nla_losses.empty:
                _max_loss_conf = float(_nla_losses["Confidence"].max())
                _wins_above = _nla_wins[_nla_wins["Confidence"] > _max_loss_conf]
                if len(_wins_above) >= 1:
                    never_lost_above = (_max_loss_conf, len(_wins_above), False)

        # --- Advanced Stats ---
        live_market_and_advanced_stats_fragment(
            overall_wr=overall_wr,
            p90_wr=p90_wr,
            p90_threshold=p90_threshold,
            up_wr=up_wr,
            down_wr=down_wr,
            avg_conf_win=avg_conf_win,
            avg_conf_loss=avg_conf_loss,
            never_lost_above=never_lost_above,
        )

        # --- Per-Model Breakdown ---
        if "Model" in history.columns and history["Model"].notna().any():
            st.markdown("#### Per-Model Breakdown")
            _model_rows = []
            for _m in ["1min", "2min", "3min", "4min", "5min"]:
                _m_trades = completed_trades[completed_trades["Model"] == _m]
                _m_total = len(_m_trades)
                if _m_total == 0:
                    continue
                _m_wins = len(_m_trades[_m_trades["Outcome"] == "Win"])
                _m_wr = _m_wins / _m_total * 100
                _m_avg_conf = _m_trades["Confidence"].mean() if "Confidence" in _m_trades.columns else None
                _m_avg_odds = (
                    _m_trades["Polymarket_Odds"].mean() * 100
                    if "Polymarket_Odds" in _m_trades.columns and _m_trades["Polymarket_Odds"].notna().any()
                    else None
                )
                _model_rows.append({
                    "Model": _m,
                    "Trades": _m_total,
                    "Wins": _m_wins,
                    "Win Rate": f"{_m_wr:.1f}%",
                    "Avg Confidence": f"{_m_avg_conf:.1f}%" if _m_avg_conf else "—",
                    "Avg Odds": f"{_m_avg_odds:.1f}%" if _m_avg_odds is not None else "—",
                })
            if _model_rows:
                st.dataframe(pd.DataFrame(_model_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No model data yet — Model column added to new predictions going forward.")
            st.divider()

        # --- The Data Table ---
        st.markdown("#### Cloud Tracker Log (Times shown in Eastern Time)")

        def highlight_outcome(val):
            if val == "Win":
                return "background-color: rgba(0, 255, 0, 0.15)"
            elif val == "Loss":
                return "background-color: rgba(255, 0, 0, 0.15)"
            return ""

        # For display, render time columns as strings in ET (avoids tz-aware dtype issues in Streamlit tables).
        history_display = history.copy()
        if "Prediction_Time" in history_display.columns:
            history_display["Prediction_Time"] = history_display["Prediction_Time"].apply(lambda x: fmt_et(x, "%Y-%m-%d %H:%M %Z"))
        if "Target_Time" in history_display.columns:
            history_display["Target_Time"] = history_display["Target_Time"].apply(lambda x: fmt_et(x, "%Y-%m-%d %H:%M %Z"))
        if "Polymarket_Odds" in history_display.columns:
            history_display["Polymarket_Odds"] = history_display["Polymarket_Odds"].apply(
                lambda x: f"{x:.4f} ({x*100:.1f}%)" if pd.notna(x) and x > 0 else ""
            )

        st.dataframe(
            history_display.iloc[::-1].style.map(highlight_outcome, subset=["Outcome"]),
            use_container_width=True,
        )

    else:
        st.info("No predictions found in the Google Sheet yet. Run a prediction to start tracking!")

def _quick_pl_sim(trades_df, apply_skip_rules=True, apply_bet_scaling=True):
    """Lightweight P&L simulation used by the auto-optimizer.
    Returns (final_balance, n_trades_executed). Mirrors the main tab4 logic exactly.
    """
    STARTING_BALANCE = 1000.0
    MIN_BET_PCT = 0.025
    MAX_BET_PCT = 0.05
    HIGH_ODDS_THRESHOLD = 0.65
    MIN_CONF_FOR_HIGH_ODDS = 60.0
    CONTRARIAN_SKIP_THRESHOLD = 0.30
    balance = STARTING_BALANCE
    n_sim = 0
    for _, row in trades_df.iterrows():
        try:
            conf = float(row["Confidence"])
            odds = float(row["Polymarket_Odds"])
            outcome = row["Outcome"]
        except (ValueError, TypeError):
            continue
        if apply_skip_rules:
            if odds < CONTRARIAN_SKIP_THRESHOLD:
                continue
            if odds >= HIGH_ODDS_THRESHOLD and conf < MIN_CONF_FOR_HIGH_ODDS:
                continue
        if odds < 0.5 and apply_bet_scaling:
            bet_pct = min(MAX_BET_PCT, MIN_BET_PCT + (conf - 50) / 100 * MAX_BET_PCT)
        else:
            bet_pct = MIN_BET_PCT
        bet_amount = balance * bet_pct
        n_sim += 1
        if outcome == "Win":
            balance += bet_amount * (1.0 / odds - 1.0)
        else:
            balance -= bet_amount
    return balance, n_sim


with tab3:
    # Inject optimal filter values BEFORE widgets are created to avoid the
    # "widget key already bound" StreamlitAPIException when Apply is clicked.
    if "_pl_apply_pending" in st.session_state:
        _pa = st.session_state.pop("_pl_apply_pending")
        st.session_state["pl_time"] = _pa["time"]
        st.session_state["pl_dir"] = _pa["dir"]
        st.session_state["pl_model"] = _pa["model"]
        st.session_state["pl_min_conf"] = _pa["min_conf"]
        if "skip_rules" in _pa:
            st.session_state["pl_skip_rules"] = _pa["skip_rules"]
        if "bet_scaling" in _pa:
            st.session_state["pl_bet_scaling"] = _pa["bet_scaling"]
        # Map optimizer-only buckets (e.g. "≥60%") to Custom with numeric bounds
        _pa_bucket = _pa["bucket"]
        _pa_range = _pa.get("bucket_range")
        _pl_bucket_opts_check = ["All", ">50%", "<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+", "Custom"]
        if _pa_bucket in _pl_bucket_opts_check:
            st.session_state["pl_odds_bucket"] = _pa_bucket
        else:
            st.session_state["pl_odds_bucket"] = "Custom"
            if _pa_range:
                st.session_state["pl_cust_lo_pct"] = int(_pa_range[0] * 100)
                st.session_state["pl_cust_hi_pct"] = min(100, int(_pa_range[1] * 100))

    st.markdown("### 💰 P&L Simulator")
    st.markdown(
        "Simulates portfolio growth starting from **$1,000** using your logged predictions, "
        "Polymarket odds at bet time, and actual outcomes. "
        "Bet sizing scales with confidence and market odds — contrarian bets (market <50%) scale 2.5–5%, "
        "high-odds markets (≥65%) require ≥60% confidence, standard zone is flat 2.5%."
    )

    # Load history (may already be loaded in tab2 context, but tabs are independent blocks)
    sim_history, _ = load_history_from_sheets()

    _pl_meta = _load_model_metadata()
    _pl_current_ver = str(_pl_meta.get("model_version", "")) if _pl_meta.get("model_version") else ""
    _pl_exclude_prev = st.toggle(
        "Exclude previous model rows from stats & log",
        value=True,
        key="pl_exclude_prev_model",
        help="Hides rows generated by older model versions. Toggle OFF to include all historical predictions.",
    )
    if _pl_exclude_prev and not sim_history.empty and _pl_current_ver and "Model_Version" in sim_history.columns:
        sim_history = sim_history[sim_history["Model_Version"].astype(str) == _pl_current_ver]

    if "pl_skip_rules" not in st.session_state:
        st.session_state["pl_skip_rules"] = True
    if "pl_bet_scaling" not in st.session_state:
        st.session_state["pl_bet_scaling"] = True
    _apply_skip_rules = st.toggle(
        "Apply skip rules (high-odds / low-confidence filter)",
        key="pl_skip_rules",
        help="When ON, bets where market odds ≥65% but AI confidence <60% are skipped. "
             "Toggle OFF to see total P&L as if every prediction had been bet at 2.5%.",
    )
    _apply_bet_scaling = st.toggle(
        "Apply bet scaling (contrarian bets scale 2.5–5% by confidence)",
        key="pl_bet_scaling",
        help="When ON, contrarian bets (market odds <50%) scale from 2.5% to 5% based on AI confidence. "
             "Toggle OFF to apply flat 2.5% to every bet for clean P&L comparison.",
    )
    _pl_row1a, _pl_row1b, _pl_row1c = st.columns(3)
    with _pl_row1a:
        _pl_time_opts = ["All Time", "Past 24h", "Past 12h", "Past 1h"]
        _pl_time_filter = st.selectbox(
            "Time window",
            options=_pl_time_opts,
            index=_pl_time_opts.index(st.session_state.get("pl_time", "All Time")),
            key="pl_time",
            help="Restrict the simulation to trades within this lookback window.",
        )
    with _pl_row1b:
        _pl_bucket_opts = ["All", ">50%", "<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+", "Custom"]
        _pl_odds_bucket = st.selectbox(
            "Odds bucket",
            options=_pl_bucket_opts,
            index=_pl_bucket_opts.index(st.session_state.get("pl_odds_bucket", "All")),
            key="pl_odds_bucket",
            help="Only include trades where Polymarket odds fell in this range. '>50%' filters out contrarian bets (market disagrees with model). 'Custom' lets you enter exact bounds.",
        )
        if _pl_odds_bucket == "Custom":
            _cust_col_lo, _cust_col_hi = st.columns(2)
            with _cust_col_lo:
                _cust_lo = st.number_input("Lower %", min_value=0, max_value=99, value=int(st.session_state.get("pl_cust_lo_pct", 60)), step=1, key="pl_cust_lo_pct") / 100
            with _cust_col_hi:
                _cust_hi = st.number_input("Upper %", min_value=1, max_value=100, value=int(st.session_state.get("pl_cust_hi_pct", 100)), step=1, key="pl_cust_hi_pct") / 100
        else:
            _cust_lo, _cust_hi = None, None
    with _pl_row1c:
        _pl_dir_opts = ["All", "UP", "DOWN"]
        _pl_dir_filter = st.selectbox(
            "Direction",
            options=_pl_dir_opts,
            index=_pl_dir_opts.index(st.session_state.get("pl_dir", "All")),
            key="pl_dir",
            help="Filter to UP (Long) or DOWN (Short) predictions only.",
        )

    _pl_col1, _pl_col2 = st.columns(2)
    with _pl_col1:
        _pl_available_models = (
            ["All"] + sorted(sim_history["Model"].dropna().unique().tolist())
            if "Model" in sim_history.columns and not sim_history.empty
            else ["All"]
        )
        _pl_model_default_idx = (
            _pl_available_models.index(st.session_state.get("pl_model", "All"))
            if st.session_state.get("pl_model", "All") in _pl_available_models
            else 0
        )
        _pl_model_filter = st.selectbox(
            "Filter by model horizon",
            options=_pl_available_models,
            index=_pl_model_default_idx,
            key="pl_model",
            help="Only include trades that used a specific model horizon. "
                 "Useful for excluding late-window 1-min or 2-min predictions.",
        )
    with _pl_col2:
        _pl_min_conf = st.number_input(
            "Min Confidence (%)",
            min_value=50.0,
            max_value=100.0,
            value=float(st.session_state.get("pl_min_conf", 50.0)),
            step=1.0,
            key="pl_min_conf",
            help="Exclude trades where AI confidence was below this threshold.",
        )

    # --- Filter presets ---
    _presets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl_filter_presets.json")

    def _load_presets():
        try:
            with open(_presets_path) as _pf:
                return json.load(_pf)
        except Exception:
            return {}

    def _save_presets(presets):
        with open(_presets_path, "w") as _pf:
            json.dump(presets, _pf, indent=2)

    _presets = _load_presets()

    with st.expander("💾 Filter Presets", expanded=False):
        _pre_save_col, _pre_load_col = st.columns(2)

        with _pre_save_col:
            st.markdown("**Save current filters**")
            _preset_name = st.text_input("Preset name", placeholder="e.g. High-odds UP", key="pl_preset_name_input")
            if st.button("Save", key="pl_preset_save_btn"):
                if _preset_name.strip():
                    _current_bucket = st.session_state.get("pl_odds_bucket", "All")
                    _presets[_preset_name.strip()] = {
                        "time": st.session_state.get("pl_time", "All Time"),
                        "bucket": _current_bucket,
                        "bucket_lo": st.session_state.get("pl_cust_lo_pct", 60) if _current_bucket == "Custom" else None,
                        "bucket_hi": st.session_state.get("pl_cust_hi_pct", 100) if _current_bucket == "Custom" else None,
                        "dir": st.session_state.get("pl_dir", "All"),
                        "model": st.session_state.get("pl_model", "All"),
                        "min_conf": float(st.session_state.get("pl_min_conf", 50.0)),
                        "skip_rules": bool(st.session_state.get("pl_skip_rules", True)),
                        "bet_scaling": bool(st.session_state.get("pl_bet_scaling", True)),
                        "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    }
                    _save_presets(_presets)
                    st.success(f"Saved '{_preset_name.strip()}'")
                    st.rerun()
                else:
                    st.warning("Enter a name before saving.")

        with _pre_load_col:
            st.markdown("**Load / delete a preset**")
            if _presets:
                _preset_choice = st.selectbox(
                    "Select preset",
                    options=list(_presets.keys()),
                    key="pl_preset_choice",
                )
                _p = _presets[_preset_choice]
                st.caption(f"Saved: {_p.get('saved_at', '—')}")
                _load_col, _del_col = st.columns(2)
                with _load_col:
                    if st.button("Load", key="pl_preset_load_btn"):
                        _pending = {
                            "time": _p.get("time", "All Time"),
                            "bucket": _p.get("bucket", "All"),
                            "bucket_range": (
                                (_p["bucket_lo"] / 100, _p["bucket_hi"] / 100)
                                if _p.get("bucket") == "Custom" and _p.get("bucket_lo") is not None
                                else None
                            ),
                            "dir": _p.get("dir", "All"),
                            "model": _p.get("model", "All"),
                            "min_conf": float(_p.get("min_conf", 50.0)),
                            "skip_rules": bool(_p.get("skip_rules", True)),
                            "bet_scaling": bool(_p.get("bet_scaling", True)),
                        }
                        st.session_state["_pl_apply_pending"] = _pending
                        st.rerun()
                with _del_col:
                    if st.button("Delete", key="pl_preset_del_btn"):
                        del _presets[_preset_choice]
                        _save_presets(_presets)
                        st.rerun()
            else:
                st.caption("No presets saved yet.")

    if sim_history.empty:
        st.info("No predictions found in the Google Sheet yet. Run predictions to start tracking!")
    else:
        completed_trades_sim = sim_history[sim_history["Outcome"].isin(["Win", "Loss"])].copy()

        # Only trades that have valid Polymarket odds recorded
        completed_with_odds = completed_trades_sim[
            completed_trades_sim["Polymarket_Odds"].notna()
            & (completed_trades_sim["Polymarket_Odds"] > 0)
        ].sort_values("Prediction_Time").reset_index(drop=True)

        # Apply all filters
        if _pl_model_filter != "All" and "Model" in completed_with_odds.columns:
            completed_with_odds = completed_with_odds[completed_with_odds["Model"] == _pl_model_filter]
        if _pl_min_conf > 50.0 and "Confidence" in completed_with_odds.columns:
            completed_with_odds = completed_with_odds[
                pd.to_numeric(completed_with_odds["Confidence"], errors="coerce") >= _pl_min_conf
            ]
        _pl_tw_hours = {"Past 24h": 24, "Past 12h": 12, "Past 1h": 1}
        if _pl_time_filter in _pl_tw_hours:
            _tw_cutoff = datetime.utcnow() - timedelta(hours=_pl_tw_hours[_pl_time_filter])
            completed_with_odds = completed_with_odds[completed_with_odds["Prediction_Time"] >= _tw_cutoff]
        if _pl_dir_filter != "All" and "Prediction" in completed_with_odds.columns:
            completed_with_odds = completed_with_odds[completed_with_odds["Prediction"] == _pl_dir_filter]
        _pl_bucket_map = {
            ">50%": (0.50, 1.01),
            "<50%": (0.0, 0.50),
            "50–60%": (0.50, 0.60), "60–70%": (0.60, 0.70),
            "70–80%": (0.70, 0.80), "80–90%": (0.80, 0.90), "90%+": (0.90, 1.01),
        }
        if _pl_odds_bucket == "Custom" and _cust_lo is not None:
            completed_with_odds = completed_with_odds[
                (completed_with_odds["Polymarket_Odds"] >= _cust_lo) &
                (completed_with_odds["Polymarket_Odds"] < _cust_hi)
            ]
        elif _pl_odds_bucket != "All":
            _blo, _bhi = _pl_bucket_map[_pl_odds_bucket]
            completed_with_odds = completed_with_odds[
                (completed_with_odds["Polymarket_Odds"] >= _blo) &
                (completed_with_odds["Polymarket_Odds"] < _bhi)
            ]

        excluded_count = len(completed_trades_sim) - len(completed_with_odds)

        # --- Auto-Optimizer ---
        st.divider()
        _opt_btn_col, _opt_result_col = st.columns([1, 3])
        _run_opt = _opt_btn_col.button(
            "⚡ Auto-Optimize Filters",
            help="Searches all filter combinations to find the highest simulated P&L (requires ≥5 trades per combo).",
        )
        if _run_opt:
            with st.spinner("Searching filter combinations…"):
                _opt_base = sim_history[
                    sim_history["Outcome"].isin(["Win", "Loss"]) &
                    sim_history["Polymarket_Odds"].notna() &
                    (sim_history["Polymarket_Odds"] > 0)
                ].copy().sort_values("Prediction_Time")

                _opt_tw_map = {"All Time": None}  # always use all-time window
                _opt_bkt_map = {
                    "All": None,
                    ">50%": (0.50, 1.01), "<50%": (0.00, 0.50),
                    "50–60%": (0.50, 0.60), "60–70%": (0.60, 0.70),
                    "70–80%": (0.70, 0.80), "80–90%": (0.80, 0.90), "90%+": (0.90, 1.01),
                    # Fine-grained 5% slices
                    "55–65%": (0.55, 0.65), "60–65%": (0.60, 0.65), "65–70%": (0.65, 0.70),
                    "65–75%": (0.65, 0.75), "70–75%": (0.70, 0.75), "75–80%": (0.75, 0.80),
                    "75–85%": (0.75, 0.85), "80–85%": (0.80, 0.85), "85–90%": (0.85, 0.90),
                    # Broad upper-tier ranges
                    "≥60%": (0.60, 1.01), "≥65%": (0.65, 1.01), "≥70%": (0.70, 1.01),
                    "≥75%": (0.75, 1.01), "≥80%": (0.80, 1.01),
                }
                _opt_dirs = ["All", "UP", "DOWN"]
                _opt_models = (
                    ["All"] + sorted(_opt_base["Model"].dropna().unique().tolist())
                    if "Model" in _opt_base.columns else ["All"]
                )
                _opt_conf_vals = [50, 55, 60, 65, 70, 75, 80]
                _opt_skip_vals = [True, False]
                _opt_scale_vals = [True, False]

                _best_bal = 1000.0
                _best_params = None

                for _otw, _otw_h in _opt_tw_map.items():
                    _s1 = _opt_base.copy()
                    if _otw_h:
                        _s1 = _s1[_s1["Prediction_Time"] >= datetime.utcnow() - timedelta(hours=_otw_h)]
                    for _obk, _obk_r in _opt_bkt_map.items():
                        _s2 = _s1.copy()
                        if _obk_r:
                            _s2 = _s2[(_s2["Polymarket_Odds"] >= _obk_r[0]) & (_s2["Polymarket_Odds"] < _obk_r[1])]
                        for _odir in _opt_dirs:
                            _s3 = _s2[_s2["Prediction"] == _odir].copy() if _odir != "All" else _s2.copy()
                            for _omdl in _opt_models:
                                _s4 = _s3[_s3["Model"] == _omdl].copy() if (_omdl != "All" and "Model" in _s3.columns) else _s3.copy()
                                for _omc in _opt_conf_vals:
                                    _s5 = _s4[_s4["Confidence"] >= _omc].copy() if (_omc > 50 and "Confidence" in _s4.columns) else _s4.copy()
                                    if len(_s5) < 5:
                                        continue
                                    for _oskip in _opt_skip_vals:
                                        for _oscale in _opt_scale_vals:
                                            _obal, _on = _quick_pl_sim(_s5, apply_skip_rules=_oskip, apply_bet_scaling=_oscale)
                                            if _obal > _best_bal:
                                                _best_bal = _obal
                                                _best_params = {
                                                    "time": _otw, "bucket": _obk,
                                                    "bucket_range": _obk_r,
                                                    "dir": _odir, "model": _omdl,
                                                    "min_conf": float(_omc),
                                                    "skip_rules": _oskip,
                                                    "bet_scaling": _oscale,
                                                    "balance": _obal, "n": _on,
                                                    "roi": (_obal - 1000.0) / 1000.0 * 100,
                                                }

                st.session_state["pl_opt_result"] = _best_params

        if st.session_state.get("pl_opt_result"):
            _r = st.session_state["pl_opt_result"]
            _skip_label = "skip rules ON" if _r.get("skip_rules", True) else "skip rules OFF"
            _scale_label = "bet scaling ON" if _r.get("bet_scaling", True) else "bet scaling OFF"
            _opt_result_col.success(
                f"**Best found:** {_r['time']} · {_r['bucket']} odds · "
                f"{_r['dir']} direction · {_r['model']} model · "
                f"Min conf {_r['min_conf']:.0f}% · {_skip_label} · {_scale_label} → "
                f"**${_r['balance']:,.2f}** ({_r['roi']:+.1f}% ROI, {_r['n']} trades)"
            )
            st.markdown(
                f"#### 🔍 Optimizer Recommendation\n\n"
                f"The auto-optimizer tested all combinations of time window, odds bucket, direction, "
                f"model, confidence threshold, skip rules, and bet scaling. "
                f"The highest simulated P&L was achieved with:\n\n"
                f"- **Time window:** {_r['time']}\n"
                f"- **Odds bucket:** {_r['bucket']}\n"
                f"- **Direction:** {_r['dir']}\n"
                f"- **Model:** {_r['model']}\n"
                f"- **Min confidence:** {_r['min_conf']:.0f}%\n"
                f"- **Skip rules:** {'ON' if _r.get('skip_rules', True) else 'OFF'}\n"
                f"- **Bet scaling:** {'ON' if _r.get('bet_scaling', True) else 'OFF'}\n\n"
                f"Simulated result: **${_r['balance']:,.2f}** balance (**{_r['roi']:+.1f}% ROI**) "
                f"over **{_r['n']}** executed trades. "
                f"This is the historically optimal configuration — it may not generalise to future trades."
            )
            if st.button("✅ Apply Optimal Filters"):
                st.session_state["_pl_apply_pending"] = {
                    "time": _r["time"], "bucket": _r["bucket"],
                    "bucket_range": _r.get("bucket_range"),
                    "dir": _r["dir"], "model": _r["model"], "min_conf": _r["min_conf"],
                    "skip_rules": _r.get("skip_rules", True),
                    "bet_scaling": _r.get("bet_scaling", True),
                }
                st.rerun()
        elif not _run_opt:
            # Show a placeholder hint when no result yet
            _opt_result_col.caption("Click ⚡ to find the filter combination with the highest historical P&L.")

        st.divider()

        if completed_with_odds.empty:
            st.info(
                "No completed predictions with Polymarket odds yet. "
                "Odds are recorded starting from the next prediction you generate."
            )
            if excluded_count > 0:
                st.caption(
                    f"{excluded_count} historical prediction(s) excluded — "
                    "Polymarket odds were not recorded at that time."
                )
        else:
            # --- Simulation constants ---
            STARTING_BALANCE = 1000.0
            MIN_BET_PCT = 0.025          # 2.5% — floor for all bets
            MAX_BET_PCT = 0.05           # 5.0% — hard cap
            HIGH_ODDS_THRESHOLD = 0.65   # market strongly agrees → payout < 1.54x
            MIN_CONF_FOR_HIGH_ODDS = 60.0  # require this confidence to bet into a high-odds market
            CONTRARIAN_SKIP_THRESHOLD = 0.30  # PM gives <30% to model's direction → skip

            balance = STARTING_BALANCE
            trades_log = []

            for _, row in completed_with_odds.iterrows():
                conf = float(row["Confidence"])
                odds = float(row["Polymarket_Odds"])

                if _apply_skip_rules and odds < CONTRARIAN_SKIP_THRESHOLD:
                    # Strongly contrarian — PM strongly disagrees with model's prediction
                    actual_outcome = row["Outcome"]
                    if actual_outcome == "Win":
                        hypothetical_pnl = round(balance * MIN_BET_PCT * (1.0 / odds - 1.0), 2)
                        skip_label = "⏭️ Skipped Contrarian (would Win)"
                    else:
                        hypothetical_pnl = round(-balance * MIN_BET_PCT, 2)
                        skip_label = "⏭️ Skipped Contrarian (would Loss)"
                    trades_log.append({
                        "Time": fmt_et(row["Prediction_Time"], "%m/%d %H:%M %Z"),
                        "Direction": row["Prediction"],
                        "Confidence": f"{conf:.1f}%",
                        "Odds": f"{odds:.3f} ({odds*100:.1f}%)",
                        "Bet %": "—",
                        "Bet $": "—",
                        "Outcome": skip_label,
                        "P&L": 0,
                        "What-If P&L": hypothetical_pnl,
                        "Balance": round(balance, 2),
                        "bet_amt": 0,
                    })
                    continue

                if odds >= HIGH_ODDS_THRESHOLD:
                    # Market is very confident your direction wins → small payout
                    # Skip unless AI is also highly confident (or skip rules disabled)
                    if _apply_skip_rules and conf < MIN_CONF_FOR_HIGH_ODDS:
                        actual_outcome = row["Outcome"]   # "Win" or "Loss" from sheet
                        if actual_outcome == "Win":
                            hypothetical_pnl = round(balance * MIN_BET_PCT * (1.0 / odds - 1.0), 2)
                            skip_label = "⏭️ Skipped (would Win)"
                        else:
                            hypothetical_pnl = round(-balance * MIN_BET_PCT, 2)
                            skip_label = "⏭️ Skipped (would Loss)"
                        trades_log.append({
                            "Time": fmt_et(row["Prediction_Time"], "%m/%d %H:%M %Z"),
                            "Direction": row["Prediction"],
                            "Confidence": f"{conf:.1f}%",
                            "Odds": f"{odds:.3f} ({odds*100:.1f}%)",
                            "Bet %": "—",
                            "Bet $": "—",
                            "Outcome": skip_label,
                            "P&L": 0,
                            "What-If P&L": hypothetical_pnl,
                            "Balance": round(balance, 2),
                            "bet_amt": 0,
                        })
                        continue
                    bet_pct = MIN_BET_PCT   # even with high conf, don't oversize a small-payout bet

                elif odds < 0.5 and _apply_bet_scaling:
                    # Contrarian — market disagrees, payout is large → scale up with confidence
                    # Linear scale: 2.5% at conf=50 → 5% at conf=100
                    bet_pct = min(MAX_BET_PCT, MIN_BET_PCT + (conf - 50) / 100 * MAX_BET_PCT)
                    bet_pct = max(MIN_BET_PCT, bet_pct)   # floor at 2.5%

                else:
                    # Standard zone (0.50 ≤ odds < 0.65): flat 2.5%
                    bet_pct = MIN_BET_PCT

                bet_amount = balance * bet_pct

                if row["Outcome"] == "Win":
                    profit = bet_amount * (1.0 / odds - 1.0)
                    balance += profit
                    pnl = profit
                else:
                    balance -= bet_amount
                    pnl = -bet_amount

                trades_log.append({
                    "Time": fmt_et(row["Prediction_Time"], "%m/%d %H:%M %Z"),
                    "Direction": row["Prediction"],
                    "Confidence": f"{conf:.1f}%",
                    "Odds": f"{odds:.3f} ({odds*100:.1f}%)",
                    "Bet %": f"{bet_pct*100:.1f}%",
                    "Bet $": round(bet_amount, 2),
                    "Outcome": row["Outcome"],
                    "P&L": round(pnl, 2),
                    "What-If P&L": "—",
                    "Balance": round(balance, 2),
                    "bet_amt": round(bet_amount, 2),
                })

            trades_log_display = trades_log
            total_pnl = balance - STARTING_BALANCE
            roi_pct = (total_pnl / STARTING_BALANCE) * 100

            # --- KPI row ---
            st.divider()
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Starting Balance", "$1,000.00")
            kpi2.metric(
                "Current Balance",
                f"${balance:,.2f}",
                delta=f"${total_pnl:+,.2f}",
                delta_color="normal",
            )
            kpi3.metric("Total P&L", f"${total_pnl:+,.2f}")
            kpi4.metric("ROI", f"{roi_pct:+.2f}%")
            st.caption(
                f"Bet-sizing: contrarian (<50% odds) scales 2.5–5% by confidence · "
                f"high-odds (≥{HIGH_ODDS_THRESHOLD*100:.0f}%) require ≥{MIN_CONF_FOR_HIGH_ODDS:.0f}% confidence · "
                f"cap {MAX_BET_PCT*100:.0f}%"
            )
            st.divider()

            # --- Balance-over-time chart ---
            sim_df = pd.DataFrame(trades_log_display).drop(columns=["bet_amt"], errors="ignore")
            sim_df.index.name = "Trade #"

            fig_bal = go.Figure()
            fig_bal.add_trace(go.Scatter(
                x=list(range(len(sim_df))),
                y=sim_df["Balance"],
                mode="lines+markers",
                line=dict(color="#00d084", width=2),
                marker=dict(
                    color=["green" if p >= 0 else "red" for p in sim_df["P&L"]],
                    size=8,
                ),
                hovertext=[
                    f"Trade {i+1}: {row['Direction']} | {row['Outcome']} | "
                    f"P&L: ${row['P&L']:+.2f} | Balance: ${row['Balance']:,.2f}"
                    for i, row in sim_df.iterrows()
                ],
                hoverinfo="text",
                name="Balance",
            ))
            fig_bal.add_hline(
                y=STARTING_BALANCE,
                line_dash="dot",
                line_color="gray",
                annotation_text="Starting $1,000",
                annotation_position="bottom right",
            )
            fig_bal.update_layout(
                title="Portfolio Balance Over Trades",
                xaxis_title="Trade #",
                yaxis_title="Balance ($)",
                template="plotly_dark",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_bal, use_container_width=True)

            # --- Trades table ---
            st.markdown("#### Trade Log")

            def highlight_sim_outcome(val):
                if val == "Win":
                    return "background-color: rgba(0, 255, 0, 0.15)"
                elif val == "Loss":
                    return "background-color: rgba(255, 0, 0, 0.15)"
                elif "would Win" in str(val):
                    return "background-color: rgba(0, 255, 0, 0.07)"   # muted green
                elif "would Loss" in str(val):
                    return "background-color: rgba(255, 0, 0, 0.07)"   # muted red
                return ""

            st.dataframe(
                sim_df.style.map(highlight_sim_outcome, subset=["Outcome"]),
                use_container_width=True,
            )

            skipped_count = sum(1 for t in trades_log_display if "Skipped" in str(t.get("Outcome", "")))
            if skipped_count:
                would_win = sum(1 for t in trades_log_display if "would Win" in str(t.get("Outcome", "")))
                would_loss = sum(1 for t in trades_log_display if "would Loss" in str(t.get("Outcome", "")))
                st.caption(
                    f"{skipped_count} bet(s) skipped — high-odds market with insufficient AI confidence "
                    f"({would_win} would have Won, {would_loss} would have Lost)."
                )

            if excluded_count > 0:
                st.info(
                    f"Note: {excluded_count} completed prediction(s) excluded from simulation "
                    "— Polymarket odds were not recorded at prediction time (pre-dates this feature)."
                )

# ── Tab 4: Odds vs Performance ────────────────────────────────────────────────
with tab4:
    st.markdown("## 📈 Odds vs Performance")

    # Load history (may already be loaded; reuse the same cache-backed function)
    _t5_history, _ = load_history_from_sheets()

    if _t5_history.empty or "Polymarket_Odds" not in _t5_history.columns:
        st.info("No data with Polymarket odds yet.")
    else:
        _odds_df = _t5_history[
            _t5_history["Outcome"].isin(["Win", "Loss"]) &
            _t5_history["Polymarket_Odds"].notna() &
            (_t5_history["Polymarket_Odds"] > 0)
        ].copy()
        _odds_df["Odds_Pct"] = _odds_df["Polymarket_Odds"] * 100

        if _odds_df.empty:
            st.info("No completed trades with Polymarket odds recorded yet.")
        else:
            # ── Filters ──────────────────────────────────────────────────────
            fc1, fc2, fc3, fc4, fc5 = st.columns(5)
            _bucket_opts = ["All", ">50%", "<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"]
            _odds_bucket = fc1.selectbox("Odds bucket", _bucket_opts, key="t5_bucket")
            _dir_filter = fc2.selectbox("Direction", ["All", "UP", "DOWN"], key="t5_dir")
            _conf_min = fc3.number_input("Min Confidence (%)", 0, 100, 0, step=5, key="t5_conf")
            _t5_models = (
                ["All"] + sorted(_odds_df["Model"].dropna().unique().tolist())
                if "Model" in _odds_df.columns
                else ["All"]
            )
            _odds_model = fc4.selectbox("Model", _t5_models, key="t5_model")
            _t5_versions = (
                ["All"] + sorted(
                    _odds_df["Model_Version"].dropna().astype(str).unique().tolist(),
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                if "Model_Version" in _odds_df.columns
                else ["All"]
            )
            _odds_ver = fc5.selectbox("Model Version", _t5_versions, key="t5_version")

            _bucket_ranges = {
                ">50%": (50, 101),
                "<50%": (0, 50),
                "50–60%": (50, 60), "60–70%": (60, 70),
                "70–80%": (70, 80), "80–90%": (80, 90), "90%+": (90, 101),
            }

            # Apply filters to the scatter dataset
            _fdf = _odds_df.copy()
            if _odds_bucket != "All":
                _blo, _bhi = _bucket_ranges[_odds_bucket]
                _fdf = _fdf[(_fdf["Odds_Pct"] >= _blo) & (_fdf["Odds_Pct"] < _bhi)]
            if _dir_filter != "All":
                _fdf = _fdf[_fdf["Prediction"] == _dir_filter]
            if _conf_min > 0 and "Confidence" in _fdf.columns:
                _fdf = _fdf[_fdf["Confidence"] >= _conf_min]
            if _odds_model != "All" and "Model" in _fdf.columns:
                _fdf = _fdf[_fdf["Model"] == _odds_model]
            if _odds_ver != "All" and "Model_Version" in _fdf.columns:
                _fdf = _fdf[_fdf["Model_Version"].astype(str) == _odds_ver]

            # Base for bucket table: apply all filters EXCEPT odds bucket
            _fdf_base = _odds_df.copy()
            if _dir_filter != "All":
                _fdf_base = _fdf_base[_fdf_base["Prediction"] == _dir_filter]
            if _conf_min > 0 and "Confidence" in _fdf_base.columns:
                _fdf_base = _fdf_base[_fdf_base["Confidence"] >= _conf_min]
            if _odds_model != "All" and "Model" in _fdf_base.columns:
                _fdf_base = _fdf_base[_fdf_base["Model"] == _odds_model]
            if _odds_ver != "All" and "Model_Version" in _fdf_base.columns:
                _fdf_base = _fdf_base[_fdf_base["Model_Version"].astype(str) == _odds_ver]

            if _fdf.empty:
                st.warning("No trades match the selected filters.")
            else:
                # ── Scatterplot: Odds vs Confidence, coloured by outcome ────
                import plotly.graph_objects as _go
                _wins_s = _fdf[_fdf["Outcome"] == "Win"]
                _loss_s = _fdf[_fdf["Outcome"] == "Loss"]

                _fig5 = _go.Figure()
                _fig5.add_trace(_go.Scatter(
                    x=_wins_s["Odds_Pct"],
                    y=_wins_s["Confidence"],
                    mode="markers",
                    marker=dict(color="rgba(0,200,100,0.65)", size=8),
                    name="Win",
                    hovertemplate="Odds: %{x:.1f}%<br>Conf: %{y:.1f}%<br>Win<extra></extra>",
                ))
                _fig5.add_trace(_go.Scatter(
                    x=_loss_s["Odds_Pct"],
                    y=_loss_s["Confidence"],
                    mode="markers",
                    marker=dict(color="rgba(220,50,50,0.65)", size=8),
                    name="Loss",
                    hovertemplate="Odds: %{x:.1f}%<br>Conf: %{y:.1f}%<br>Loss<extra></extra>",
                ))
                _fig5.update_layout(
                    title="Polymarket Odds vs Model Confidence (coloured by outcome)",
                    xaxis_title="Polymarket Odds (%)",
                    yaxis_title="Model Confidence (%)",
                    height=420,
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(_fig5, use_container_width=True)

                # ── Stats table by odds bucket (always uses full _odds_df, not filtered) ──
                st.markdown("#### Win Rate by Odds Bucket")
                _bucket_rows = []
                for _blabel, (_blo, _bhi) in _bucket_ranges.items():
                    _bdf = _fdf_base[(_fdf_base["Odds_Pct"] >= _blo) & (_fdf_base["Odds_Pct"] < _bhi)]
                    if len(_bdf) == 0:
                        continue
                    _bwins = len(_bdf[_bdf["Outcome"] == "Win"])
                    _bwr = _bwins / len(_bdf) * 100
                    _bavg_conf = _bdf["Confidence"].mean() if "Confidence" in _bdf.columns else None
                    _bucket_rows.append({
                        "Odds Range": _blabel,
                        "Trades": len(_bdf),
                        "Wins": _bwins,
                        "Win Rate": f"{_bwr:.1f}%",
                        "Avg Confidence": f"{_bavg_conf:.1f}%" if _bavg_conf is not None else "—",
                    })
                if _bucket_rows:
                    st.dataframe(pd.DataFrame(_bucket_rows), hide_index=True, use_container_width=True)

            # ── Trends & Patterns blurb (always uses full _odds_df for signal strength) ──
            st.markdown("#### 🔍 Trend & Pattern Analysis")
            _insights = []

            # Time of day
            if "Prediction_Time" in _odds_df.columns and len(_odds_df) >= 10:
                _tdf = _odds_df.copy()
                _tdf["_hour"] = _tdf["Prediction_Time"].dt.hour
                _hourly = (
                    _tdf.groupby("_hour")
                    .apply(lambda x: pd.Series({
                        "trades": len(x),
                        "win_rate": (x["Outcome"] == "Win").mean() * 100,
                    }))
                    .reset_index()
                )
                _hourly = _hourly[_hourly["trades"] >= 3]
                if not _hourly.empty:
                    _bh = _hourly.loc[_hourly["win_rate"].idxmax()]
                    _wh = _hourly.loc[_hourly["win_rate"].idxmin()]
                    _bh_et = int((_bh["_hour"] - 5) % 24)
                    _wh_et = int((_wh["_hour"] - 5) % 24)
                    _insights.append(
                        f"**⏰ Time of Day:** Best hour is **{int(_bh['_hour'])}:00 UTC "
                        f"({_bh_et}:00 ET)** with a {_bh['win_rate']:.0f}% win rate "
                        f"({int(_bh['trades'])} trades). "
                        f"Weakest hour is {int(_wh['_hour'])}:00 UTC ({_wh_et}:00 ET) "
                        f"at {_wh['win_rate']:.0f}% ({int(_wh['trades'])} trades). "
                        f"Consider prioritising trades during your peak UTC window."
                    )

            # Confidence tier analysis
            if "Confidence" in _odds_df.columns and len(_odds_df) >= 10:
                _ctier_defs = [(50, 60), (60, 70), (70, 80), (80, 101)]
                _ctier_rows = []
                for _clo, _chi in _ctier_defs:
                    _ctdf = _odds_df[(_odds_df["Confidence"] >= _clo) & (_odds_df["Confidence"] < _chi)]
                    if len(_ctdf) >= 3:
                        _ctier_rows.append(
                            (_clo, min(_chi, 100), len(_ctdf), (_ctdf["Outcome"] == "Win").mean() * 100)
                        )
                if _ctier_rows:
                    _best_ct = max(_ctier_rows, key=lambda x: x[3])
                    _note = (
                        " Higher confidence does not always equal higher win rate — check if the 80%+ tier underperforms."
                        if len(_ctier_rows) > 1 and _best_ct[1] < 80
                        else ""
                    )
                    _insights.append(
                        f"**🎯 Confidence Tiers:** The **{_best_ct[0]}–{_best_ct[1]}% confidence band** "
                        f"achieves the highest win rate at **{_best_ct[3]:.0f}%** "
                        f"({_best_ct[2]} trades).{_note}"
                    )

            # Direction at high confidence
            if "Prediction" in _odds_df.columns and "Confidence" in _odds_df.columns and len(_odds_df) >= 10:
                _high_conf = _odds_df[_odds_df["Confidence"] >= 70]
                if len(_high_conf) >= 5:
                    _up_hc = _high_conf[_high_conf["Prediction"] == "UP"]
                    _dn_hc = _high_conf[_high_conf["Prediction"] == "DOWN"]
                    _up_wr_hc = ((_up_hc["Outcome"] == "Win").mean() * 100) if len(_up_hc) >= 3 else None
                    _dn_wr_hc = ((_dn_hc["Outcome"] == "Win").mean() * 100) if len(_dn_hc) >= 3 else None
                    if _up_wr_hc is not None and _dn_wr_hc is not None:
                        _better_hc = "UP (Long)" if _up_wr_hc >= _dn_wr_hc else "DOWN (Short)"
                        _insights.append(
                            f"**📐 Direction at High Confidence (≥70%):** When the model is highly confident, "
                            f"**{_better_hc}** predictions win more often "
                            f"(UP: {_up_wr_hc:.0f}% over {len(_up_hc)} trades, "
                            f"DOWN: {_dn_wr_hc:.0f}% over {len(_dn_hc)} trades). "
                            f"Consider prioritising {_better_hc} calls when confidence exceeds 70%."
                        )
                    elif _up_wr_hc is not None:
                        _insights.append(
                            f"**📐 Direction at High Confidence (≥70%):** Only UP predictions have enough data "
                            f"({len(_up_hc)} trades, {_up_wr_hc:.0f}% win rate) — insufficient DOWN data."
                        )
                    elif _dn_wr_hc is not None:
                        _insights.append(
                            f"**📐 Direction at High Confidence (≥70%):** Only DOWN predictions have enough data "
                            f"({len(_dn_hc)} trades, {_dn_wr_hc:.0f}% win rate) — insufficient UP data."
                        )

            # Model vs market agreement (based on odds at bet time, not PM_Resolution outcome)
            if "Polymarket_Odds" in _odds_df.columns and "Confidence" in _odds_df.columns:
                _mma_df = _odds_df[_odds_df["Polymarket_Odds"].notna()].copy()
                _mma_df["_market_agrees"] = _mma_df["Polymarket_Odds"] > 0.5
                _agree_mkt = _mma_df[_mma_df["_market_agrees"]]
                _disagree_mkt = _mma_df[~_mma_df["_market_agrees"]]

                if len(_agree_mkt) >= 5 and len(_disagree_mkt) >= 5:
                    _awr_mkt = (_agree_mkt["Outcome"] == "Win").mean() * 100
                    _dwr_mkt = (_disagree_mkt["Outcome"] == "Win").mean() * 100
                    _a_avg_conf = _agree_mkt["Confidence"].mean()
                    _d_avg_conf = _disagree_mkt["Confidence"].mean()
                    _a_avg_odds = _agree_mkt["Polymarket_Odds"].mean() * 100
                    _d_avg_odds = _disagree_mkt["Polymarket_Odds"].mean() * 100

                    _conf_odds_corr = _mma_df[["Confidence", "Polymarket_Odds"]].corr().iloc[0, 1]
                    _corr_label = (
                        "strong positive" if _conf_odds_corr > 0.4
                        else "moderate positive" if _conf_odds_corr > 0.15
                        else "weak/no" if _conf_odds_corr > -0.15
                        else "inverse"
                    )

                    _signal_mkt = (
                        "Strong signal — betting with market odds correlates meaningfully with outcomes."
                        if abs(_awr_mkt - _dwr_mkt) > 10
                        else "Marginal signal — contrarian and aligned bets perform similarly."
                    )

                    _insights.append(
                        f"**🤝 Model vs Market Odds Alignment:** When Polymarket odds favour the model's direction (odds >50%), "
                        f"win rate is **{_awr_mkt:.0f}%** ({len(_agree_mkt)} trades, avg odds {_a_avg_odds:.0f}%, avg conf {_a_avg_conf:.0f}%). "
                        f"When the model is contrarian (odds ≤50%), win rate is {_dwr_mkt:.0f}% "
                        f"({len(_disagree_mkt)} trades, avg odds {_d_avg_odds:.0f}%, avg conf {_d_avg_conf:.0f}%). "
                        f"Confidence–odds correlation: **{_corr_label}** (r={_conf_odds_corr:.2f}). "
                        f"{_signal_mkt}"
                    )
                elif len(_agree_mkt) >= 5:
                    _awr_mkt = (_agree_mkt["Outcome"] == "Win").mean() * 100
                    _insights.append(
                        f"**🤝 Model vs Market Odds Alignment:** Only market-aligned trades have enough data "
                        f"({len(_agree_mkt)} trades, {_awr_mkt:.0f}% win rate) — insufficient contrarian data yet."
                    )
                elif len(_disagree_mkt) >= 5:
                    _dwr_mkt = (_disagree_mkt["Outcome"] == "Win").mean() * 100
                    _insights.append(
                        f"**🤝 Model vs Market Odds Alignment:** Only contrarian trades have enough data "
                        f"({len(_disagree_mkt)} trades, {_dwr_mkt:.0f}% win rate) — insufficient aligned data yet."
                    )

            if _insights:
                for _ins in _insights:
                    st.markdown(_ins)
                    st.markdown("")
            else:
                st.info("Not enough data yet to surface meaningful trends (need ≥10 completed trades with Polymarket odds).")

with tab5:
    st.markdown("### 🧠 Model Health")
    st.markdown(
        "Shows training data coverage, prediction accuracy, and performance trends for each of the "
        "5 horizon models (1min–5min). Each model predicts BTC direction with a different amount "
        "of time remaining before the 5-minute window closes."
    )

    # --- Model file metadata ---
    _mf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_5m_rf_model.joblib")
    try:
        _mf_mtime = datetime.utcfromtimestamp(os.path.getmtime(_mf_path))
        _mf_size_kb = os.path.getsize(_mf_path) / 1024
        _mf_age_days = (datetime.utcnow() - _mf_mtime).days
        _mf_age_str = f"{_mf_age_days}d ago" if _mf_age_days > 0 else "today"
    except OSError:
        _mf_mtime = None
        _mf_size_kb = 0
        _mf_age_days = None
        _mf_age_str = "unknown"

    # --- Load model_metadata.json if present ---
    _model_meta = {}
    try:
        _meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.json")
        with open(_meta_path) as _f:
            _model_meta = json.load(_f)
    except Exception:
        pass

    # --- Update history ---
    _model_history = []
    try:
        _hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_history.json")
        with open(_hist_path) as _f:
            _model_history = json.load(_f)
    except Exception:
        pass

    # Bootstrap from metadata if history file is absent but metadata exists
    if not _model_history and _model_meta:
        if _model_meta.get("retrained_at_utc"):
            _model_history = [{
                "retrained_at_utc": _model_meta["retrained_at_utc"],
                "script": _model_meta.get("script", "—"),
                "total_rows": _model_meta.get("total_rows"),
                "new_rows_added": _model_meta.get("new_rows_added"),
                "data_start": _model_meta.get("data_start"),
                "data_end": _model_meta.get("data_end"),
            }]
            if _model_meta.get("previous_retrained_at_utc") and _model_meta["previous_retrained_at_utc"] != "—":
                _model_history.insert(0, {
                    "retrained_at_utc": _model_meta["previous_retrained_at_utc"],
                    "script": "—",
                    "total_rows": _model_meta.get("previous_total_rows"),
                    "new_rows_added": None,
                    "data_start": None,
                    "data_end": None,
                })

    # --- Load live feature importances from joblib model ---
    _live_importances = {}  # {hz_str: {feature: importance}}
    try:
        _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_5m_rf_model.joblib")
        _loaded_model = joblib.load(_model_path)
        if isinstance(_loaded_model, dict):
            # train_model.py format: {1: rf, 2: rf, ...}
            for _hz_key, _rf in _loaded_model.items():
                if hasattr(_rf, "feature_importances_") and hasattr(_rf, "feature_names_in_"):
                    _live_importances[str(_hz_key)] = dict(
                        zip(_rf.feature_names_in_.tolist(), _rf.feature_importances_.tolist())
                    )
        elif hasattr(_loaded_model, "feature_importances_") and hasattr(_loaded_model, "feature_names_in_"):
            # update_brain.py format: single model
            _single_fi = dict(zip(_loaded_model.feature_names_in_.tolist(), _loaded_model.feature_importances_.tolist()))
            for _hz_key in range(1, 6):
                _live_importances[str(_hz_key)] = _single_fi
    except Exception:
        pass
    # Fallback to metadata feature_importance if joblib unavailable
    if not _live_importances and _model_meta.get("feature_importance"):
        _meta_fi = _model_meta["feature_importance"]
        _is_per_hz = isinstance(next(iter(_meta_fi.values()), None), dict)
        if _is_per_hz:
            _live_importances = _meta_fi
        else:
            for _hz_key in range(1, 6):
                _live_importances[str(_hz_key)] = _meta_fi

    # --- Training data stats ---
    _train_stats = _load_training_stats()
    _csv_err = _train_stats.get("_error")
    if _csv_err and "FileNotFoundError" not in _csv_err and "No such file" not in _csv_err:
        st.warning(f"Could not load training CSV: {_csv_err}")
    _valid_stats = {k: v for k, v in _train_stats.items() if k != "_error"}
    _total_train_rows = sum(v["rows"] for v in _valid_stats.values()) if _valid_stats else 0
    _all_earliests = [v["earliest"] for v in _valid_stats.values() if v.get("earliest") is not None]
    _all_latests = [v["latest"] for v in _valid_stats.values() if v.get("latest") is not None]
    _data_start = pd.to_datetime(min(_all_earliests)) if _all_earliests else None
    _data_end = pd.to_datetime(max(_all_latests)) if _all_latests else None

    # --- Prefer metadata timestamp; fall back to file mtime ---
    if _model_meta.get("retrained_at_utc"):
        _retrain_display = _model_meta["retrained_at_utc"][:10]   # "YYYY-MM-DD"
        _retrain_dt = datetime.strptime(_model_meta["retrained_at_utc"], "%Y-%m-%d %H:%M:%S")
        _mf_age_days = (datetime.utcnow() - _retrain_dt).days
        _mf_age_str = f"{_mf_age_days}d ago" if _mf_age_days > 0 else "today"
    else:
        _retrain_display = _mf_mtime.strftime("%Y-%m-%d") if _mf_mtime else "—"

    # --- Fallback to metadata row counts if CSV stats unavailable ---
    if not _total_train_rows and _model_meta.get("total_rows"):
        _total_train_rows = _model_meta["total_rows"]
    if _data_start is None and _model_meta.get("data_start"):
        _data_start = pd.to_datetime(_model_meta["data_start"])
    if _data_end is None and _model_meta.get("data_end"):
        _data_end = pd.to_datetime(_model_meta["data_end"])

    _model_version = _model_meta.get("model_version")

    # --- Summary bar ---
    _smry_cols = st.columns(5)
    with _smry_cols[0]:
        st.metric("Last Retrained", _retrain_display, delta=_mf_age_str, delta_color="inverse")
    with _smry_cols[1]:
        st.metric("Model Version", f"v{_model_version}" if _model_version else "—")
    with _smry_cols[2]:
        st.metric("Training Rows (total)", f"{_total_train_rows:,}" if _total_train_rows else "—")
    with _smry_cols[3]:
        st.metric("Data From", _data_start.strftime("%Y-%m-%d") if _data_start is not None else "—")
    with _smry_cols[4]:
        st.metric("Data To", _data_end.strftime("%Y-%m-%d") if _data_end is not None else "—")

    # --- Provenance caption ---
    if _model_meta:
        _script = _model_meta.get("script", "unknown")
        _new_rows = _model_meta.get("new_rows_added")
        _prev_date = _model_meta.get("previous_retrained_at_utc", "—")
        _parts = [f"Retrained via **{_script}**"]
        if _new_rows is not None and _new_rows > 0:
            _parts.append(f"**+{_new_rows:,} new rows** added")
        elif _new_rows == 0:
            _parts.append("no new rows added")
        if _prev_date and _prev_date != "—":
            _parts.append(f"previous retrain: {_prev_date} UTC")
        st.caption("  ·  ".join(_parts))
    else:
        st.caption("No metadata file found — run a retrain to generate provenance data.")

    st.divider()

    # --- Per-horizon model cards ---
    _mh_history, _ = load_history_from_sheets()
    _now_utc = datetime.utcnow()
    _30d_cutoff = _now_utc - timedelta(days=30)

    for _hz in range(1, 6):
        _hz_label = f"{_hz}min"
        _hz_data = _valid_stats.get(_hz, {})
        _hz_rows = _hz_data.get("rows", 0)
        _hz_earliest = pd.to_datetime(_hz_data.get("earliest")) if _hz_data.get("earliest") else None
        _hz_latest = pd.to_datetime(_hz_data.get("latest")) if _hz_data.get("latest") else None
        # Fallback: use metadata rows_per_horizon if CSV stats missing
        if not _hz_rows and _model_meta.get("rows_per_horizon"):
            _hz_rows = _model_meta["rows_per_horizon"].get(str(_hz), 0)
        # Fallback to global data span when CSV stats unavailable
        if _hz_earliest is None:
            _hz_earliest = _data_start
        if _hz_latest is None:
            _hz_latest = _data_end

        # Prediction history for this horizon
        if _mh_history is not None and len(_mh_history) > 0 and "Model" in _mh_history.columns:
            _hz_hist = _mh_history[
                (_mh_history["Model"] == _hz_label) &
                (_mh_history["Outcome"].isin(["Win", "Loss"]))
            ].copy()
        else:
            _hz_hist = pd.DataFrame()

        _n_preds = len(_hz_hist)

        # All-time win rate
        if _n_preds >= 5:
            _wr_all = (_hz_hist["Outcome"] == "Win").mean() * 100
        else:
            _wr_all = None

        # Recent win rate (last 30 days)
        if _mh_history is not None and "Prediction_Time" in _hz_hist.columns and _n_preds >= 5:
            _hz_recent = _hz_hist[pd.to_datetime(_hz_hist["Prediction_Time"], errors="coerce") >= _30d_cutoff]
            _wr_recent = (_hz_recent["Outcome"] == "Win").mean() * 100 if len(_hz_recent) >= 5 else None
        else:
            _wr_recent = None

        # Performance delta (pp)
        _delta_pp = (_wr_recent - _wr_all) if (_wr_all is not None and _wr_recent is not None) else None

        # Status badge
        if _mf_age_days is None:
            _status = "⚠️ Monitor"
        elif (
            _mf_age_days > 30
            or (_wr_recent is not None and _wr_recent < 45)
            or (_delta_pp is not None and _delta_pp < -8)
        ):
            _status = "🔴 Retrain recommended"
        elif (
            _mf_age_days > 14
            or (_delta_pp is not None and _delta_pp < -4)
        ):
            _status = "⚠️ Monitor"
        else:
            _status = "✅ Healthy"

        _hz_version_tag = f" v{_model_version}" if _model_version else ""
        with st.expander(f"**{_hz_label} model{_hz_version_tag}** — {_status}", expanded=True):
            _card_cols = st.columns(5)
            with _card_cols[0]:
                st.metric("Training rows", f"{_hz_rows:,}" if _hz_rows else "—")
            with _card_cols[1]:
                st.metric("Predictions logged", str(_n_preds) if _n_preds else "—")
            with _card_cols[2]:
                st.metric(
                    "All-time win rate",
                    f"{_wr_all:.1f}%" if _wr_all is not None else "< 5 trades",
                )
            with _card_cols[3]:
                if _wr_recent is not None:
                    _delta_label = f"{_delta_pp:+.1f} pp vs all-time" if _delta_pp is not None else None
                    st.metric("Recent win rate (30d)", f"{_wr_recent:.1f}%", delta=_delta_label)
                else:
                    st.metric("Recent win rate (30d)", "< 5 trades")
            with _card_cols[4]:
                st.caption("Training data span")
                if _hz_earliest and _hz_latest:
                    st.markdown(
                        f"<div style='font-size:0.82em;line-height:1.4'>"
                        f"{_hz_earliest.strftime('%Y-%m-%d')}<br>→ {_hz_latest.strftime('%Y-%m-%d')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("—")

            # Feature importance for this horizon
            _hz_fi = _live_importances.get(str(_hz))
            if not _hz_fi:
                _meta_fi = _model_meta.get("feature_importance", {})
                if _meta_fi:
                    _is_per_hz = isinstance(next(iter(_meta_fi.values()), None), dict)
                    _hz_fi = _meta_fi.get(str(_hz)) if _is_per_hz else _meta_fi or None
            if _hz_fi:
                st.markdown("**Top Features**")
                _fi_df = pd.DataFrame(
                    sorted(_hz_fi.items(), key=lambda x: x[1], reverse=True),
                    columns=["Feature", "Importance"],
                )
                _fi_df["Importance"] = _fi_df["Importance"].map(lambda v: f"{v:.4f}")
                st.dataframe(_fi_df, hide_index=True, use_container_width=True)

    st.divider()

    # --- Retraining guidance ---
    st.markdown("#### Retraining Guidance")
    st.info(
        "**How often should these models be retrained?**\n\n"
        "Random Forest models trained on short-term BTC technical indicators (RSI, MACD, EMA, Bollinger Bands) "
        "typically degrade within **2–6 weeks** as market regimes shift. "
        "Volatility regime changes — such as post-halving periods, macro shocks, or sustained trending markets — "
        "can accelerate degradation significantly.\n\n"
        "**Recommended cadence:** Retrain every **2–4 weeks**, or immediately when:\n"
        "- Rolling 7-day accuracy drops >5 percentage points below your all-time baseline\n"
        "- A sustained new volatility regime begins (e.g. BTC breaks a major level with high volume)\n"
        "- Model age exceeds 30 days\n\n"
        "**To retrain:** Run `python update_brain.py` from the BTCPredictor directory. "
        "This fetches the latest Kraken 1-min candles, stitches them to the existing dataset, and retrains all 5 horizon models."
    )

    st.divider()
    with st.expander("📋 Update History", expanded=False):
        if _model_history:
            _hist_rows = []
            for _e in reversed(_model_history):  # newest first
                _hist_rows.append({
                    "Version": f"v{_e['model_version']}" if _e.get("model_version") else "—",
                    "Date / Time (UTC)": _e.get("retrained_at_utc", "—"),
                    "Script": _e.get("script", "—"),
                    "Total Rows": f"{_e['total_rows']:,}" if _e.get("total_rows") else "—",
                    "New Rows": f"+{_e['new_rows_added']:,}" if _e.get("new_rows_added") else "—",
                    "Data Start": str(_e["data_start"])[:10] if _e.get("data_start") else "—",
                    "Data End":   str(_e["data_end"])[:10]   if _e.get("data_end")   else "—",
                })
            st.dataframe(
                pd.DataFrame(_hist_rows),
                hide_index=True,
                use_container_width=True,
            )
            _hist_with_fi = [e for e in _model_history if e.get("feature_importance")]
            if _hist_with_fi:
                st.divider()
                st.markdown("**Feature Importance by Retrain**")
                for _e in reversed(_hist_with_fi):  # newest first
                    _e_ver = f"v{_e['model_version']} · " if _e.get("model_version") else ""
                    _fi_label = f"{_e_ver}{_e.get('retrained_at_utc', '—')} — {_e.get('script', '—')}"
                    with st.expander(_fi_label, expanded=False):
                        _fi_data = _e["feature_importance"]
                        _is_per_horizon = isinstance(next(iter(_fi_data.values()), None), dict)
                        if _is_per_horizon:
                            for _h_str in sorted(_fi_data.keys()):
                                st.caption(f"Horizon {_h_str}min")
                                _fi_df = pd.DataFrame(
                                    sorted(_fi_data[_h_str].items(), key=lambda x: x[1], reverse=True),
                                    columns=["Feature", "Importance"],
                                )
                                st.dataframe(_fi_df, hide_index=True, use_container_width=True)
                        else:
                            _fi_df = pd.DataFrame(
                                sorted(_fi_data.items(), key=lambda x: x[1], reverse=True),
                                columns=["Feature", "Importance"],
                            )
                            st.dataframe(_fi_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No update history recorded yet. Run train_model.py or update_brain.py to begin tracking.")

# ── Tab 6: Auto Trade (Beta) ─────────────────────────────────────────────────
with tab6:
    st.warning("⚠️ Beta — paper trading only. No real funds are used.")

    # ── Controls ─────────────────────────────────────────────────────────────
    _at_col1, _at_col2, _at_col3 = st.columns(3)
    with _at_col1:
        st.session_state.at_enabled = st.toggle(
            "Enable Auto Trader", value=st.session_state.at_enabled
        )
    with _at_col2:
        _at_threshold = st.slider(
            "Min Confidence (%)", min_value=50, max_value=90,
            value=60, step=1, key="at_threshold"
        )
    with _at_col3:
        _at_max_pct = st.slider(
            "Max % per Trade", min_value=5, max_value=50,
            value=10, step=5, key="at_max_pct"
        )

    # ── Restore balances + trade history from sheet on first load ────────────
    if not st.session_state.at_btc_seeded and st.session_state.at_trade_log == []:
        try:
            _at_ws = get_autotrader_sheet()
            if _at_ws is not None:
                _at_rows = _at_ws.get_all_values()
                if len(_at_rows) > 1:
                    _at_headers = _at_rows[0]
                    _at_data_rows = _at_rows[1:]
                    # Restore balances from last row
                    _at_last = dict(zip(_at_headers, _at_data_rows[-1]))
                    st.session_state.at_btc = float(_at_last.get("BTC_Balance", 0)) or None
                    st.session_state.at_cash = float(_at_last.get("Cash_Balance", 500.0))
                    if st.session_state.at_btc:
                        st.session_state.at_btc_seeded = True
                    # Rebuild at_trade_log from all rows (newest first)
                    _at_dir_map = {"UP": "BUY", "DOWN": "SELL"}
                    for _r in reversed(_at_data_rows):
                        _rd = dict(zip(_at_headers, _r))
                        try:
                            st.session_state.at_trade_log.append({
                                "Trade_Time": _rd.get("Trade_Time", ""),
                                "Direction": _at_dir_map.get(_rd.get("Direction", ""), _rd.get("Direction", "")),
                                "Confidence": float(_rd.get("Confidence", 0)),
                                "Price": float(_rd.get("Price", 0)),
                                "BTC_Change": float(_rd.get("BTC_Change", 0)),
                                "Cash_Change": float(_rd.get("Cash_Change", 0)),
                                "BTC_Balance": float(_rd.get("BTC_Balance", 0)),
                                "Cash_Balance": float(_rd.get("Cash_Balance", 0)),
                                "Portfolio_Value": float(_rd.get("Portfolio_Value", 0)),
                                "Model_Used": _rd.get("Model_Used", ""),
                            })
                        except Exception:
                            pass
        except Exception:
            pass

    # ── Bot logic ─────────────────────────────────────────────────────────────
    if st.session_state.at_enabled:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, key="at_refresh")

        try:
            _at_live = get_live_prediction_data()
            _at_state = _at_live.iloc[-1:]
            _at_cur_price = float(_at_state["Close"].values[0])
            _at_live_price = get_live_ticker_price()
            _at_entry_price = _at_live_price if _at_live_price else _at_cur_price

            # Seed BTC on first trade if not yet done
            if not st.session_state.at_btc_seeded:
                st.session_state.at_btc = 500.0 / _at_entry_price
                st.session_state.at_btc_seeded = True

            _at_price_chg = 0.0

            _AT_FEATURE_COLS = [
                'RSI_14', 'MACD', 'MACD_Signal', 'EMA_9', 'EMA_21',
                'BB_Upper', 'BB_Lower', 'Volume_ROC',
                'price_change_since_window_start', 'price_change_abs'
            ]
            _at_state = _at_state.copy()
            _at_state["price_change_since_window_start"] = _at_price_chg
            _at_state["price_change_abs"] = abs(_at_price_chg)
            _at_state_pred = _at_state[_AT_FEATURE_COLS]

            _at_horizon_model = model[5]
            _at_pred_val = _at_horizon_model.predict(_at_state_pred)[0]
            _at_probs = _at_horizon_model.predict_proba(_at_state_pred)[0]
            _at_direction = "BUY" if _at_pred_val == 1 else "SELL"
            _at_conf = (_at_probs[1] if _at_pred_val == 1 else _at_probs[0]) * 100

            if _at_conf >= _at_threshold:
                _at_factor = (_at_conf / 100.0 - _at_threshold / 100.0) / (1.0 - _at_threshold / 100.0)
                _at_pct = (_at_max_pct / 100.0) * (0.5 + 0.5 * _at_factor)
                _at_trade_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                if _at_direction == "BUY":
                    _at_cash_used = round(st.session_state.at_cash * _at_pct, 2)
                    if _at_cash_used < 1.0:
                        st.caption("No trade: insufficient cash")
                    else:
                        _at_btc_bought = _at_cash_used / _at_entry_price
                        st.session_state.at_cash -= _at_cash_used
                        st.session_state.at_btc += _at_btc_bought
                        _at_btc_change = round(_at_btc_bought, 8)   # positive = bought
                        _at_cash_change = round(-_at_cash_used, 2)  # negative = spent
                        _at_portfolio_value = round(st.session_state.at_cash + st.session_state.at_btc * _at_entry_price, 2)

                        _at_log_entry = {
                            "Trade_Time": _at_trade_time_str,
                            "Direction": _at_direction,
                            "Confidence": round(_at_conf, 1),
                            "Price": round(_at_entry_price, 2),
                            "BTC_Change": _at_btc_change,
                            "Cash_Change": _at_cash_change,
                            "BTC_Balance": round(st.session_state.at_btc, 8),
                            "Cash_Balance": round(st.session_state.at_cash, 2),
                            "Portfolio_Value": _at_portfolio_value,
                            "Model_Used": "5min",
                        }
                        st.session_state.at_trade_log.insert(0, _at_log_entry)

                        try:
                            _at_ws = get_autotrader_sheet()
                            if _at_ws is not None:
                                _at_ws.append_row([
                                    _at_trade_time_str, _at_direction,
                                    round(_at_conf, 1), round(_at_entry_price, 2),
                                    _at_btc_change, _at_cash_change,
                                    round(st.session_state.at_btc, 8),
                                    round(st.session_state.at_cash, 2),
                                    _at_portfolio_value, "5min",
                                ])
                        except Exception:
                            pass

                        st.info(f"BUY {_at_btc_change:.6f} BTC @ ${_at_entry_price:,.2f} | Spent: ${abs(_at_cash_change):.2f} | Conf: {_at_conf:.1f}%")

                else:  # DOWN
                    _at_btc_sold = round(st.session_state.at_btc * _at_pct, 8)
                    if _at_btc_sold * _at_entry_price < 1.0:
                        st.caption("No trade: insufficient BTC")
                    else:
                        _at_cash_received = _at_btc_sold * _at_entry_price
                        st.session_state.at_btc -= _at_btc_sold
                        st.session_state.at_cash += _at_cash_received
                        _at_btc_change = round(-_at_btc_sold, 8)         # negative = sold
                        _at_cash_change = round(_at_cash_received, 2)    # positive = received
                        _at_portfolio_value = round(st.session_state.at_cash + st.session_state.at_btc * _at_entry_price, 2)

                        _at_log_entry = {
                            "Trade_Time": _at_trade_time_str,
                            "Direction": _at_direction,
                            "Confidence": round(_at_conf, 1),
                            "Price": round(_at_entry_price, 2),
                            "BTC_Change": _at_btc_change,
                            "Cash_Change": _at_cash_change,
                            "BTC_Balance": round(st.session_state.at_btc, 8),
                            "Cash_Balance": round(st.session_state.at_cash, 2),
                            "Portfolio_Value": _at_portfolio_value,
                            "Model_Used": "5min",
                        }
                        st.session_state.at_trade_log.insert(0, _at_log_entry)

                        try:
                            _at_ws = get_autotrader_sheet()
                            if _at_ws is not None:
                                _at_ws.append_row([
                                    _at_trade_time_str, _at_direction,
                                    round(_at_conf, 1), round(_at_entry_price, 2),
                                    _at_btc_change, _at_cash_change,
                                    round(st.session_state.at_btc, 8),
                                    round(st.session_state.at_cash, 2),
                                    _at_portfolio_value, "5min",
                                ])
                        except Exception:
                            pass

                        st.info(f"SELL {abs(_at_btc_change):.6f} BTC @ ${_at_entry_price:,.2f} | Received: ${_at_cash_change:.2f} | Conf: {_at_conf:.1f}%")
            else:
                _at_checked_at = datetime.utcnow().strftime("%H:%M:%S UTC")
                st.caption(f"No trade: confidence {_at_conf:.1f}% below threshold {_at_threshold}% — last checked {_at_checked_at}")

        except Exception as _at_trade_err:
            st.warning(f"Auto trader error: {_at_trade_err}")

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    _at_current_price = get_live_ticker_price() or (float(get_live_prediction_data().iloc[-1]["Close"]) if True else 0)
    _at_btc_value = st.session_state.at_btc * _at_current_price if st.session_state.at_btc else 0
    _at_total = st.session_state.at_cash + _at_btc_value

    _at_m1, _at_m2, _at_m3, _at_m4 = st.columns(4)
    _at_m1.metric("Portfolio Value", f"${_at_total:,.2f}", delta=f"${_at_total - 1000.0:+,.2f}")
    _at_m2.metric("Cash", f"${st.session_state.at_cash:,.2f}")
    _at_m3.metric("BTC Held", f"{st.session_state.at_btc:.6f} BTC" if st.session_state.at_btc else "—")
    _at_m4.metric("BTC Value", f"${_at_btc_value:,.2f}")

    # ── Portfolio value charts ─────────────────────────────────────────────────
    if st.session_state.at_trade_log:
        # Time-window toggle
        _at_window = st.radio(
            "Chart window", ["All time", "Last hour"],
            horizontal=True, key="at_window"
        )
        _at_chart_data = list(reversed(st.session_state.at_trade_log))  # chronological
        if _at_window == "Last hour":
            _at_cutoff = (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            _at_chart_data = [t for t in _at_chart_data if t["Trade_Time"] >= _at_cutoff]

        # ── Shared data extraction ────────────────────────────────────────────
        _at_times  = [t["Trade_Time"]      for t in _at_chart_data]
        _at_dirs   = [t["Direction"]       for t in _at_chart_data]
        _at_confs  = [t["Confidence"]      for t in _at_chart_data]
        _at_prices = [t["Price"]           for t in _at_chart_data]
        _at_pvals  = [t["Portfolio_Value"] for t in _at_chart_data]

        # ── Hold BTC benchmark ────────────────────────────────────────────────
        _at_hold_vals = []
        if _at_prices:
            _at_hold_btc = 1000.0 / _at_prices[0]
            _at_hold_vals = [round(_at_hold_btc * p, 2) for p in _at_prices]

        # ── Whaling simulation ────────────────────────────────────────────────
        _at_whale_vals = []
        if _at_prices:
            _wh_btc  = 500.0 / _at_prices[0]
            _wh_cash = 500.0
            for d, p in zip(_at_dirs, _at_prices):
                if d == "BUY" and _wh_cash > 0:
                    _wh_btc  += _wh_cash / p
                    _wh_cash  = 0.0
                elif d == "SELL" and _wh_btc > 0:
                    _wh_cash += _wh_btc * p
                    _wh_btc   = 0.0
                _at_whale_vals.append(round(_wh_cash + _wh_btc * p, 2))

        # ── DCA simulation ────────────────────────────────────────────────────
        _at_dca_vals = []
        _AT_DCA_AMOUNT = 500.0 * 0.025  # $12.50
        if _at_prices:
            _dc_btc  = 500.0 / _at_prices[0]
            _dc_cash = 500.0
            for d, p in zip(_at_dirs, _at_prices):
                if d == "BUY" and _dc_cash >= _AT_DCA_AMOUNT:
                    _dc_btc  += _AT_DCA_AMOUNT / p
                    _dc_cash -= _AT_DCA_AMOUNT
                _at_dca_vals.append(round(_dc_cash + _dc_btc * p, 2))

        # ── Chart helper ──────────────────────────────────────────────────────
        def _make_at_chart(title, pvals, times, dirs, confs, prices, hold_vals):
            colors  = ["#00c896" if d == "BUY" else "#ff4b4b" for d in dirs]
            symbols = ["triangle-up" if d == "BUY" else "triangle-down" for d in dirs]
            hover = [
                f"{times[i]}<br>{dirs[i]} | Conf: {confs[i]:.1f}%<br>"
                f"Price: ${prices[i]:,.2f}<br>Portfolio: ${pvals[i]:,.2f}"
                for i in range(len(pvals))
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=pvals, mode="lines+markers",
                line=dict(color="#7c8cf8", width=2),
                marker=dict(color=colors, symbol=symbols, size=12,
                            line=dict(width=1, color="white")),
                hovertext=hover, hoverinfo="text", name="Portfolio Value",
            ))
            if hold_vals:
                fig.add_trace(go.Scatter(
                    x=times, y=hold_vals, mode="lines",
                    line=dict(color="#f0c040", width=1, dash="dot"),
                    name="Hold BTC", hoverinfo="skip",
                ))
            fig.add_hline(y=1000.0, line_dash="dot", line_color="gray",
                          annotation_text="Start $1,000", annotation_position="bottom right")
            fig.update_layout(
                title=title, xaxis_title="Trade Time (UTC)",
                yaxis_title="Portfolio Value ($)", template="plotly_dark",
                height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
            )
            return fig

        if _at_chart_data:
            # Chart 1: Buy and Sell (actual bot)
            st.plotly_chart(
                _make_at_chart("Buy and Sell", _at_pvals, _at_times, _at_dirs,
                               _at_confs, _at_prices, _at_hold_vals),
                use_container_width=True,
            )
            # Chart 2: Whaling
            st.plotly_chart(
                _make_at_chart("Whaling", _at_whale_vals, _at_times, _at_dirs,
                               _at_confs, _at_prices, _at_hold_vals),
                use_container_width=True,
            )
            # Chart 3: DCA
            st.plotly_chart(
                _make_at_chart("DCA", _at_dca_vals, _at_times, _at_dirs,
                               _at_confs, _at_prices, _at_hold_vals),
                use_container_width=True,
            )

    # ── Trade history ─────────────────────────────────────────────────────────
    if st.session_state.at_trade_log:
        st.markdown("#### Trade History")
        st.dataframe(
            pd.DataFrame(st.session_state.at_trade_log),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No trades yet. Enable the bot and wait for a signal above the confidence threshold.")
