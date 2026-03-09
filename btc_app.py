import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import os
import joblib
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


def fetch_polymarket_resolution(target_time) -> str | None:
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
        if not event.get("closed"):
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
    "Polymarket_Odds", "PM_Resolution", "Seconds_Left", "Model",
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


def load_history_from_sheets():
    try:
        client = get_gspread_client()
        sheet = client.open("BTC_AI_Tracker").sheet1
        _ensure_sheet_headers(sheet)
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
        for _col in ["Model", "PM_Resolution"]:
            if _col not in df.columns:
                df[_col] = ""

        return df, sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None


def resolve_pending_trades_in_sheets(live_data, history_df, sheet):
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
            close_price = fetch_pyth_price_at(target_time)

            if close_price is None:
                # Last resort: Kraken candle at or after target
                valid_candles = live_data[live_data.index >= target_time]
                if valid_candles.empty:
                    continue   # can't resolve yet — leave Pending
                close_price = float(valid_candles["Close"].iloc[0])

            # NaN-safe reference price: window start > entry
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


@_fragment(run_every="2s")
def live_market_and_advanced_stats_fragment(
    pending_trades,
    overall_wr,
    p90_wr,
    p90_threshold,
    up_wr,
    down_wr,
    avg_conf_win,
    avg_conf_loss,
):
    """Only this section reruns every 2 seconds while the session is active."""
    st.markdown("#### ⚡ Live Market & Advanced Stats")
    st.caption("Auto-updating live price every 2 seconds (fragment rerun).")

    # Store the last successful live price so the UI stays stable through transient API hiccups.
    if "last_live_price" not in st.session_state:
        st.session_state.last_live_price = None
        st.session_state.last_live_price_ts = None

    live_price = get_live_ticker_price()
    if live_price is not None:
        st.session_state.last_live_price = float(live_price)
        st.session_state.last_live_price_ts = datetime.utcnow()
    else:
        live_price = st.session_state.last_live_price

    col_live, col_stats1, col_stats2 = st.columns([1.5, 1, 1])

    with col_live:
        if live_price is not None:
            st.metric("Live BTC/USDT Price", f"${float(live_price):,.2f}")

            ts = st.session_state.get("last_live_price_ts")
            if ts:
                st.caption(f"Last tick: {fmt_et(ts, '%H:%M:%S %Z')} • Source: Kraken (CCXT)")
            else:
                st.caption("Source: Kraken (CCXT)")

            if pending_trades is not None and not pending_trades.empty:
                st.markdown("**Active Open Bets:**")
                for _, row in pending_trades.iterrows():
                    try:
                        ref_raw = row.get("Window_Start_Price")
                        ref = (
                            float(ref_raw)
                            if pd.notna(ref_raw) and float(ref_raw) > 0
                            else float(row["Entry_Price"])
                        )
                    except Exception:
                        continue

                    direction = row.get("Prediction", "")
                    diff = float(live_price) - ref if direction == "UP" else ref - float(live_price)
                    status_color = "green" if diff > 0 else "red"
                    status_icon = "🟢 Profit" if diff > 0 else "🔴 Loss"
                    target_str = fmt_et(row.get("Target_Time"), "%H:%M %Z") or "N/A"
                    bc1, bc2, bc3 = st.columns([1, 2, 2])
                    bc1.markdown(f"**{direction}**")
                    bc2.markdown(f"Open @ **${ref:,.2f}** → closes {target_str}")
                    bc3.markdown(f":{status_color}[{status_icon} (${abs(diff):,.2f})]")
            else:
                st.markdown("*No active bets currently open.*")
        else:
            st.warning("Could not fetch live price (Kraken).")

    with col_stats1:
        st.markdown("**Core Win Rates:**")
        st.write(f"- Overall Win Rate: **{overall_wr:.1f}%**")

        if p90_wr is not None and p90_threshold is not None:
            st.write(f"- Top 10% Conf Win Rate: **{p90_wr:.1f}%** *(>={p90_threshold:.1f}% Conf)*")
        else:
            st.write("- Top 10% Conf Win Rate: **N/A**")

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


# --- 7. UI Layout (Tabs) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Predictor",
    "📊 Analytics & 24h Visualizer",
    "🎯 Polymarket Odds",
    "💰 P&L Simulator",
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

            history_df = resolve_pending_trades_in_sheets(live_data, history_df, sheet)

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

                # Seconds-precise model selection
                _now = datetime.utcnow()
                _seconds_left = max(0, int((target_time - _now).total_seconds()))

                if _seconds_left < 90:          # < 1.5 min → use 1-min model
                    _minutes_to_end = 1
                else:
                    _now_minute = current_time.minute % 5
                    _minutes_to_end = max(1, 5 - int(_now_minute))

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

                # Polymarket odds row
                if pm_odds:
                    st.markdown("**Polymarket Implied Odds (current window):**")
                    pm_col_up, pm_col_down = st.columns(2)
                    pm_col_up.metric(
                        "Polymarket UP",
                        f"{pm_odds['up'] * 100:.1f}%",
                        delta=f"Payout {1 / pm_odds['up']:.2f}x",
                    )
                    pm_col_down.metric(
                        "Polymarket DOWN",
                        f"{pm_odds['down'] * 100:.1f}%",
                        delta=f"Payout {1 / pm_odds['down']:.2f}x",
                    )
                else:
                    st.caption("Polymarket odds unavailable for this window.")

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
                        ]
                        sheet.append_row(new_row)
                        _fetch_sheet_records.clear()   # invalidate read cache so other tabs see new row
                        st.success("✅ Prediction successfully logged to Google Sheets!")

                    st.session_state.last_auto_target = target_time
                    st.session_state.last_pred_target = target_time
                    st.session_state.last_pred_candle_ts = current_time
                    st.session_state.last_pred_wall_time = datetime.utcnow()
                    if pm_odds:
                        st.session_state.last_pred_odds_up = pm_odds["up"]

with tab2:
    st.markdown("### Model Performance Analytics")

    history, _sheet2 = load_history_from_sheets()

    if st.button("🔄 Resolve Pending Trades", help="Check Polymarket and Pyth for outcomes of all pending windows"):
        with st.spinner("Resolving pending trades…"):
            _live2 = get_live_prediction_data()
            history = resolve_pending_trades_in_sheets(_live2, history, _sheet2)
        st.success("Done — pending trades resolved where data was available.")
        st.rerun()

    _exclude_pre_odds = st.toggle(
        "Exclude pre-Polymarket rows from stats & log",
        value=True,
        help="Hides rows recorded before Polymarket odds were tracked. Does not delete data from the sheet.",
    )
    if _exclude_pre_odds and not history.empty and "Polymarket_Odds" in history.columns:
        history = history[history["Polymarket_Odds"].notna() & (history["Polymarket_Odds"] > 0)]

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

        # --- Precompute stats once (fragment uses these values while only the price updates) ---
        pending_trades = history[history["Outcome"] == "Pending"]

        wins = len(completed_trades[completed_trades["Outcome"] == "Win"])
        overall_wr = (wins / total_completed * 100) if total_completed > 0 else 0.0

        p90_threshold = None
        p90_wr = None
        if total_completed > 0 and "Confidence" in completed_trades.columns:
            try:
                p90_threshold = float(completed_trades["Confidence"].quantile(0.90))
                p90_trades = completed_trades[completed_trades["Confidence"] >= p90_threshold]
                if len(p90_trades) > 0:
                    p90_wins = len(p90_trades[p90_trades["Outcome"] == "Win"])
                    p90_wr = (p90_wins / len(p90_trades) * 100)
            except Exception:
                p90_threshold, p90_wr = None, None

        up_wr = None
        down_wr = None
        avg_conf_win = None
        avg_conf_loss = None
        if total_completed > 0:
            up_trades = completed_trades[completed_trades["Prediction"] == "UP"]
            down_trades = completed_trades[completed_trades["Prediction"] == "DOWN"]

            up_wr = (
                (len(up_trades[up_trades["Outcome"] == "Win"]) / len(up_trades) * 100) if len(up_trades) > 0 else 0.0
            )
            down_wr = (
                (len(down_trades[down_trades["Outcome"] == "Win"]) / len(down_trades) * 100)
                if len(down_trades) > 0
                else 0.0
            )

            if "Confidence" in completed_trades.columns:
                avg_conf_win = completed_trades[completed_trades["Outcome"] == "Win"]["Confidence"].mean()
                avg_conf_loss = completed_trades[completed_trades["Outcome"] == "Loss"]["Confidence"].mean()
                avg_conf_win = float(avg_conf_win) if pd.notna(avg_conf_win) else 0.0
                avg_conf_loss = float(avg_conf_loss) if pd.notna(avg_conf_loss) else 0.0

        # --- Live Market & Advanced Stats (auto-refreshes every 2s without full page rerun) ---
        live_market_and_advanced_stats_fragment(
            pending_trades=pending_trades,
            overall_wr=overall_wr,
            p90_wr=p90_wr,
            p90_threshold=p90_threshold,
            up_wr=up_wr,
            down_wr=down_wr,
            avg_conf_win=avg_conf_win,
            avg_conf_loss=avg_conf_loss,
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
                _model_rows.append({
                    "Model": _m,
                    "Trades": _m_total,
                    "Wins": _m_wins,
                    "Win Rate": f"{_m_wr:.1f}%",
                    "Avg Confidence": f"{_m_avg_conf:.1f}%" if _m_avg_conf else "—",
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

with tab3:
    st.markdown("### 🎯 Polymarket Live Odds")
    st.markdown("Live market odds fetched via the Polymarket API for the current 5-minute BTC window.")

    col_pred, col_odds = st.columns([1, 2])

    with col_pred:
        st.markdown("#### Last AI Prediction")
        last_pred_history, _ = load_history_from_sheets()
        if not last_pred_history.empty:
            last = last_pred_history.iloc[-1]
            direction_icon = "⬆️" if last.get("Prediction") == "UP" else "⬇️"
            st.metric("Direction", f"{direction_icon} {last.get('Prediction', 'N/A')}")
            st.metric("AI Confidence", f"{last.get('Confidence', 'N/A')}%")
            target_display = fmt_et(last.get("Target_Time"), "%H:%M %Z") or str(last.get("Target_Time", "N/A"))
            st.metric("Target Window", target_display)
            outcome = last.get("Outcome", "Pending")
            outcome_color = "green" if outcome == "Win" else "red" if outcome == "Loss" else "orange"
            st.markdown(f"**Outcome:** :{outcome_color}[{outcome}]")
            last_odds = last.get("Polymarket_Odds")
            if pd.notna(last_odds) and float(last_odds) > 0:
                st.metric("Odds at Prediction", f"{float(last_odds)*100:.1f}%",
                          delta=f"Payout {1/float(last_odds):.2f}x")
        else:
            st.info("No predictions logged yet.")

    with col_odds:
        st.markdown("#### Live Odds — Current Window")
        if st.button("🔄 Refresh Odds", key="refresh_polymarket_odds"):
            fetch_polymarket_odds.clear()
            st.rerun()
        current_window_target = snap_to_polymarket_window(datetime.utcnow().replace(microsecond=0))

        with st.expander("🔍 Odds fetch diagnostics", expanded=False):
            _diag_ts = int(current_window_target.replace(tzinfo=timezone.utc).timestamp()) - 300  # match fetch_polymarket_odds
            _diag_slug = f"btc-updown-5m-{_diag_ts}"
            _diag_url = f"https://gamma-api.polymarket.com/events?slug={_diag_slug}"
            st.write(f"**Window target (UTC):** `{current_window_target}`")
            st.write(f"**Slug tried:** `{_diag_slug}`")
            st.write(f"**API URL:** `{_diag_url}`")
            try:
                import requests as _req
                _raw = _req.get(_diag_url, timeout=5).json()
                st.json(_raw)
            except Exception as _e:
                st.error(f"Request failed: {_e}")

        live_odds = fetch_polymarket_odds(current_window_target)

        if live_odds:
            odds_col_up, odds_col_down = st.columns(2)
            odds_col_up.metric(
                "UP probability",
                f"{live_odds['up'] * 100:.1f}%",
                delta=f"Implied payout {1 / live_odds['up']:.2f}x",
            )
            odds_col_down.metric(
                "DOWN probability",
                f"{live_odds['down'] * 100:.1f}%",
                delta=f"Implied payout {1 / live_odds['down']:.2f}x",
            )
            _src_label = "🟢 CLOB (real-time)" if live_odds.get("source") == "clob" else "🟡 Gamma (cached)"
            st.caption(
                f"Market: `{live_odds['slug']}` · Source: {_src_label} · Window closes at "
                f"{fmt_et(current_window_target, '%H:%M %Z')} · Auto-refreshes every 10s · use button above for immediate refresh"
            )
        else:
            st.warning(
                "Could not fetch live odds from Polymarket. "
                "The market for this window may not be open yet, or the API is temporarily unavailable."
            )

        polymarket_url = get_polymarket_url()
        if polymarket_url and not polymarket_url.startswith("https://polymarket.com/event/PLACEHOLDER"):
            st.link_button("Open on Polymarket ↗", polymarket_url)

        st.markdown("---")
        st.markdown("#### 📊 Polymarket Odds — Current Window")

        # Accumulate odds readings for the current window in session_state
        if "pm_odds_history" not in st.session_state:
            st.session_state.pm_odds_history = []
        if "pm_odds_window" not in st.session_state:
            st.session_state.pm_odds_window = None

        # Reset history when the window rolls over
        if st.session_state.pm_odds_window != current_window_target:
            st.session_state.pm_odds_history = []
            st.session_state.pm_odds_window = current_window_target

        if live_odds:
            st.session_state.pm_odds_history.append({
                "ts": datetime.utcnow(),
                "up": live_odds["up"] * 100,
                "down": live_odds["down"] * 100,
            })

        if len(st.session_state.pm_odds_history) >= 2:
            import plotly.graph_objects as go
            _hist = st.session_state.pm_odds_history
            _times = [h["ts"] for h in _hist]
            _ups = [h["up"] for h in _hist]
            _downs = [h["down"] for h in _hist]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=_times, y=_ups, mode="lines+markers",
                                     name="UP %", line=dict(color="lime", width=2)))
            fig.add_trace(go.Scatter(x=_times, y=_downs, mode="lines+markers",
                                     name="DOWN %", line=dict(color="tomato", width=2)))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            _chart_title = ""
            if live_odds and live_odds.get("price_to_beat"):
                _chart_title = f"Strike price: ${live_odds['price_to_beat']:,.2f}"
            fig.update_layout(
                title=_chart_title,
                yaxis=dict(title="Probability %", range=[0, 100]),
                xaxis_title="Time (UTC)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif live_odds:
            st.caption("Collecting readings… chart appears after the second tick (~10s).")
        else:
            st.caption("No odds data available for chart.")

        # Show price_to_beat as a prominent metric if parseable
        if live_odds and live_odds.get("price_to_beat"):
            st.metric(
                "🎯 Polymarket Strike Price",
                f"${live_odds['price_to_beat']:,.2f}",
                help="BTC price to beat — parsed from the Polymarket market question for this window.",
            )

with tab4:
    st.markdown("### 💰 P&L Simulator")
    st.markdown(
        "Simulates portfolio growth starting from **$1,000** using your logged predictions, "
        "Polymarket odds at bet time, and actual outcomes. "
        "Bet sizing scales with confidence and market odds — contrarian bets (market <50%) scale 2.5–5%, "
        "high-odds markets (≥65%) require ≥60% confidence, standard zone is flat 2.5%."
    )

    # Load history (may already be loaded in tab2 context, but tabs are independent blocks)
    sim_history, _ = load_history_from_sheets()

    _apply_skip_rules = st.toggle(
        "Apply skip rules (high-odds / low-confidence filter)",
        value=True,
        help="When ON, bets where market odds ≥65% but AI confidence <60% are skipped. "
             "Toggle OFF to see total P&L as if every prediction had been bet at 2.5%.",
    )
    _pl_col1, _pl_col2 = st.columns(2)
    with _pl_col1:
        _pl_available_models = (
            ["All"] + sorted(sim_history["Model"].dropna().unique().tolist())
            if "Model" in sim_history.columns and not sim_history.empty
            else ["All"]
        )
        _pl_model_filter = st.selectbox(
            "Filter by model horizon",
            options=_pl_available_models,
            index=0,
            help="Only include trades that used a specific model horizon. "
                 "Useful for excluding late-window 1-min or 2-min predictions.",
        )
    with _pl_col2:
        _pl_min_conf = st.number_input(
            "Min Confidence (%)",
            min_value=50.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Exclude trades where AI confidence was below this threshold.",
        )

    if sim_history.empty:
        st.info("No predictions found in the Google Sheet yet. Run predictions to start tracking!")
    else:
        completed_trades_sim = sim_history[sim_history["Outcome"].isin(["Win", "Loss"])].copy()

        # Only trades that have valid Polymarket odds recorded
        completed_with_odds = completed_trades_sim[
            completed_trades_sim["Polymarket_Odds"].notna()
            & (completed_trades_sim["Polymarket_Odds"] > 0)
        ].sort_values("Prediction_Time").reset_index(drop=True)

        # Apply model and confidence filters
        if _pl_model_filter != "All" and "Model" in completed_with_odds.columns:
            completed_with_odds = completed_with_odds[completed_with_odds["Model"] == _pl_model_filter]
        if _pl_min_conf > 50.0 and "Confidence" in completed_with_odds.columns:
            completed_with_odds = completed_with_odds[
                pd.to_numeric(completed_with_odds["Confidence"], errors="coerce") >= _pl_min_conf
            ]

        excluded_count = len(completed_trades_sim) - len(completed_with_odds)

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

            balance = STARTING_BALANCE
            trades_log = []

            for _, row in completed_with_odds.iterrows():
                conf = float(row["Confidence"])
                odds = float(row["Polymarket_Odds"])

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

                elif odds < 0.5:
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
