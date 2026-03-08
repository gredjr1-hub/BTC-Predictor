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


@st.cache_data(ttl=30)
def fetch_polymarket_odds(target_time):
    """Fetch UP/DOWN prices for the Polymarket 5-min BTC window closing at target_time.
    Returns {"up": float, "down": float, "slug": str} or None on failure.
    """
    import json
    import requests as _requests
    ts = int(target_time.replace(tzinfo=timezone.utc).timestamp())
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
        for market in markets:
            raw_outcomes = market.get("outcomes", "[]")
            raw_prices = market.get("outcomePrices", "[]")
            if isinstance(raw_outcomes, str):
                outcomes = json.loads(raw_outcomes)
                prices = json.loads(raw_prices)
            else:
                outcomes = raw_outcomes
                prices = raw_prices
            for outcome, price in zip(outcomes, prices):
                key = str(outcome).strip().lower()
                if "up" in key:
                    result["up"] = float(price)
                elif "down" in key:
                    result["down"] = float(price)

        if "up" not in result or "down" not in result:
            return None
        result["slug"] = slug
        return result
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
    df["Price_Delta_From_Window_Start"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)

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


def get_live_ticker_price():
    """Fetch live BTC/USDT price (Kraken). Returns None if fetch fails."""
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


def load_history_from_sheets():
    try:
        client = get_gspread_client()
        sheet = client.open("BTC_AI_Tracker").sheet1
        records = sheet.get_all_records()

        if not records:
            return pd.DataFrame(), sheet

        df = pd.DataFrame(records)
        df["Prediction_Time"] = pd.to_datetime(df["Prediction_Time"])
        df["Target_Time"] = pd.to_datetime(df["Target_Time"])
        # Gracefully handle sheets that predate the Polymarket_Odds column
        if "Polymarket_Odds" not in df.columns:
            df["Polymarket_Odds"] = np.nan
        else:
            df["Polymarket_Odds"] = pd.to_numeric(df["Polymarket_Odds"], errors="coerce")
        # Gracefully handle sheets that predate the Window_Start_Price column
        if "Window_Start_Price" not in df.columns:
            df["Window_Start_Price"] = np.nan
        else:
            df["Window_Start_Price"] = pd.to_numeric(df["Window_Start_Price"], errors="coerce")
        return df, sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None


def resolve_pending_trades_in_sheets(live_data, history_df, sheet):
    if history_df.empty or sheet is None:
        return history_df

    pending_mask = history_df["Outcome"] == "Pending"
    latest_live_time = live_data.index[-1]

    for idx, row in history_df[pending_mask].iterrows():
        target_time = row["Target_Time"]

        if target_time <= latest_live_time:
            valid_candles = live_data[live_data.index >= target_time]

            if not valid_candles.empty:
                actual_close = valid_candles["Close"].iloc[0]
                prediction = row["Prediction"]

                # Use window start price if available; fall back to entry price for legacy rows
                ref_price_raw = row.get("Window_Start_Price")
                ref_price = (
                    float(ref_price_raw)
                    if pd.notna(ref_price_raw) and float(ref_price_raw) > 0
                    else float(row["Entry_Price"])
                )

                outcome = (
                    "Win"
                    if (prediction == "UP" and actual_close > ref_price)
                    or (prediction == "DOWN" and actual_close < ref_price)
                    else "Loss"
                )

                history_df.at[idx, "Close_Price"] = round(actual_close, 2)
                history_df.at[idx, "Outcome"] = outcome

                sheet.update_cell(idx + 2, 6, round(actual_close, 2))
                sheet.update_cell(idx + 2, 7, outcome)

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
                        entry = float(row["Entry_Price"])
                    except Exception:
                        continue

                    direction = row.get("Prediction", "")
                    diff = float(live_price) - entry if direction == "UP" else entry - float(live_price)
                    status_color = "green" if diff > 0 else "red"
                    status_icon = "🟢 Profit" if diff > 0 else "🔴 Drawdown"
                    st.markdown(
                        f"- {direction} from **${entry:,.2f}** | :{status_color}[{status_icon} (${abs(diff):,.2f})]"
                    )
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

    # Session state for auto-prediction deduplication
    if "last_auto_target" not in st.session_state:
        st.session_state.last_auto_target = None

    # Check whether auto-mode should fire this rerun
    now_utc = datetime.utcnow().replace(microsecond=0)
    auto_trigger = (
        prediction_mode == "Auto"
        and auto_pilot
        and is_at_polymarket_boundary(now_utc)
        and st.session_state.last_auto_target != snap_to_polymarket_window(now_utc)
    )

    if auto_trigger:
        st.info("Auto-prediction firing at Polymarket window boundary...")

    if st.button("Generate Live Prediction", type="primary") or auto_trigger:
        with st.spinner("Fetching live data & consulting AI..."):
            live_data = get_live_prediction_data()
            history_df, sheet = load_history_from_sheets()

            history_df = resolve_pending_trades_in_sheets(live_data, history_df, sheet)

            current_state = live_data.iloc[-1:]
            current_price = float(current_state["Close"].values[0])
            current_time = current_state.index[0]  # UTC-naive candle timestamp

            target_time = snap_to_polymarket_window(current_time)

            # Window start = 5 minutes before target_time
            window_start_time = target_time - timedelta(minutes=5)

            # Look up start-of-window price from live_data (already fetched)
            window_start_candles = live_data[live_data.index == window_start_time]
            if not window_start_candles.empty:
                window_start_price = float(window_start_candles["Close"].values[0])
            else:
                # Fallback: use current price (same behavior as before)
                window_start_price = current_price

            # Fetch Polymarket odds for this window (non-blocking — fails gracefully)
            fetch_polymarket_odds.clear()          # bypass cache — always fetch fresh at prediction time
            pm_odds = fetch_polymarket_odds(target_time)

            if not history_df.empty and (history_df["Target_Time"] == target_time).any():
                st.warning(
                    f"Prediction already generated for window {fmt_et(target_time, '%H:%M %Z')}. "
                    "Wait for the next 5-minute window."
                )
            else:
                # Align to model's exact feature set (guards against version mismatches)
                _model_cols = list(model.feature_names_in_)
                current_state_pred = current_state[_model_cols]
                prediction_val = model.predict(current_state_pred)[0]
                probabilities = model.predict_proba(current_state_pred)[0]

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

                st.markdown(f"**Signal Strength:** :{color}[{signal_strength}]")

                # Window countdown
                _now = datetime.utcnow()
                _seconds_left = max(0, int((target_time - _now).total_seconds()))
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

                if sheet:
                    # Odds for the predicted direction (stored as col 8 for P&L calc)
                    odds_val = round(pm_odds[direction.lower()], 4) if pm_odds else ""
                    new_row = [
                        str(current_time),
                        current_price,          # Entry_Price (when bet was placed)
                        direction,
                        round(confidence_pct, 2),
                        str(target_time),
                        "",                     # Close_Price (filled by resolver)
                        "Pending",              # Outcome (filled by resolver)
                        odds_val,               # Polymarket_Odds
                        window_start_price,     # Window_Start_Price (col 9)
                    ]
                    sheet.append_row(new_row)
                    st.success("✅ Prediction successfully logged to Google Sheets!")

                st.session_state.last_auto_target = target_time

with tab2:
    st.markdown("### Model Performance Analytics")

    history, _ = load_history_from_sheets()

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
            _diag_ts = int(current_window_target.replace(tzinfo=timezone.utc).timestamp())
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
            st.caption(
                f"Market: `{live_odds['slug']}` · Window closes at "
                f"{fmt_et(current_window_target, '%H:%M %Z')} · Auto-refreshes every 30s · use button above for immediate refresh"
            )
        else:
            st.warning(
                "Could not fetch live odds from Polymarket. "
                "The market for this window may not be open yet, or the API is temporarily unavailable."
            )

        polymarket_url = get_polymarket_url()
        if polymarket_url and not polymarket_url.startswith("https://polymarket.com/event/PLACEHOLDER"):
            st.link_button("Open on Polymarket ↗", polymarket_url)

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

    if sim_history.empty:
        st.info("No predictions found in the Google Sheet yet. Run predictions to start tracking!")
    else:
        completed_trades_sim = sim_history[sim_history["Outcome"].isin(["Win", "Loss"])].copy()

        # Only trades that have valid Polymarket odds recorded
        completed_with_odds = completed_trades_sim[
            completed_trades_sim["Polymarket_Odds"].notna()
            & (completed_trades_sim["Polymarket_Odds"] > 0)
        ].sort_values("Prediction_Time").reset_index(drop=True)

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
                    # Skip unless AI is also highly confident
                    if conf < MIN_CONF_FOR_HIGH_ODDS:
                        trades_log.append({
                            "Time": fmt_et(row["Prediction_Time"], "%m/%d %H:%M %Z"),
                            "Direction": row["Prediction"],
                            "Confidence": f"{conf:.1f}%",
                            "Odds": f"{odds:.3f} ({odds*100:.1f}%)",
                            "Bet %": "—",
                            "Bet $": 0,
                            "Outcome": "Skipped (low conf / high odds)",
                            "P&L": 0,
                            "Balance": round(balance, 2),
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
                    "Balance": round(balance, 2),
                })

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
            sim_df = pd.DataFrame(trades_log)
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
                return ""

            st.dataframe(
                sim_df.style.map(highlight_sim_outcome, subset=["Outcome"]),
                use_container_width=True,
            )

            skipped_count = sum(1 for t in trades_log if "Skipped" in str(t.get("Outcome", "")))
            if skipped_count:
                st.caption(f"{skipped_count} bet(s) skipped — high-odds market with insufficient AI confidence.")

            if excluded_count > 0:
                st.info(
                    f"Note: {excluded_count} completed prediction(s) excluded from simulation "
                    "— Polymarket odds were not recorded at prediction time (pre-dates this feature)."
                )
