import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# ------------------------------------------------------------
# Timezone (Eastern Time / America/New_York)
# NOTE: This is "Eastern Time" and will show EST/EDT automatically.
# ------------------------------------------------------------
ET_TZ = ZoneInfo("America/New_York")


def dt_to_sheet_str(dt: datetime) -> str:
    """Stable, parse-friendly timestamp string with offset, to-the-second."""
    dt = dt.astimezone(ET_TZ).replace(microsecond=0)
    # Example: 2026-03-05 05:38:50-0500
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")


# --- 1. Page Setup & Live Sync ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin AI Trading Terminal")

st.sidebar.markdown("### ⚙️ Terminal Settings")
auto_pilot = st.sidebar.toggle("Enable Live Sync (5m interval)")

if auto_pilot:
    # Full-app rerun every 5 minutes (kept for your existing "Live Sync" behavior)
    st_autorefresh(interval=300000, key="data_refresh")
    st.sidebar.success("Live Sync Active: Fetching latest data every 5 minutes.")


# --- 2. Exchange (cached) ---
@st.cache_resource
def get_exchange():
    """Create a single CCXT exchange client (shared across reruns)."""
    return ccxt.kraken({"enableRateLimit": True})


# --- 3. Load the Brain ---
@st.cache_resource
def load_model():
    return joblib.load("btc_5m_rf_model.joblib")


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Error: {e}")
    st.stop()


# --- 4. Data Fetchers (KRAKEN) ---
def get_live_prediction_data():
    exchange = get_exchange()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1m", limit=100)

    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    # CCXT timestamps are UTC ms; convert to tz-aware ET for consistent display/logic
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(ET_TZ)
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
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(ET_TZ)
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


def _parse_time_to_et(series: pd.Series) -> pd.Series:
    """
    Parse times from Sheets into tz-aware ET timestamps.

    Robust strategy:
    - Always parse with utc=True (handles values with offsets and tz-naive values)
    - Convert parsed values to ET

    This keeps older tz-naive logs (which were effectively UTC in the original app) correct,
    and also correctly parses new values written with explicit -0500/-0400 offsets.
    """
    if series is None:
        return pd.Series(dtype="datetime64[ns, America/New_York]")

    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_convert(ET_TZ)


def _format_times_for_table(series: pd.Series) -> pd.Series:
    """Format timestamps for display (prevents Streamlit/Styler showing blanks/None)."""
    # Keep a raw string fallback in case pandas can't parse a particular value.
    raw_str = series.astype(str).replace({"NaT": "", "None": ""})

    s = _parse_time_to_et(series)
    formatted = s.dt.strftime("%Y-%m-%d %H:%M:%S %Z").fillna("")
    # If parsing failed (blank formatted), fall back to the raw value.
    return formatted.where(formatted != "", raw_str)


def load_history_from_sheets():
    try:
        client = get_gspread_client()
        sheet = client.open("BTC_AI_Tracker").sheet1
        records = sheet.get_all_records()

        if not records:
            return pd.DataFrame(), sheet

        df = pd.DataFrame(records)

        if "Prediction_Time" in df.columns:
            df["Prediction_Time"] = _parse_time_to_et(df["Prediction_Time"])
        if "Target_Time" in df.columns:
            df["Target_Time"] = _parse_time_to_et(df["Target_Time"])

        return df, sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None


def resolve_pending_trades_in_sheets(live_data, history_df, sheet):
    if history_df.empty or sheet is None:
        return history_df

    if "Outcome" not in history_df.columns or "Target_Time" not in history_df.columns:
        return history_df

    pending_mask = history_df["Outcome"] == "Pending"
    latest_live_time = live_data.index[-1]  # tz-aware ET

    for idx, row in history_df[pending_mask].iterrows():
        target_time = row.get("Target_Time", pd.NaT)
        if pd.isna(target_time):
            continue

        if target_time <= latest_live_time:
            valid_candles = live_data[live_data.index >= target_time]

            if not valid_candles.empty:
                actual_close = float(valid_candles["Close"].iloc[0])
                entry_price = float(row["Entry_Price"])
                prediction = row["Prediction"]

                outcome = (
                    "Win"
                    if (prediction == "UP" and actual_close > entry_price)
                    or (prediction == "DOWN" and actual_close < entry_price)
                    else "Loss"
                )

                history_df.at[idx, "Close_Price"] = round(actual_close, 2)
                history_df.at[idx, "Outcome"] = outcome

                # Sheet row index offset (+2) = header row + 0-index -> 1-index
                sheet.update_cell(idx + 2, 6, round(actual_close, 2))
                sheet.update_cell(idx + 2, 7, outcome)

    return history_df


# --- 6. Live Market Fragment (2s auto-refresh, no full app rerun) ---
# Streamlit renamed experimental_fragment -> fragment. Support both.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

if _fragment is None:
    # Fallback: app will still run, but the live ticker won't auto-refresh without a full rerun.
    def _fragment(*args, **kwargs):
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
        st.session_state.last_live_price_ts = datetime.now(ET_TZ).replace(microsecond=0)
    else:
        live_price = st.session_state.last_live_price

    col_live, col_stats1, col_stats2 = st.columns([1.5, 1, 1])

    with col_live:
        if live_price is not None:
            st.metric("Live BTC/USDT Price", f"${float(live_price):,.2f}")

            ts = st.session_state.get("last_live_price_ts")
            if ts:
                st.caption(f"Last tick: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')} • Source: Kraken (CCXT)")
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
tab1, tab2 = st.tabs(["🔮 Live Predictor", "📊 Analytics & 24h Visualizer"])

with tab1:
    st.markdown("### Generate Next Move")
    if st.button("Generate Live Prediction", type="primary"):
        with st.spinner("Fetching live data & consulting AI..."):
            live_data = get_live_prediction_data()
            history_df, sheet = load_history_from_sheets()

            history_df = resolve_pending_trades_in_sheets(live_data, history_df, sheet)

            current_state = live_data.iloc[-1:]
            current_price = float(current_state["Close"].values[0])
            candle_time = current_state.index[0]  # tz-aware ET, minute candle timestamp

            prediction_time = datetime.now(ET_TZ).replace(microsecond=0)
            target_time = prediction_time + timedelta(minutes=5)

            # Prevent duplicate predictions for the same latest 1-minute candle
            if not history_df.empty and "Prediction_Time" in history_df.columns:
                last_pred_time = history_df["Prediction_Time"].iloc[-1]
                try:
                    last_pred_floor = pd.Timestamp(last_pred_time).floor("min")
                except Exception:
                    last_pred_floor = None

                if last_pred_floor is not None and last_pred_floor == pd.Timestamp(candle_time).floor("min"):
                    st.warning("Prediction already generated for the latest 1-minute candle. Wait for the next candle.")
                    st.stop()

            prediction_val = model.predict(current_state)[0]
            probabilities = model.predict_proba(current_state)[0]

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

            st.markdown(f"### Prediction Time: {prediction_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            st.markdown(f"### Target Time: {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current BTC Price", f"${current_price:,.2f}")
            col2.metric("AI Prediction", f"{direction}", delta="Long" if direction == "UP" else "-Short")
            col3.metric("AI Confidence", f"{confidence_pct:.1f}%")

            st.markdown(f"**Signal Strength:** :{color}[{signal_strength}]")

            if sheet:
                new_row = [
                    dt_to_sheet_str(prediction_time),
                    current_price,
                    direction,
                    round(confidence_pct, 2),
                    dt_to_sheet_str(target_time),
                    "",
                    "Pending",
                ]
                sheet.append_row(new_row, value_input_option="RAW")
                st.success("✅ Prediction successfully logged to Google Sheets!")

with tab2:
    st.markdown("### Model Performance Analytics")

    history, _ = load_history_from_sheets()

    if not history.empty:
        completed_trades = history[history["Outcome"].isin(["Win", "Loss"])]
        total_completed = len(completed_trades)

        # --- Thermal Streaks ---
        def calculate_streak(history_df, hours):
            cutoff = datetime.now(ET_TZ) - timedelta(hours=hours)
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
        st.markdown("#### 24-Hour Trade Overlay")
        with st.spinner("Rendering 24-hour chart..."):
            chart_data = get_24h_chart_data()

            # Plotly is happiest with tz-naive datetimes for axes; keep ET wall-clock.
            chart_x = chart_data.index.tz_localize(None)

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=chart_x,
                        open=chart_data["Open"],
                        high=chart_data["High"],
                        low=chart_data["Low"],
                        close=chart_data["Close"],
                        name="BTC/USDT",
                    )
                ]
            )

            recent_history = history[history["Prediction_Time"] >= chart_data.index[0]]

            for _, row in recent_history.iterrows():
                symbol = "triangle-up" if row["Prediction"] == "UP" else "triangle-down"
                color = "green" if row["Outcome"] == "Win" else "red" if row["Outcome"] == "Loss" else "yellow"
                size = 14 if row["Outcome"] != "Pending" else 10

                x_ts = row.get("Prediction_Time")
                if pd.notna(x_ts) and hasattr(x_ts, "tzinfo") and x_ts.tzinfo is not None:
                    x_ts = x_ts.tz_convert(ET_TZ).tz_localize(None)

                fig.add_trace(
                    go.Scatter(
                        x=[x_ts],
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
                (len(up_trades[up_trades["Outcome"] == "Win"]) / len(up_trades) * 100)
                if len(up_trades) > 0
                else 0.0
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

        # Streamlit + Styler can be flaky with tz-aware datetimes in a dataframe.
        # For display, convert the datetime columns to ET strings (seconds + EST/EDT).
        display_history = history.copy()
        if "Prediction_Time" in display_history.columns:
            display_history["Prediction_Time"] = _format_times_for_table(display_history["Prediction_Time"])
        if "Target_Time" in display_history.columns:
            display_history["Target_Time"] = _format_times_for_table(display_history["Target_Time"])

        st.dataframe(
            display_history.iloc[::-1].style.map(highlight_outcome, subset=["Outcome"]),
            use_container_width=True,
        )

    else:
        st.info("No predictions found in the Google Sheet yet. Run a prediction to start tracking!")
