 import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="TradePilot V2",
    page_icon="📱",
    layout="centered"
)

st.markdown("""
<style>
.block-container {
    max-width: 760px;
    padding: 1rem .8rem 3rem;
}
h1 {font-size: 1.8rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
button {min-height: 44px;}
</style>
""", unsafe_allow_html=True)

st.title("📱 TradePilot V2")
st.caption("Mobile stock research & paper-trading dashboard")

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]


@st.cache_data(ttl=60)
def load_data(ticker, period):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


def calculate_indicators(data, fast, slow, rsi_len, atr_len):
    x = data.copy()

    x["Fast SMA"] = x["Close"].rolling(fast).mean()
    x["Slow SMA"] = x["Close"].rolling(slow).mean()

    delta = x["Close"].diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / rsi_len,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / rsi_len,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))

    x["EMA12"] = x["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    x["EMA26"] = x["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    x["MACD"] = x["EMA12"] - x["EMA26"]

    x["MACD Signal"] = x["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    previous_close = x["Close"].shift(1)

    true_range = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - previous_close).abs(),
            (x["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = true_range.rolling(atr_len).mean()

    x["BUY"] = (
        (x["Fast SMA"] > x["Slow SMA"]) &
        (x["RSI"] > 50) &
        (x["MACD"] > x["MACD Signal"])
    )

    x["SELL"] = (
        (x["Fast SMA"] < x["Slow SMA"]) &
        (x["RSI"] < 50) &
        (x["MACD"] < x["MACD Signal"])
    )

    x["Signal"] = np.select(
        [x["BUY"], x["SELL"]],
        ["BUY", "SELL"],
        default="HOLD"
    )

    return x


def backtest(data):
    required = [
        "Fast SMA",
        "Slow SMA",
        "RSI",
        "MACD Signal"
    ]

    z = data.dropna(subset=required).copy()

    position = (
        z["BUY"].astype(int) -
        z["SELL"].astype(int)
    )

    position = (
        position
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
    )

    daily_return = (
        z["Close"]
        .pct_change()
        .fillna(0)
    )

    strategy_return = (
        position.shift(1).fillna(0) *
        daily_return
    )

    equity = (
        1 + strategy_return
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak
    ) - 1

    return (
        equity.iloc[-1] - 1,
        drawdown.min(),
        equity
    )


with st.sidebar:
    st.header("⚙️ TradePilot Settings")

    ticker = st.text_input(
        "Primary ticker",
        "AAPL"
    ).upper().strip()

    watchlist_text = st.text_input(
        "Watchlist",
        ", ".join(DEFAULT_WATCHLIST)
    )

    watchlist = [
        symbol.strip().upper()
        for symbol in watchlist_text.split(",")
        if symbol.strip()
    ]

    period = st.selectbox(
        "History",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

    fast = st.slider(
        "Fast SMA",
        5,
        100,
        20
    )

    slow = st.slider(
        "Slow SMA",
        20,
        200,
        50
    )

    rsi_len = st.slider(
        "RSI period",
        5,
        50,
        14
    )

    atr_len = st.slider(
        "ATR period",
        5,
        50,
        14
    )

    account = st.number_input(
        "Paper account ($)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=100.0
    )

    risk = st.slider(
        "Risk per trade (%)",
        0.1,
        5.0,
        1.0,
        0.1
    )

    stop_pct = st.slider(
        "Stop distance (%)",
        0.5,
        20.0,
        5.0,
        0.5
    )


if not ticker:
    st.info("Enter a ticker in Settings.")
    st.stop()

try:
    df = load_data(ticker, period)

    if df.empty:
        st.error("No market data found for this ticker.")
        st.stop()

    df = calculate_indicators(
        df,
        fast,
        slow,
        rsi_len,
        atr_len
    )

    last = df.iloc[-1]

    price = float(last["Close"])
    signal = str(last["Signal"])

    rsi = (
        float(last["RSI"])
        if pd.notna(last["RSI"])
        else np.nan
    )

    atr = (
        float(last["ATR"])
        if pd.notna(last["ATR"])
        else np.nan
    )

    st.subheader("⭐ Watchlist")

    watchlist_cols = st.columns(
        min(len(watchlist), 3)
    )

    for i, symbol in enumerate(watchlist):
        try:
            watch_data = load_data(
                symbol,
                "5d"
            )

            if not watch_data.empty:
                watch_price = float(
                    watch_data["Close"].iloc[-1]
                )

                previous = (
                    float(watch_data["Close"].iloc[-2])
                    if len(watch_data) > 1
                    else watch_price
                )

                change = (
                    (watch_price - previous)
                    / previous * 100
                    if previous else 0
                )

                with watchlist_cols[
                    i % len(watchlist_cols)
                ]:
                    st.metric(
                        symbol,
                        f"${watch_price:,.2f}",
                        f"{change:+.2f}%"
                    )

        except Exception:
            pass

    st.subheader(f"📊 {ticker}")

    c1, c2 = st.columns(2)

    c1.metric(
        "Price",
        f"${price:,.2f}"
    )

    c2.metric(
        "Signal",
        signal
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "RSI",
        f"{rsi:.1f}"
        if np.isfinite(rsi)
        else "—"
    )

    c2.metric(
        "ATR",
        f"${atr:.2f}"
        if np.isfinite(atr)
        else "—"
    )

    if signal == "BUY":
        st.success(
            "BUY condition detected by the selected rules."
        )
    elif signal == "SELL":
        st.error(
            "SELL condition detected by the selected rules."
        )
    else:
        st.info(
            "HOLD — no complete signal condition detected."
        )

    st.subheader(f"📈 {ticker} Price")

    st.line_chart(
        df[
            [
                "Close",
                "Fast SMA",
                "Slow SMA"
            ]
        ].dropna(),
        height=300
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Indicators",
            "Risk",
            "Backtest",
            "Data"
        ]
    )

    with tab1:
        st.subheader("MACD")

        st.line_chart(
            df[
                ["MACD", "MACD Signal"]
            ].dropna(),
            height=220
        )

        st.subheader("RSI")

        st.line_chart(
            df[["RSI"]].dropna(),
            height=220
        )

        st.subheader("Recent signals")

        st.dataframe(
            df[
                [
                    "Close",
                    "Fast SMA",
                    "Slow SMA",
                    "RSI",
                    "MACD",
                    "ATR",
                    "Signal"
                ]
            ].tail(15).round(2),
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.subheader("💰 Paper Trade Risk")

        risk_dollars = (
            account * risk / 100
        )

        if np.isfinite(atr) and atr > 0:
            stop_distance = max(
                atr * 2,
                price * stop_pct / 100
            )
        else:
            stop_distance = (
                price * stop_pct / 100
            )

        shares = (
            int(risk_dollars / stop_distance)
            if stop_distance > 0
            else 0
        )

        position_value = shares * price
        stop_price = price - stop_distance
        target_price = price + stop_distance * 2

        a, b = st.columns(2)

        a.metric(
            "Max risk",
            f"${risk_dollars:,.2f}"
        )

        b.metric(
            "Shares",
            f"{shares:,}"
        )

        a, b = st.columns(2)

        a.metric(
            "Position value",
            f"${position_value:,.2f}"
        )

        b.metric(
            "ATR",
            f"${atr:,.2f}"
            if np.isfinite(atr)
            else "—"
        )

        a, b = st.columns(2)

        a.metric(
            "Example stop",
            f"${stop_price:,.2f}"
        )

        b.metric(
            "2R target",
            f"${target_price:,.2f}"
        )

        st.caption(
            "Paper-trading calculation only. "
            "TradePilot does not place real-money orders."
        )

    with tab3:
        st.subheader("📊 Rule-Based Backtest")

        if len(df.dropna()) < 2:
            st.info("Not enough data to run the backtest.")
        else:
            total_return, max_drawdown, equity = backtest(df)

            m1, m2 = st.columns(2)

            m1.metric(
                "Strategy return",
                f"{total_return * 100:.2f}%"
            )

            m2.metric(
                "Max drawdown",
                f"{max_drawdown * 100:.2f}%"
            )

            st.line_chart(
                equity.rename("Equity"),
                height=250
            )

            st.caption(
                "Historical rule-based backtest only; "
                "not a prediction of future results."
            )

    with tab4:
        st.subheader("📋 Market Data")

        st.dataframe(
            df.tail(30).round(2),
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.error("TradePilot could not load the dashboard data.")
    st.exception(e)
