import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="TradePilot Mobile", page_icon="📱", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 760px; padding: 1rem .8rem 3rem;}
h1 {font-size: 1.8rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
button {min-height: 44px;}
</style>
""", unsafe_allow_html=True)

st.title("📱 TradePilot")
st.caption("Mobile stock research & paper-trading dashboard")

@st.cache_data(ttl=60)
def load_data(ticker, period):
    d = yf.download(ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()

def indicators(d, fast, slow, rsi_len):
    x = d.copy()
    x["Fast SMA"] = x["Close"].rolling(fast).mean()
    x["Slow SMA"] = x["Close"].rolling(slow).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100/(1+rs)
    x["EMA12"] = x["Close"].ewm(span=12, adjust=False).mean()
    x["EMA26"] = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = x["EMA12"] - x["EMA26"]
    x["MACD Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["BUY"] = (x["Fast SMA"] > x["Slow SMA"]) & (x["RSI"] > 50) & (x["MACD"] > x["MACD Signal"])
    x["SELL"] = (x["Fast SMA"] < x["Slow SMA"]) & (x["RSI"] < 50) & (x["MACD"] < x["MACD Signal"])
    x["Signal"] = np.select([x["BUY"], x["SELL"]], ["BUY", "SELL"], default="HOLD")
    return x

def backtest(x):
    z = x.dropna(subset=["Fast SMA","Slow SMA","RSI","MACD Signal"]).copy()
    pos = z["BUY"].astype(int) - z["SELL"].astype(int)
    pos = pos.replace(0, np.nan).ffill().fillna(0)
    ret = z["Close"].pct_change().fillna(0)
    strat = pos.shift(1).fillna(0) * ret
    equity = (1 + strat).cumprod()
    peak = equity.cummax()
    dd = equity/peak - 1
    return equity.iloc[-1]-1, dd.min()

# Mobile controls
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    period = st.selectbox("History", ["3mo","6mo","1y","2y","5y"], index=2)
    fast = st.slider("Fast SMA", 5, 100, 20)
    slow = st.slider("Slow SMA", 20, 200, 50)
    rsi_len = st.slider("RSI", 5, 50, 14)
    account = st.number_input("Paper account ($)", 100.0, 10000000.0, 10000.0, 100.0)
    risk = st.slider("Risk per trade (%)", .1, 5.0, 1.0, .1)
    stop_pct = st.slider("Stop distance (%)", .5, 20.0, 5.0, .5)

if not ticker:
    st.info("Enter a ticker in Settings.")
    st.stop()

try:
    df = load_data(ticker, period)
    if df.empty:
        st.error("No market data found.")
        st.stop()

    df = indicators(df, fast, slow, rsi_len)
    last = df.iloc[-1]
    price = float(last["Close"])
    signal = str(last["Signal"])
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else np.nan

    c1, c2 = st.columns(2)
    c1.metric("Price", f"${price:,.2f}")
    c2.metric("Signal", signal)

    c1, c2 = st.columns(2)
    c1.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
    c2.metric("MACD", f"{float(last['MACD']):.2f}")

    if signal == "BUY":
        st.success("BUY condition detected by the selected rules.")
    elif signal == "SELL":
        st.error("SELL condition detected by the selected rules.")
    else:
        st.info("HOLD — no complete signal condition detected.")

    st.subheader(f"📈 {ticker} Price")
    st.line_chart(df[["Close","Fast SMA","Slow SMA"]].dropna(), height=300)

    tab1, tab2, tab3 = st.tabs(["Indicators", "Risk", "Backtest"])

    with tab1:
        st.line_chart(df[["MACD","MACD Signal"]].dropna(), height=220)
        st.line_chart(df[["RSI"]].dropna(), height=220)
        st.dataframe(
            df[["Close","Fast SMA","Slow SMA","RSI","MACD","Signal"]].tail(15).round(2),
            use_container_width=True, hide_index=True
        )

    with tab2:
        risk_dollars = account * risk/100
        stop_distance = price * stop_pct/100
        shares = int(risk_dollars/stop_distance) if stop_distance else 0
        stop = price - stop_distance
        target = price + stop_distance*2
        a,b = st.columns(2)
        a.metric("Max risk", f"${risk_dollars:,.2f}")
        b.metric("Shares", f"{shares:,}")
        a,b = st.columns(2)
        a.metric("Example stop", f"${stop:,.2f}")
        b.metric("2R target", f"${target:,.2f}")
        st.caption("Position sizing is an example based on your inputs; it does not execute a trade.")

    with tab3:
        strategy, drawdown = backtest(df)
        a,b = st.columns(2)
        a.metric("Strategy return", f"{strategy*100:.2f}%")
        b.metric("Max drawdown", f"{drawdown*100:.2f}%")
        st.line_chart((1 + df.dropna(subset=["Fast SMA","Slow SMA","RSI","MACD Signal"])["Close"].pct_change().fillna(0)).cumprod(), height=220)
        st.caption("Historical backtests are hypothetical and do not guarantee future results.")

except Exception as e:
    st.error(f"Could not load {ticker}: {e}")

st.divider()
st.caption("Educational tool. No real-money orders are placed.")
