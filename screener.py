import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import requests
from ta.trend import EMAIndicator, SMAIndicator, WMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

try:
    import talib
except ImportError:
    talib = None

st.set_page_config(layout="wide")
st.title("S&P 500 Technical Analysis Screener with Patterns")

# --- INPUT ---
symbol = st.text_input("Enter a stock symbol:", "AAPL").upper()
df = yf.download(symbol, period="1y", interval="1d")
if df.empty:
    st.warning("No data found. Please check the ticker.")
    st.stop()

# --- SQUEEZE DATA ---
open_ = df['Open'].squeeze()
high = df['High'].squeeze()
low = df['Low'].squeeze()
close = df['Close'].squeeze()
volume = df['Volume'].squeeze()

# --- INDICATORS ---
df['EMA12'] = EMAIndicator(close, window=12).ema_indicator()
df['EMA26'] = EMAIndicator(close, window=26).ema_indicator()
df['SMA20'] = SMAIndicator(close, window=20).sma_indicator()
df['WMA20'] = WMAIndicator(close, window=20).wma()
df['MACD'] = MACD(close).macd()
df['RSI'] = RSIIndicator(close, window=14).rsi()
bb = BollingerBands(close, window=20, window_dev=2)
df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()
stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
df['Stochastic'] = stoch.stoch()

# --- PATTERNS ---
pattern_scores = {}
pattern_signals = {}
if talib:
    patterns = {
        'Hammer': talib.CDLHAMMER,
        'Shooting Star': talib.CDLSHOOTINGSTAR,
        'Engulfing': talib.CDLENGULFING,
        'Morning Star': talib.CDLMORNINGSTAR,
        'Evening Star': talib.CDLEVENINGSTAR,
    }
    for name, func in patterns.items():
        result = func(open_, high, low, close)
        df[name] = result
        confidence = (result != 0).sum() / len(result)
        pattern_scores[name] = confidence
        if result.iloc[-1] > 0:
            pattern_signals[name] = "Buy"
        elif result.iloc[-1] < 0:
            pattern_signals[name] = "Sell"
        else:
            pattern_signals[name] = "Neutral"
else:
    pattern_signals = {"Pattern Analysis": "TA-Lib Not Installed"}

# --- INTERPRET SIGNALS ---
def interpret_macd(macd_series):
    return "Buy" if macd_series.iloc[-1] > 0 else "Sell" if macd_series.iloc[-1] < 0 else "Neutral"

def interpret_rsi(rsi_series):
    if rsi_series.iloc[-1] > 70:
        return "Sell"
    elif rsi_series.iloc[-1] < 30:
        return "Buy"
    else:
        return "Neutral"

def interpret_stochastic(stoch_series):
    val = stoch_series.iloc[-1]
    if val > 80:
        return "Sell"
    elif val < 20:
        return "Buy"
    else:
        return "Neutral"

def interpret_ema_cross(df):
    if df['EMA12'].iloc[-1] > df['EMA26'].iloc[-1]:
        return "Buy"
    elif df['EMA12'].iloc[-1] < df['EMA26'].iloc[-1]:
        return "Sell"
    else:
        return "Neutral"

# --- SUMMARY TABLE ---
summary = {
    "MACD": interpret_macd(df['MACD']),
    "RSI": interpret_rsi(df['RSI']),
    "Stochastic": interpret_stochastic(df['Stochastic']),
    "EMA Cross": interpret_ema_cross(df),
}

# Add pattern signals
summary.update(pattern_signals)

# --- OVERALL SIGNAL ---
signal_values = list(summary.values())
buy_count = signal_values.count("Buy")
sell_count = signal_values.count("Sell")

if buy_count > sell_count:
    overall = "BUY"
elif sell_count > buy_count:
    overall = "SELL"
else:
    overall = "NEUTRAL"

# --- TARGET PRICE ESTIMATES ---

# Current price
current_price = close.iloc[-1]

# === VOLATILITY-BASED TARGET ===
volatility = df['Close'].pct_change().rolling(window=20).std().iloc[-1]
confidence_factor = 1.5  # can be tuned
if overall == "BUY":
    vol_target = float(current_price * (1 + volatility * confidence_factor))
elif overall == "SELL":
    vol_target = float(current_price * (1 - volatility * confidence_factor))
else:
    vol_target = float(current_price)

vol_diff_pct = ((vol_target - current_price) / current_price) * 100

# === FIBONACCI-BASED TARGET ===
recent_high = df['High'].rolling(window=50).max().iloc[-1]
recent_low = df['Low'].rolling(window=50).min().iloc[-1]
diff = recent_high - recent_low

fib_0_382 = recent_high - 0.382 * diff
fib_0_618 = recent_high - 0.618 * diff
fib_1_618 = recent_high + 0.618 * diff

if overall == "BUY":
    fib_target = float(fib_1_618)
elif overall == "SELL":
    fib_target = float(fib_0_382)
else:
    fib_target = float((fib_0_382 + fib_0_618) / 2)

fib_diff_pct = ((fib_target - current_price) / current_price) * 100

# === DISPLAY PRICE ESTIMATES ===
st.subheader(f"📊 Overall Signal: **{overall}**")

colA, colB, colC = st.columns(3)
colA.metric("💰 Current Price", f"${current_price:.2f}")
colB.metric("🎯 Volatility Target", f"${vol_target:.2f}", f"{vol_diff_pct:.2f}%")
colC.metric("🔮 Fibonacci Target", f"${fib_target:.2f}", f"{fib_diff_pct:.2f}%")

# --- DISPLAY SUMMARY TABLE ---
summary_df = pd.DataFrame(list(summary.items()), columns=["Indicator / Pattern", "Signal"])
st.dataframe(summary_df.set_index("Indicator / Pattern"), use_container_width=True)


# --- COMPANY PROFILE ---
col1, col2 = st.columns([1, 3])

with col1:
    try:
        profile = yf.Ticker(symbol).info
        domain = profile.get("website", "").replace("https://", "").replace("www.", "").split("/")[0]
        if domain:
            logo_url = f"https://logo.clearbit.com/{domain}"
            st.image(logo_url, width=100)
        else:
            st.warning("Logo not found.")
    except Exception:
        st.warning("Failed to fetch logo.")

with col2:
    try:
        st.subheader("Company Overview")
        st.markdown(f"**Industry:** {profile.get('industry', 'N/A')}")
        st.markdown(f"**Products:** {profile.get('longBusinessSummary', 'N/A')[:500]}...")

        rev = profile.get('totalRevenue', None)
        net = profile.get('netIncomeToCommon', None)
        rev_growth = profile.get('revenueGrowth', None)
        net_growth = profile.get('netIncomeToCommonGrowth', None)

        st.markdown(f"**Latest Revenue:** ${rev:,} ({profile.get('financialCurrency', '')})" if rev else "Revenue data unavailable.")
        st.markdown(f"**Net Income:** ${net:,}" if net else "Net Income data unavailable.")
        if rev_growth is not None:
            st.markdown(f"**YoY Revenue Growth:** {rev_growth * 100:.2f}%")
        if net_growth is not None:
            st.markdown(f"**YoY Net Income Growth:** {net_growth * 100:.2f}%")

    except Exception:
        st.warning("Company profile not available.")

# --- CHART ---
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=open_,
    high=high,
    low=low,
    close=close,
    name="Candlesticks"
)])
fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], line=dict(color='blue', width=1), name='EMA12'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'], line=dict(color='orange', width=1), name='EMA26'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='green', width=1), name='SMA20'))
fig.add_trace(go.Scatter(x=df.index, y=df['WMA20'], line=dict(color='purple', width=1), name='WMA20'))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='grey', width=1, dash='dash'), name='BB High'))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='grey', width=1, dash='dash'), name='BB Low'))

fig.update_layout(title=f"{symbol} Price Chart with Indicators", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

st.subheader("RSI")
st.line_chart(df['RSI'])

st.subheader("MACD")
st.line_chart(df['MACD'])

st.subheader("Stochastic Oscillator")
st.line_chart(df['Stochastic'])

if st.checkbox("Show Raw Data"):
    st.write(df.tail(20))
