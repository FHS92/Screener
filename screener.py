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
st.title("S&P 500 Screener App by FHS")

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

# ===================== FUNDAMENTAL VALUATION BLOCK =====================

st.subheader("📉 Fundamental Valuation")

try:
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # Current price
    current_price = close.iloc[-1]

    # Get fundamentals with fallback
    pe = info.get("trailingPE")
    ps = info.get("priceToSalesTrailing12Months")
    pb = info.get("priceToBook")
    ev_to_ebitda = info.get("enterpriseToEbitda")
    eps = info.get("trailingEps")
    revenue = info.get("totalRevenue")
    net_income = info.get("netIncomeToCommon")
    book_value = info.get("bookValue")
    ebitda = info.get("ebitda")

    # Sector benchmarking
    sector = info.get("sector", None)
    if sector:
        sp500 = yf.Ticker("^GSPC")
        sector_pe = pe  # default to same if sector data not found
        sector_ps = ps
        sector_pb = pb
        sector_ev_ebitda = ev_to_ebitda

        # Optional: build dictionary with known sector multiples (can be expanded)
        sector_benchmarks = {
            "Technology": {"PE": 25, "PS": 6, "PB": 10, "EV/EBITDA": 20},
            "Healthcare": {"PE": 20, "PS": 5, "PB": 4, "EV/EBITDA": 15},
            "Financial Services": {"PE": 12, "PS": 3, "PB": 1.5, "EV/EBITDA": 10},
            "Consumer Cyclical": {"PE": 18, "PS": 2, "PB": 3, "EV/EBITDA": 12},
            "Energy": {"PE": 10, "PS": 1.5, "PB": 1.2, "EV/EBITDA": 6}
        }
        benchmarks = sector_benchmarks.get(sector, {})
        sector_pe = benchmarks.get("PE", pe or 15)
        sector_ps = benchmarks.get("PS", ps or 3)
        sector_pb = benchmarks.get("PB", pb or 2)
        sector_ev_ebitda = benchmarks.get("EV/EBITDA", ev_to_ebitda or 10)
    else:
        sector_pe = pe or 15
        sector_ps = ps or 3
        sector_pb = pb or 2
        sector_ev_ebitda = ev_to_ebitda or 10

    # === TARGET PRICE ESTIMATES ===
    targets = {}

    if eps and sector_pe:
        targets['P/E'] = eps * sector_pe
    if revenue and sector_ps:
        shares_outstanding = info.get("sharesOutstanding")
        if shares_outstanding:
            revenue_per_share = revenue / shares_outstanding
            targets['P/S'] = revenue_per_share * sector_ps
    if book_value and sector_pb:
        targets['P/B'] = book_value * sector_pb
    if ebitda and sector_ev_ebitda:
        enterprise_value = ebitda * sector_ev_ebitda
        debt = info.get("totalDebt", 0)
        cash = info.get("totalCash", 0)
        equity_value = enterprise_value - debt + cash
        shares_outstanding = info.get("sharesOutstanding", None)
        if shares_outstanding and equity_value:
            targets['EV/EBITDA'] = equity_value / shares_outstanding

    # --- Display Table of Metrics and Target Prices ---
    method_rows = []
    for method, target in targets.items():
        diff_pct = ((target - current_price) / current_price) * 100
        method_rows.append({
            "Method": method,
            "Target Price": f"${target:.2f}",
            "Current Price": f"${current_price:.2f}",
            "Difference %": f"{diff_pct:+.2f}%",
            "Direction": "BUY" if diff_pct > 5 else "SELL" if diff_pct < -5 else "NEUTRAL"
        })

    method_df = pd.DataFrame(method_rows)
    st.dataframe(method_df.set_index("Method"), use_container_width=True)

    # --- AGGREGATE TARGET ---
    if targets:
        avg_target = sum(targets.values()) / len(targets)
        total_diff_pct = ((avg_target - current_price) / current_price) * 100
        overall_fundamental = "BUY" if total_diff_pct > 5 else "SELL" if total_diff_pct < -5 else "NEUTRAL"

        col1, col2, col3 = st.columns(3)
        col1.metric("🔎 Fundamental Target", f"${avg_target:.2f}")
        col2.metric("💰 Current Price", f"${current_price:.2f}")
        col3.metric("📊 Fundamental Direction", f"{overall_fundamental}", delta=f"{total_diff_pct:.2f}%", delta_color="inverse" if overall_fundamental == "SELL" else "normal")

  # --- FUNDAMENTAL VALUATION RAW DATA BLOCK ---
except Exception as e:
    st.warning(f"⚠️ Error: {e}")
    
try:
    st.subheader("📄 Raw Fundamentals")

    ticker_obj = yf.Ticker(symbol)
    info = ticker_obj.info
    # Determine latest report date safely from timestamp
    fiscal_date = info.get("lastFiscalYearEnd")

    if isinstance(fiscal_date, (int, float)) and fiscal_date > 0:
        report_date = pd.to_datetime(fiscal_date, unit='s').strftime("%Y-%m-%d")
    else:
        report_date = "Unavailable"

    st.markdown(f"**🗓 Data as of:** {report_date}")

    # Collect fundamentals
    fundamentals_raw = {
        "Revenue (TTM)": info.get("totalRevenue"),
        "Net Income (TTM)": info.get("netIncomeToCommon"),
        "Free Cash Flow": info.get("freeCashflow"),
        "Book Value": info.get("bookValue"),
        "EPS (TTM)": info.get("trailingEps"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price/Sales Ratio": info.get("priceToSalesTrailing12Months"),
        "Price/Book Ratio": info.get("priceToBook"),
        "PEG Ratio": info.get("pegRatio"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Beta": info.get("beta"),
        "Shares Outstanding": info.get("sharesOutstanding"),
        "Market Cap": info.get("marketCap"),
    }

    # Display with formatting
    for key, val in fundamentals_raw.items():
        if val is None:
            st.markdown(f"- **{key}:** N/A")
        elif isinstance(val, (int, float)):
            if abs(val) >= 1_000_000:
                val_fmt = f"${val/1_000_000_000:.2f}B" if abs(val) >= 1_000_000_000 else f"${val/1_000_000:.2f}M"
            else:
                val_fmt = f"${val:,.2f}"
            st.markdown(f"- **{key}:** {val_fmt}")
        else:
            st.markdown(f"- **{key}:** {val}")
except Exception as e:
    st.error(f"⚠️ Error in fundamental valuation: {e}")
 
