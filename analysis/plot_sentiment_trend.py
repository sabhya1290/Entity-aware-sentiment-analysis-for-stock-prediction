import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================================
# STEP 1: Load your sentiment data
# =========================================
# Change this to your file name
sentiment_file = "data/processed/dataset_with_dates.csv"

sent_df = pd.read_csv(sentiment_file)

# Make sure Date column is proper datetime
sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce")
sent_df = sent_df.dropna(subset=["Date"])

# If you have multiple news rows per day, aggregate to daily mean sentiment
sent_daily = (
    sent_df.groupby("Date", as_index=False)["sentiment_score"]
    .mean()
    .sort_values("Date")
)

# 30-day moving average for sentiment
sent_daily["sentiment_ma30"] = sent_daily["sentiment_score"].rolling(window=30).mean()

# =========================================
# STEP 2: Download NIFTY 500 from Yahoo Finance
# =========================================
ticker = "^CRSLDX"   # NIFTY 500
market_df = yf.download(
    ticker,
    start="2002-01-01",
    end="2018-01-01",
    auto_adjust=False,
    progress=False
)

# Reset index
market_df = market_df.reset_index()

# Keep only needed columns
market_df = market_df[["Date", "Close"]].copy()

# 30-day moving average for market
market_df["market_ma30"] = market_df["Close"].rolling(window=30).mean()

# =========================================
# STEP 3: Merge sentiment and market data by date
# =========================================
merged = pd.merge(sent_daily, market_df, on="Date", how="inner")

# Drop rows where moving averages are not available yet
merged = merged.dropna(subset=["sentiment_ma30", "market_ma30"]).copy()

# =========================================
# STEP 4: Normalize both series to compare on same chart
# =========================================
# This makes both lines comparable visually
merged["sentiment_norm"] = (
    (merged["sentiment_ma30"] - merged["sentiment_ma30"].min()) /
    (merged["sentiment_ma30"].max() - merged["sentiment_ma30"].min())
)

merged["market_norm"] = (
    (merged["market_ma30"] - merged["market_ma30"].min()) /
    (merged["market_ma30"].max() - merged["market_ma30"].min())
)

# =========================================
# STEP 5: Detect local minima on market curve
# =========================================
# These can be used as red/green dots like in the paper-style figure
merged["prev_market"] = merged["market_norm"].shift(1)
merged["next_market"] = merged["market_norm"].shift(-1)

market_minima = merged[
    (merged["market_norm"] < merged["prev_market"]) &
    (merged["market_norm"] < merged["next_market"])
].copy()

# For a cleaner graph, keep only important minima
# Adjust threshold if too many points appear
market_minima = market_minima[market_minima["market_norm"] < 0.45]

# =========================================
# STEP 6: Plot
# =========================================
plt.figure(figsize=(16, 7))

# Sentiment line
plt.plot(
    merged["Date"],
    merged["sentiment_norm"],
    label="News-based Sentiment Index (30-day MA)",
    linewidth=1.5
)

# Market line
plt.plot(
    merged["Date"],
    merged["market_norm"],
    label="NIFTY 500 Index (30-day MA)",
    linewidth=1.5
)

# Mark minima
plt.scatter(
    market_minima["Date"],
    market_minima["market_norm"],
    s=45,
    label="Market Troughs"
)

plt.title("Sentiment Index vs NIFTY 500 (30-day Moving Average)")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================
# STEP 7: Save merged output
# =========================================
merged.to_csv("merged_sentiment_market.csv", index=False)
print("Saved: merged_sentiment_market.csv")