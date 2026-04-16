import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================================
# STEP 1: Load your sentiment data
# =========================================
sentiment_file = "data/processed/dataset_with_dates.csv"

sent_df = pd.read_csv(sentiment_file)

# Convert Date column
sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce")

# Use existing numeric sentiment column: label
sent_df = sent_df.dropna(subset=["Date", "label"])

# Make sure label is numeric
sent_df["label"] = pd.to_numeric(sent_df["label"], errors="coerce")
sent_df = sent_df.dropna(subset=["label"])

# =========================================
# STEP 2: Aggregate daily sentiment
# =========================================
sent_daily = (
    sent_df.groupby("Date", as_index=False)["label"]
    .mean()
    .sort_values("Date")
)

# 30-day moving average for sentiment
sent_daily["sentiment_ma30"] = sent_daily["label"].rolling(window=30).mean()

# =========================================
# STEP 3: Download market data
# =========================================
ticker = "^CRSLDX"   # NIFTY 500
# If this does not work, replace with:
# ticker = "^NSEI"   # NIFTY 50 fallback

market_df = yf.download(
    ticker,
    start="2002-01-01",
    end="2018-01-01",
    auto_adjust=False,
    progress=False
)

market_df = market_df.reset_index()

# Some yfinance versions return MultiIndex columns
if isinstance(market_df.columns, pd.MultiIndex):
    market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]

market_df = market_df[["Date", "Close"]].copy()
market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
market_df = market_df.dropna(subset=["Date", "Close"])

# 30-day moving average for market
market_df["market_ma30"] = market_df["Close"].rolling(window=30).mean()

# =========================================
# STEP 4: Merge sentiment and market data
# =========================================
merged = pd.merge(sent_daily, market_df, on="Date", how="inner")
merged = merged.dropna(subset=["sentiment_ma30", "market_ma30"]).copy()

# =========================================
# STEP 5: Normalize both series
# =========================================
sent_range = merged["sentiment_ma30"].max() - merged["sentiment_ma30"].min()
market_range = merged["market_ma30"].max() - merged["market_ma30"].min()

if sent_range == 0:
    merged["sentiment_norm"] = 0.5
else:
    merged["sentiment_norm"] = (
        (merged["sentiment_ma30"] - merged["sentiment_ma30"].min()) / sent_range
    )

if market_range == 0:
    merged["market_norm"] = 0.5
else:
    merged["market_norm"] = (
        (merged["market_ma30"] - merged["market_ma30"].min()) / market_range
    )

# =========================================
# STEP 6: Detect market troughs
# =========================================
merged["prev_market"] = merged["market_norm"].shift(1)
merged["next_market"] = merged["market_norm"].shift(-1)

market_minima = merged[
    (merged["market_norm"] < merged["prev_market"]) &
    (merged["market_norm"] < merged["next_market"])
].copy()

market_minima = market_minima[market_minima["market_norm"] < 0.45]

# =========================================
# STEP 7: Plot
# =========================================
plt.figure(figsize=(16, 7))

plt.plot(
    merged["Date"],
    merged["sentiment_norm"],
    label="Sentiment Index (30-day MA)",
    linewidth=1.5
)

plt.plot(
    merged["Date"],
    merged["market_norm"],
    label="NIFTY 500 Index (30-day MA)",
    linewidth=1.5
)

plt.scatter(
    market_minima["Date"],
    market_minima["market_norm"],
    s=40,
    label="Market Troughs"
)

plt.title("Sentiment vs Market")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================
# STEP 8: Save output
# =========================================
merged.to_csv("merged_sentiment_market.csv", index=False)
print("DONE: Graph generated and merged_sentiment_market.csv saved")