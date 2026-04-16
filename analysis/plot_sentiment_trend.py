import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ast

# =========================================
# STEP 1: Load your sentiment data
# =========================================
sentiment_file = "data/processed/dataset_with_dates.csv"

sent_df = pd.read_csv(sentiment_file)

# Convert Date column
sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce")

# =========================================
# STEP 2: Convert Decisions → sentiment score
# =========================================
def extract_sentiment(decision):
    try:
        d = ast.literal_eval(decision)   # convert string → dict
        val = list(d.values())[0]        # get sentiment label
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        return mapping.get(val, None)
    except:
        return None

sent_df["sentiment"] = sent_df["Decisions"].apply(extract_sentiment)

# Remove bad rows
sent_df = sent_df.dropna(subset=["Date", "sentiment"])

# =========================================
# STEP 3: Aggregate daily sentiment
# =========================================
sent_daily = (
    sent_df.groupby("Date", as_index=False)["sentiment"]
    .mean()
    .sort_values("Date")
)

# 30-day moving average
sent_daily["sentiment_ma30"] = sent_daily["sentiment"].rolling(window=30).mean()

# =========================================
# STEP 4: Download market data (NIFTY 500)
# =========================================
ticker = "^CRSLDX"   # NIFTY 500

market_df = yf.download(
    ticker,
    start="2002-01-01",
    end="2018-01-01",
    auto_adjust=False,
    progress=False
)

market_df = market_df.reset_index()
market_df = market_df[["Date", "Close"]]

# Moving average
market_df["market_ma30"] = market_df["Close"].rolling(window=30).mean()

# =========================================
# STEP 5: Merge
# =========================================
merged = pd.merge(sent_daily, market_df, on="Date", how="inner")

merged = merged.dropna(subset=["sentiment_ma30", "market_ma30"])

# =========================================
# STEP 6: Normalize (for same scale)
# =========================================
merged["sentiment_norm"] = (
    (merged["sentiment_ma30"] - merged["sentiment_ma30"].min()) /
    (merged["sentiment_ma30"].max() - merged["sentiment_ma30"].min())
)

merged["market_norm"] = (
    (merged["market_ma30"] - merged["market_ma30"].min()) /
    (merged["market_ma30"].max() - merged["market_ma30"].min())
)

# =========================================
# STEP 7: Detect market troughs
# =========================================
merged["prev"] = merged["market_norm"].shift(1)
merged["next"] = merged["market_norm"].shift(-1)

market_minima = merged[
    (merged["market_norm"] < merged["prev"]) &
    (merged["market_norm"] < merged["next"])
]

# Optional: filter strong dips
market_minima = market_minima[market_minima["market_norm"] < 0.45]

# =========================================
# STEP 8: Plot
# =========================================
plt.figure(figsize=(16, 7))

plt.plot(
    merged["Date"],
    merged["sentiment_norm"],
    label="Sentiment Index (30-day MA)"
)

plt.plot(
    merged["Date"],
    merged["market_norm"],
    label="NIFTY 500 Index (30-day MA)"
)

plt.scatter(
    market_minima["Date"],
    market_minima["market_norm"],
    s=40,
    label="Market Troughs"
)

plt.title("Sentiment vs Market (Paper-style Graph)")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)

plt.show()

# =========================================
# STEP 9: Save output
# =========================================
merged.to_csv("merged_sentiment_market.csv", index=False)

print("DONE ✅ Graph generated + CSV saved")