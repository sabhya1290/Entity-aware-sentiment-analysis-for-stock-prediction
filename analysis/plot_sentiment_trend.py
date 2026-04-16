import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


sentiment_file = "data/processed/sentfin_entity_aware_date.csv"

sent_df = pd.read_csv(sentiment_file)

sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce")

sent_df = sent_df.dropna(subset=["Date", "label"])

sent_df["label"] = pd.to_numeric(sent_df["label"], errors="coerce")
sent_df = sent_df.dropna(subset=["label"])

sent_daily = (
    sent_df.groupby("Date", as_index=False)["label"]
    .mean()
    .sort_values("Date")
)

sent_daily["sentiment_ma30"] = sent_daily["label"].rolling(window=30).mean()


ticker = "^CRSLDX"  

market_df = yf.download(
    ticker,
    start="2002-01-01",
    end="2018-01-01",
    auto_adjust=False,
    progress=False
)

market_df = market_df.reset_index()

if isinstance(market_df.columns, pd.MultiIndex):
    market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]

market_df = market_df[["Date", "Close"]].copy()
market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
market_df = market_df.dropna(subset=["Date", "Close"])

market_df["market_ma30"] = market_df["Close"].rolling(window=30).mean()


merged = pd.merge(sent_daily, market_df, on="Date", how="inner")
merged = merged.dropna(subset=["sentiment_ma30", "market_ma30"]).copy()


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


merged["prev_market"] = merged["market_norm"].shift(1)
merged["next_market"] = merged["market_norm"].shift(-1)

market_minima = merged[
    (merged["market_norm"] < merged["prev_market"]) &
    (merged["market_norm"] < merged["next_market"])
].copy()

market_minima = market_minima[market_minima["market_norm"] < 0.45]

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
plt.savefig("sentiment_trend.png", dpi=300)
plt.show()

merged.to_csv("merged_sentiment_market.csv", index=False)
print("DONE: Graph generated and merged_sentiment_market.csv saved")