import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# PATHS
# =========================
INPUT_CSV = "data/processed/test.csv"
MODEL_PATH = "outputs/best_model"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

print("Columns:", df.columns.tolist())

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().reset_index(drop=True)

# fake dates for trend visualization
df["date"] = pd.date_range(start="2002-01-01", periods=len(df), freq="D")

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# prediction label map
id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =========================
# PREDICT
# =========================
predictions = []

with torch.no_grad():
    for text in df[TEXT_COLUMN].astype(str):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(id_to_label[pred_id])

df["predicted_sentiment"] = predictions

# =========================
# CONVERT ACTUAL LABELS
# =========================
# if your label column is already numeric: 0,1,2
df["actual_sentiment"] = df[LABEL_COLUMN].map(id_to_label)

# numeric mapping for plotting
score_map = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}

df["actual_num"] = df["actual_sentiment"].map(score_map)
df["predicted_num"] = df["predicted_sentiment"].map(score_map)

# =========================
# DAILY TREND
# =========================
daily = df.groupby("date")[["actual_num", "predicted_num"]].mean().reset_index()

daily["actual_ma30"] = daily["actual_num"].rolling(30, min_periods=1).mean()
daily["predicted_ma30"] = daily["predicted_num"].rolling(30, min_periods=1).mean()

# =========================
# PLOT
# =========================
plt.figure(figsize=(14, 7))
plt.plot(daily["date"], daily["actual_ma30"], label="Actual Sentiment (30-day MA)")
plt.plot(daily["date"], daily["predicted_ma30"], linestyle="--", label="Predicted Sentiment (30-day MA)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Actual vs Predicted Sentiment Trend")
plt.legend()
plt.tight_layout()
plt.savefig("sentiment_trend.png", dpi=300, bbox_inches="tight")
print("Saved plot as sentiment_trend.png")
plt.show()
