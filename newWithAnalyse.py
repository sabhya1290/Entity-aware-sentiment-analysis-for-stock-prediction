import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def finbert_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_id = torch.argmax(probs).item()

    return labels[sentiment_id], probs[0][sentiment_id].item()

# Load FinBERT
MODEL_NAME = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["negative", "neutral", "positive"]

df = pd.read_csv("stockNewsByBBC.csv") 


sentiments = []
confidences = []

for text in tqdm(df["text"]):
    sentiment, confidence = finbert_sentiment(str(text))
    sentiments.append(sentiment)
    confidences.append(confidence)


df["sentiment"] = sentiments
df["confidence"] = confidences


df.to_csv("news_with_sentiment.csv", index=False)
