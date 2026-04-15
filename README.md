# 📊 Financial Sentiment Analysis using FinBERT (SEntFiN 1.0)

## 🚀 Project Overview

This project focuses on **financial sentiment analysis** using a fine-tuned transformer model.
We use the **SEntFiN 1.0 dataset** to train and evaluate models that classify financial news headlines into:

* **Positive**
* **Neutral**
* **Negative**

The core objective is to understand how well a model can capture sentiment in financial text and analyze its behavior through evaluation metrics and trend visualization.

---

## 🧠 Model Used

* Base Model: `FinBERT` (from Hugging Face)
* Fine-tuned on: **SEntFiN 1.0 dataset**
* Framework: **PyTorch + Transformers**

---

## 📂 Project Structure

```
open_ended_project/
├── data/
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── outputs/
│   └── best_model/        # Trained model (download separately)
├── src/
│   ├── train.py           # Model training
│   ├── evaluate.py        # Accuracy, F1, confusion matrix
│   └── predict.py         # Inference
├── analysis/
│   └── plot_sentiment_trend.py   # Trend visualization
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sabhya1290/Entity-aware-sentiment-analysis-for-stock-prediction.git
cd open_ended_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Trained Model

Due to GitHub file size limitations, the trained model is hosted externally.

👉 **Download model from Google Drive:**
(https://drive.google.com/drive/folders/1lEw-LQMjX0jQE20xLpPaeMMYfT0ySjOv?usp=drive_link)

After downloading, place it in:

```
outputs/best_model/
```

---

## 🏋️ Training the Model

```bash
python src/train.py
```

---

## 📈 Model Evaluation

```bash
python src/evaluate.py
```

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📊 Sentiment Trend Visualization

```bash
python analysis/plot_sentiment_trend.py
```

### 🔍 What this graph shows

* Comparison between **actual sentiment** and **model-predicted sentiment**
* Uses:

  * Sentiment mapping:

    * Negative = 0
    * Neutral = 1
    * Positive = 2
  * **30-point moving average** for smoothing

### ⚠️ Important Note

The SEntFiN dataset does **not contain real timestamps**.
Therefore:

* Synthetic dates are generated for visualization
* The graph represents **trend comparison**, not real-world time-series analysis
* It does **NOT reflect stock market correlation**

---

## 📚 Dataset

* **SEntFiN 1.0**
* Financial news headlines with entity-level sentiment labels

---

## 🧠 Key Learnings

* Transformer-based models (FinBERT) perform well on financial text
* Sentiment classification can be improved using domain-specific data
* Trend visualization helps understand model behavior beyond accuracy metrics

---

## ⚠️ Limitations

* No real-time or timestamped data in SEntFiN
* Trend graph uses synthetic dates
* No direct integration with stock market indices (e.g., NSE 500)

---

## 🚀 Future Work

* Integrate real-time financial news APIs
* Compare sentiment with stock market indices (NSE 500)
* Build a real-time trading signal system
* Deploy model using Streamlit or web app

---

## 👨‍💻 Author

* Sabhya
* Manglesh 
* Dhani

---

## 📜 License

This project is for academic and research purposes.
