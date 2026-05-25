# News Sentiment & Stock Price Prediction

Predicting stock price movements by combining financial news sentiment (scored via GPT-4) with machine learning classification models.

---

## Overview

This project builds an end-to-end pipeline that transforms raw financial news into trading signals:

1. **News Collection** — Fetch financial news articles via API
2. **Sentiment Scoring** — Send articles to GPT-4 to generate structured sentiment scores
3. **Feature Engineering** — Process sentiment scores alongside price and volume data
4. **Classification Models** — Train Logistic Regression and Random Forest models to predict stock price direction
5. **Evaluation** — Assess model performance with accuracy, F1-score, and confusion matrix

---

## Motivation

Traditional sentiment analysis tools rely on fixed keyword dictionaries, which struggle with the nuance of financial language. This project leverages GPT-4's contextual understanding to produce more reliable sentiment signals, then tests whether those signals have predictive power over short-term stock price movements.

---

## Pipeline

```
News API → Raw Articles
    ↓
GPT-4 Sentiment Scoring (score + reasoning per article)
    ↓
Feature Engineering (sentiment score, lag features, price data)
    ↓
Model Training (Logistic Regression / Random Forest)
    ↓
Prediction → Stock Price Direction (Up / Down)
    ↓
Evaluation (Accuracy, F1, Confusion Matrix)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| LLM / NLP | OpenAI GPT-4 API |
| ML Models | Logistic Regression, Random Forest |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| News Data | NewsAPI *(or specify your source)* |

---

## Project Structure

```
news-sentiment-stock-prediction/
├── README.md
├── main.py                  # Main pipeline script
├── sentiment_scoring.py     # GPT-4 sentiment scoring module
├── model.py                 # Model training and evaluation
├── data/
│   └── sample_news.csv      # Sample input data
├── results/
│   └── confusion_matrix.png # Evaluation results
└── report.pdf               # Full project report
```

---

## Key Results

> *(Fill in after running your models — e.g., test accuracy, F1-score, which model performed better)*

- Random Forest achieved higher stability across different time windows compared to Logistic Regression
- GPT-4 sentiment scores showed stronger predictive signal than raw keyword-based scores
- Model performance varied across market regimes (high vs. low volatility periods)

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY="your_key_here"

# 3. Run the pipeline
python main.py
```

> **Note:** An OpenAI API key is required for sentiment scoring.

---

## Limitations & Future Work

- Sentiment scoring via API introduces latency — not suitable for real-time trading
- Future work: explore fine-tuned financial LLMs (e.g., FinBERT) as a faster alternative
- Expanding to multi-stock or sector-level analysis

---

*This project was completed as part of coursework at National Tsing Hua University (NTHU).*
