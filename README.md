# Fake News Detection

## Overview

A machine learning project that detects whether a news article is **FAKE** or **REAL** using NLP.
The goal is to practice an **end-to-end ML pipeline**: data preprocessing → model training → evaluation → deployment.

---

## Dataset

- Source: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- \~45k articles (balanced between FAKE and REAL).
- Columns: `title`, `text`, `subject`, `date`.

---

## Workflow

1. **Preprocessing** → clean text (lowercase, remove stopwords, punctuation).
2. **Feature Engineering** → TF-IDF vectors (baseline).
3. **Modeling** → Logistic Regression, SVM, XGBoost (compare performance).
4. **Evaluation** → Accuracy, Precision, Recall, F1, Confusion Matrix.
5. **Deployment** → Simple **Streamlit app** where users paste text to check if it’s fake or real.

## How to Run

```bash
git clone https://github.com/lunatic-bot/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
python src/train_baseline.py
streamlit run app/streamlit_app.py
```

---

## Tech Stack

- **Python**, **pandas**, **scikit-learn**, **nltk**
- **Models**: Logistic Regression, SVM, XGBoost
- **Deployment**: Streamlit

---
