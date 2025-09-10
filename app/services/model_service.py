import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as es:
    raise RuntimeError(f"Error loading model/vectorizer : {es}")

def predict(text: str):
    try:
        ## Convert text into TF-IDF features
        X = vectorizer.transform([text])

        ## get prediction probabilities
        probs = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]

        ## confidence is the probability of predicted class
        confidence = float(np.max(probs))

        ## map numeric label to string
        label_map = {0: "FAKE", 1: "REAL"}
        prediction = label_map.get(pred_class, str(pred_class))

        return prediction, confidence

    except Exception as es:
        raise RuntimeError(f"Prediction error : {es}")
