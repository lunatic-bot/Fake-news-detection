from pathlib import Path
import pandas as pd

from preprocessing import preprocess_dataframe
from vectorization import create_tfidf_features, encode_labels
from model_training import train_logistic_regression, evaluate_model, save_model

# Step 1: Define paths
BASE_DIR = Path().resolve().parent
RAW_CSV = BASE_DIR / 'data/raw/combined_news_data.csv'
CLEANED_CSV = BASE_DIR / 'data/processed/cleaned_news_data.csv'
MODEL_PATH = BASE_DIR / 'models/fake_news_model.pkl'
VECTORIZER_PATH = BASE_DIR / 'models/tfidf_vectorizer.pkl'

print(RAW_CSV)

# Step 2: Load raw data
try:
    raw_df = pd.read_csv(RAW_CSV)
    print(raw_df.columns)
except Exception as e:
    print(f"Error loading CSV: {e}")
    raw_df = pd.DataFrame()  # fallback empty df

# Step 3: Preprocess text
cleaned_df = preprocess_dataframe(raw_df, output_csv_path=CLEANED_CSV)

print(cleaned_df.head())
# Step 4: Vectorization
X, tfvec = create_tfidf_features(cleaned_df['cleaned_text'])
y = encode_labels(cleaned_df['label'])

# Step 5: Train model
model, X_train, X_test, y_train, y_test = train_logistic_regression(X, y)

# Step 6: Evaluate
metrics, top_real, top_fake = evaluate_model(model, X_test, y_test, feature_names=tfvec.get_feature_names_out())

# Step 7: Save model & vectorizer
save_model(model, tfvec, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)

print("Pipeline completed successfully.")
