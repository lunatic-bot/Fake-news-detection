from pathlib import Path
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd

# Load stopwords once globally
stop_words = set(stopwords.words('english'))

def remove_html(text):
    """
    Remove HTML tags from a string using BeautifulSoup.
    """
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    except Exception as e:
        # In case text is not valid HTML
        print(f"Error removing HTML: {e}")
        return text  # fallback to original text

def clean_text(text, lemmatize=False):
    """
    Clean a single text string:
    - Lowercase
    - Remove HTML tags
    - Remove URLs, punctuation, numbers
    - Remove stopwords
    - Optional: lemmatization (if implemented)
    Returns cleaned text as a single string.
    """
    try:
        text = text.lower()  # lowercase
        text = remove_html(text)
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = re.sub(r"[^a-zA-Z]", " ", text)      # keep letters only

        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]

        # Optional lemmatization
        # if lemmatize:
        #     tokens = [lemmatizer.lemmatize(w) for w in tokens]

        return " ".join(tokens)

    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""  # return empty string if cleaning fails

def preprocess_dataframe(df, output_csv_path=None):
    """
    Preprocess a DataFrame:
    - Apply text cleaning to 'full_text' column
    - Keep only ['cleaned_text', 'subject', 'label'] columns
    - Optionally save cleaned DataFrame to CSV
    Returns cleaned DataFrame.
    """
    try:
        # Check required column exists
        if 'full_text' not in df.columns:
            raise KeyError("Input DataFrame must have a 'full_text' column")

        # Apply cleaning
        df['cleaned_text'] = df['full_text'].apply(lambda x: clean_text(x))
        
        df = df[~['cleaned_text'].isna()]
        df = df[df['cleaned_text'].str.strip() != ""]

        # Select relevant columns
        cleaned_df = df[['cleaned_text', 'subject', 'label']]

        # Save to CSV if path provided
        if output_csv_path:
            try:
                cleaned_df.to_csv(output_csv_path, index=False)
            except Exception as e:
                print(f"Error saving CSV: {e}")

        return cleaned_df

    except Exception as e:
        print(f"Error in preprocessing DataFrame: {e}")
        return pd.DataFrame()  # return empty DF on failure




# from sklearn.feature_extraction.text import TfidfVectorizer

# tfvec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
# X = tfvec.fit_transform(cleaned_df['cleaned_text'])
# feature_names = tfvec.get_feature_names_out()
# feature_names
# X[0].toarray()
# y = pd.get_dummies(cleaned_df['label'])
# y = y.iloc[:,1].values
# y = y.astype(int)
# y


