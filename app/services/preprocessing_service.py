from pathlib import Path
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd

# Load stopwords once globally
stop_words = set(stopwords.words('english'))

def remove_html(text):
    try:
        text = str(text)  # ensure it's a string
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error removing HTML: {e}")
        return ""

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
    

def text_preprocess(text):
    try:
        text = remove_html(text)
        cleaned_text = clean_text(text)

        return cleaned_text
    
    except Exception as es:
        raise ValueError(f"Error while preprocessing the text : {es}")