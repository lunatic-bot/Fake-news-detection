from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def create_tfidf_features(text_series, ngram_range=(1,2), max_features=5000):
    """
    Convert a series of cleaned text into a TF-IDF feature matrix.

    Parameters:
    ----------
    text_series : pd.Series
        Series containing cleaned text strings.
    ngram_range : tuple, default=(1,2)
        The lower and upper boundary of the n-grams to be extracted.
    max_features : int, default=5000
        Maximum number of features to keep in the vocabulary.

    Returns:
    -------
    X : sparse matrix
        TF-IDF features for each text entry.
    vectorizer : TfidfVectorizer object
        Fitted vectorizer (useful for transforming new data).
    """
    try:
        if text_series.isna().any():
            raise ValueError("Input text_series contains NaN values. Remove or fill NaNs before vectorization.")

        # Initialize TF-IDF vectorizer
        tfvec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

        # Fit and transform the text series
        X = tfvec.fit_transform(text_series)

        return X, tfvec

    except Exception as e:
        print(f"Error in TF-IDF vectorization: {e}")
        return None, None


def encode_labels(label_series):
    """
    Convert binary labels into numeric array suitable for ML.

    Parameters:
    ----------
    label_series : pd.Series
        Series containing labels (e.g., 'FAKE', 'REAL').

    Returns:
    -------
    y : numpy array
        Binary encoded labels (0 or 1).
    """
    try:
        if label_series.isna().any():
            raise ValueError("Input label_series contains NaN values. Remove or fill NaNs before encoding.")

        # Convert labels to 0/1 using one-hot encoding
        y = pd.get_dummies(label_series)
        y = y.iloc[:,1].values   # select second column as numeric target
        y = y.astype(int)

        return y

    except Exception as e:
        print(f"Error in label encoding: {e}")
        return None
