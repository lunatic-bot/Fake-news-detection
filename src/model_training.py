from preprocessing import preprocess_dataframe
from vectorization import create_tfidf_features, encode_labels

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

def train_logistic_regression(X, y, test_size=0.2, random_state=42):
    """
    Train a Logistic Regression model with optional train/test split.

    Parameters:
    ----------
    X : feature matrix
    y : target labels
    test_size : float, default=0.2
    random_state : int, default=42

    Returns:
    -------
    model : trained LogisticRegression object
    X_train, X_test, y_train, y_test : split data
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None, None, None


def evaluate_model(model, X_test, y_test, feature_names=None, top_n=20):
    """
    Evaluate a trained model and print metrics and top feature indicators.

    Parameters:
    ----------
    model : trained model
    X_test, y_test : test data
    feature_names : list or array of TF-IDF feature names
    top_n : int, number of top features to display

    Returns:
    -------
    metrics : dict of accuracy, precision, recall, f1
    top_real_words, top_fake_words : arrays of top indicative words
    """
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Accuracy: {acc}")
        print(classification_report(y_test, y_pred))

        top_real_words = top_fake_words = None
        if feature_names is not None:
            coefficients = model.coef_[0]
            # Top FAKE indicators: largest positive coefficients
            top_fake_idx = np.argsort(coefficients)[-top_n:]
            top_fake_words = feature_names[top_fake_idx]

            # Top REAL indicators: largest negative coefficients
            top_real_idx = np.argsort(coefficients)[:top_n]
            top_real_words = feature_names[top_real_idx]

            print("Top REAL words:", top_real_words)
            print("Top FAKE words:", top_fake_words)

        metrics = {
            "accuracy": acc,
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "f1": report['1']['f1-score']
        }

        return metrics, top_real_words, top_fake_words

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None, None, None


def cross_validate_model(model, X, y, cv_folds=5, scoring_metrics=None):
    """
    Perform cross-validation on the model and return mean metrics.

    Parameters:
    ----------
    model : sklearn model
    X, y : data
    cv_folds : int, default=5
    scoring_metrics : list of str, optional

    Returns:
    -------
    mean_scores : dict of mean scores for each metric
    """
    try:
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

        cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=scoring_metrics)
        mean_scores = {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring_metrics}
        return mean_scores

    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return None


def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Save trained model and vectorizer to disk using pickle.
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print("Model and vectorizer saved successfully.")
    except Exception as e:
        print(f"Error saving model/vectorizer: {e}")
