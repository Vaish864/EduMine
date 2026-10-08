import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean a feedback comment before sending it to the model.
    """

    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split into words
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


def train_sentiment_model():
    """
    Train the sentiment classification model.
    """

    # Load processed dataset
    df = pd.read_csv(
        "data/processed_dataset.csv",
        encoding="latin1"
    )

    # Remove missing sentiment values
    df = df.dropna(subset=["sentiment"])

    # Text and labels
    X_text = df["cleaned_comments"]
    y = df["sentiment"]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000
    )

    X = vectorizer.fit_transform(X_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000
    )

    # Train
    model.fit(X_train, y_train)

    # Test
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Sentiment model trained successfully!")
    print("Training records:", X_train.shape[0])
    print("Testing records:", X_test.shape[0])
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a new feedback comment.
    """

    # Clean new feedback
    cleaned_text = clean_text(text)

    # Convert new text into TF-IDF
    text_vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_vector)[0]

    return prediction


if __name__ == "__main__":

    # Train the model
    model, vectorizer = train_sentiment_model()

    print("\n--- Live Feedback Prediction ---")

    # Example new feedback
    new_feedback = input("Enter student feedback: ")

    # Predict sentiment
    result = predict_sentiment(
        new_feedback,
        model,
        vectorizer
    )

    print("\nPredicted Sentiment:", result)