import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load NLTK resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess a single feedback comment.
    """

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into words
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


def preprocess_dataset():
    """
    Read the original dataset, clean the feedback,
    remove duplicate comments, and save the processed dataset.
    """

    # Read original dataset
    df = pd.read_csv(
        "data/original_dataset.csv",
        encoding="latin1"
    )

    # Remove duplicate comments
    df = df.drop_duplicates(subset=["comments"]).copy()

    # Create readable sentiment labels
    sentiment_map = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }

    df["sentiment"] = df["sentiments"].map(sentiment_map)

    # Clean feedback text
    df["cleaned_comments"] = df["comments"].apply(clean_text)

    # Save processed dataset
    df.to_csv(
        "data/processed_dataset.csv",
        index=False
    )

    print("Preprocessing completed successfully!")
    print("Processed records:", len(df))
    print("Saved to: data/processed_dataset.csv")


if __name__ == "__main__":
    preprocess_dataset()