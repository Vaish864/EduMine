import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix():
    """
    Convert cleaned feedback comments into TF-IDF numerical features.
    """

    # Load processed dataset
    df = pd.read_csv(
        "data/processed_dataset.csv",
        encoding="latin1"
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000
    )

    # Convert cleaned comments into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(
        df["cleaned_comments"]
    )

    # Get feature names (important words)
    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF created successfully!")
    print("Number of feedback records:", tfidf_matrix.shape[0])
    print("Number of TF-IDF features:", tfidf_matrix.shape[1])

    print("\nSample features:")
    print(feature_names[:20])

    return tfidf_matrix, vectorizer, df


if __name__ == "__main__":
    create_tfidf_matrix()