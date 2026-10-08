import joblib


# Load the saved model and TF-IDF vectorizer
model = joblib.load(
    "mining/saved_model/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "mining/saved_model/tfidf_vectorizer.pkl"
)


def predict_sentiment(feedback):
    """
    Predict sentiment for a new student feedback.
    """

    # Convert feedback into TF-IDF features
    feedback_vector = vectorizer.transform(
        [feedback]
    )

    # Predict sentiment
    prediction = model.predict(
        feedback_vector
    )[0]

    return prediction


if __name__ == "__main__":

    test_feedback = [
        "The teacher explains everything clearly and gives useful examples.",
        "The teaching is confusing and difficult to understand.",
        "The overall teaching experience was average."
    ]

    for feedback in test_feedback:

        sentiment = predict_sentiment(
            feedback
        )

        print("\nFeedback:")
        print(feedback)

        print("Predicted Sentiment:", sentiment)