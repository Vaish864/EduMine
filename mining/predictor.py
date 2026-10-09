
import joblib

MODEL_PATH = "mining/saved_model/sentiment_model.pkl"
VECTORIZER_PATH = "mining/saved_model/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(feedback):
    """Predict positive, neutral, or negative sentiment."""

    if not isinstance(feedback, str) or not feedback.strip():
        return None

    feedback_vector = vectorizer.transform([feedback.strip()])
    prediction = str(model.predict(feedback_vector)[0]).strip().lower()

    if prediction not in {"positive", "neutral", "negative"}:
        raise ValueError(f"Unexpected sentiment label: {prediction}")

    return prediction


if __name__ == "__main__":
    test_feedback = [
        "The teacher explains everything clearly and gives useful examples.",
        "The teaching is confusing and difficult to understand.",
        "The overall teaching experience was average.",
        ""
    ]

    for feedback in test_feedback:
        print("\nFeedback:", repr(feedback))
        print("Predicted sentiment:", predict_sentiment(feedback))