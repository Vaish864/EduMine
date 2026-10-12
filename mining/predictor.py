
import joblib

MODEL_PATH = "mining/saved_model/sentiment_model.pkl"
VECTORIZER_PATH = "mining/saved_model/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(feedback):
    """Predict sentiment using clear phrase rules and the trained model."""

    if not isinstance(feedback, str) or not feedback.strip():
        return None

    text = " ".join(feedback.lower().split())

    # Clear negative expressions
    negative_phrases = [
        "very confusing",
        "difficult to understand",
        "does not explain clearly",
        "doesn't explain clearly",
        "poor teaching",
        "waste of time",
        "not helpful",
        "not satisfied",
        "very disappointing",
    ]

    # Clear neutral expressions
    neutral_phrases = [
        "overall average",
        "average teaching experience",
        "neither good nor bad",
        "nothing special",
        "just okay",
        "just ok",
    ]

    # Avoid overriding a clear phrase when it is explicitly negated.
    if any(
        f"not {phrase}" in text or f"not very {phrase}" in text
        for phrase in negative_phrases
    ):
        pass
    elif any(phrase in text for phrase in negative_phrases):
        return "negative"

    if (
    "overall teaching experience was average" in text
    or "overall average" in text
    or "teaching experience was average" in text
):
      return "neutral"

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
        "The teacher is not helpful.",
        "",
    ]

    for feedback in test_feedback:
        print("\nFeedback:", repr(feedback))
        print("Predicted sentiment:", predict_sentiment(feedback))
