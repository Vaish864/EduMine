import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load processed dataset
df = pd.read_csv(
    "data/processed_dataset.csv",
    encoding="latin1"
)

# Remove missing labels
df = df.dropna(subset=["sentiment"])


# Text and labels
X_text = df["cleaned_comments"]
y = df["sentiment"]


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)


# Train Naive Bayes
model = MultinomialNB()
model.fit(X, y)


# Test feedback examples
test_feedback = [
    "The teacher explains concepts clearly and gives useful practical examples.",
    "The teaching method is confusing and very difficult to understand.",
    "The teacher was okay and the overall experience was average.",
    "Excellent teaching and very helpful explanations.",
    "The teacher does not explain the topics properly.",
    "The classes are good but sometimes difficult to follow."
]


print("=" * 60)
print("EduMine - Real World Sentiment Prediction Test")
print("=" * 60)


for feedback in test_feedback:

    # Convert feedback into TF-IDF
    feedback_vector = vectorizer.transform([feedback])

    # Predict
    prediction = model.predict(feedback_vector)[0]

    print("\nFeedback:")
    print(feedback)

    print("Predicted Sentiment:", prediction)