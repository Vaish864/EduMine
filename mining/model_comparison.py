import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load processed dataset
df = pd.read_csv(
    "data/processed_dataset.csv",
    encoding="latin1"
)

# Remove missing sentiment labels
df = df.dropna(subset=["sentiment"])

# Text and labels
X_text = df["cleaned_comments"]
y = df["sentiment"]


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Linear SVM": LinearSVC(),

    "Naive Bayes": MultinomialNB()
}


print("=" * 60)
print("EduMine - Sentiment Model Comparison")
print("=" * 60)


for name, model in models.items():

    print(f"\n{name}")
    print("-" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )