import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Load cleaned dataset
df = pd.read_csv(
    "data/cleaned_student_feedback.csv"
)

# Use the same 100,000-record sample
from sklearn.model_selection import train_test_split

df_sample, _ = train_test_split(
    df,
    train_size=100000,
    random_state=42,
    stratify=df["Sentiment"]
)

X_text = df_sample["StudentComments"]
y = df_sample["Sentiment"]


# Create TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(X_text)


# Train model
model = LinearSVC(
    class_weight="balanced"
)

model.fit(X, y)


# New feedback examples
test_feedback = [
    "The teacher explains concepts very clearly and gives excellent practical examples.",
    "The teaching is confusing and the concepts are very difficult to understand.",
    "The teacher is okay and the overall experience is average.",
    "Excellent teacher, very helpful and supportive.",
    "The classes are boring and the teacher does not explain properly.",
    "The teacher gives good explanations but the workload is too high.",
    "I am satisfied with the teaching.",
    "The teacher is neither good nor bad."
]


print("=" * 60)
print("EduMine - New Feedback Prediction Test")
print("=" * 60)


for feedback in test_feedback:

    feedback_vector = vectorizer.transform(
        [feedback]
    )

    prediction = model.predict(
        feedback_vector
    )[0]

    print("\nFeedback:")
    print(feedback)

    print("Predicted Sentiment:", prediction)