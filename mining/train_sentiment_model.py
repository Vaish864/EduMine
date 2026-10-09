
import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

print("=" * 65)
print("EduMine - Improved Sentiment Model Training")
print("=" * 65)

# --------------------------------------------------
# 1. Load the cleaned dataset
# --------------------------------------------------

DATA_PATH = "data/cleaned_student_feedback.csv"
MODEL_DIR = "mining/saved_model"

df = pd.read_csv(DATA_PATH)

# Keep only rows with usable text and sentiment labels.
df = df.dropna(subset=["StudentComments", "Sentiment"]).copy()

df["StudentComments"] = df["StudentComments"].astype(str).str.strip()
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()

df = df[df["StudentComments"] != ""]
df = df[df["Sentiment"].isin(["positive", "neutral", "negative"])]

print("\nUsable feedback records:", len(df))
print("\nFull dataset sentiment distribution:")
print(df["Sentiment"].value_counts())
print("\nFull dataset sentiment percentages:")
print((df["Sentiment"].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# 2. Select a stratified sample
# --------------------------------------------------

SAMPLE_SIZE = min(100000, len(df))

if df["Sentiment"].nunique() != 3:
    raise ValueError(
        "Expected positive, neutral, and negative labels. "
        "Check the dataset sentiment values."
    )

if SAMPLE_SIZE < len(df):
    df_sample, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        random_state=42,
        stratify=df["Sentiment"]
    )
else:
    df_sample = df.copy()

print("\nSample size:", len(df_sample))
print("\nSample sentiment distribution:")
print(df_sample["Sentiment"].value_counts())

# --------------------------------------------------
# 3. Split raw text BEFORE fitting TF-IDF
# --------------------------------------------------

X_text = df_sample["StudentComments"]
y = df_sample["Sentiment"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train_text))
print("Testing records:", len(X_test_text))

# --------------------------------------------------
# 4. Fit TF-IDF on training text only
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training feature matrix:", X_train.shape)
print("Testing feature matrix:", X_test.shape)

# --------------------------------------------------
# 5. Train the Linear SVM
# --------------------------------------------------

print("\nTraining LinearSVC...")

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

labels = ["negative", "neutral", "positive"]

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n" + "=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

print(f"\nAccuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=3
    )
)

print("\nConfusion matrix")
print("Rows = actual labels; columns = predicted labels")
print("Label order:", labels)
print(confusion_matrix(y_test, y_pred, labels=labels))

# --------------------------------------------------
# 7. Save the model and vectorizer
# --------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "sentiment_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

print("\nModel and vectorizer saved successfully.")
print("Model:", os.path.join(MODEL_DIR, "sentiment_model.pkl"))
print("Vectorizer:", os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("\nTraining completed.")