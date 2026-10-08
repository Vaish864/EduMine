import joblib
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


print("=" * 60)
print("EduMine - Sentiment Model Training")
print("=" * 60)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/cleaned_student_feedback.csv"
)

print("\nTotal unique feedback:", len(df))


# --------------------------------------------------
# 2. Take a manageable sample
# --------------------------------------------------

# Use 100,000 feedback records for model training
df_sample, _ = train_test_split(
    df,
    train_size=100000,
    random_state=42,
    stratify=df["Sentiment"]
)

print("Training dataset:", len(df_sample))


# --------------------------------------------------
# 3. Separate text and sentiment
# --------------------------------------------------

X_text = df_sample["StudentComments"]
y = df_sample["Sentiment"]


# --------------------------------------------------
# 4. Convert text into TF-IDF features
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(X_text)

print("TF-IDF matrix shape:", X.shape)


# --------------------------------------------------
# 5. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# --------------------------------------------------
# 6. Train Linear SVM
# --------------------------------------------------

print("\nTraining Linear SVM...")

model = LinearSVC(
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)
# --------------------------------------------------
# 8. Save trained model and TF-IDF vectorizer
# --------------------------------------------------

model_dir = "mining/saved_model"

os.makedirs(model_dir, exist_ok=True)

joblib.dump(
    model,
    os.path.join(model_dir, "sentiment_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(model_dir, "tfidf_vectorizer.pkl")
)

print("\nModel saved successfully!")
print("Saved model: mining/saved_model/sentiment_model.pkl")
print("Saved vectorizer: mining/saved_model/tfidf_vectorizer.pkl")