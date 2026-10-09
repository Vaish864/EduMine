
import pandas as pd

DATA_PATH = "data/cleaned_student_feedback.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["StudentComments", "Sentiment", "Rating"]).copy()
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])

df = df[df["Sentiment"].isin(["positive", "neutral", "negative"])]

print("\nSentiment counts:")
print(df["Sentiment"].value_counts())

print("\nRating summary by sentiment:")
print(
    df.groupby("Sentiment")["Rating"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\nRating distribution for each sentiment (%):")
distribution = pd.crosstab(
    df["Sentiment"],
    df["Rating"],
    normalize="index"
).mul(100).round(1)

print(distribution)

print("\nExamples where rating and sentiment may disagree:")

examples = {
    "positive": df[
        (df["Sentiment"] == "positive") & (df["Rating"] <= 2)
    ],
    "neutral": df[
        (df["Sentiment"] == "neutral") &
        ((df["Rating"] <= 2) | (df["Rating"] >= 4))
    ],
    "negative": df[
        (df["Sentiment"] == "negative") & (df["Rating"] >= 4)
    ],
}

for label, group in examples.items():
    print(f"\n--- {label.upper()} label ---")
    if group.empty:
        print("No examples found.")
        continue

    for row in group.sample(
        n=min(5, len(group)),
        random_state=42
    ).itertuples(index=False):
        print(f"Rating: {row.Rating} | Sentiment: {row.Sentiment}")
        print(f"Comment: {str(row.StudentComments)[:250]}\n")