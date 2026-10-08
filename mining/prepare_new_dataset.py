import pandas as pd


print("=" * 60)
print("EduMine - Preparing New Sentiment Dataset")
print("=" * 60)

# Load the TSV dataset
df = pd.read_csv(
    "data/student_feedback.tsv",
    sep="\t"
)

print("\nOriginal records:", len(df))

# Keep only the columns we need
df = df[
    ["StudentComments", "Rating", "Sentiment"]
].copy()

# Remove duplicate comments
df = df.drop_duplicates(
    subset=["StudentComments"]
).copy()

print("Unique comments after removing duplicates:", len(df))

print("\nSentiment distribution after deduplication:")
print(df["Sentiment"].value_counts())

print("\nPercentage distribution:")
print(
    df["Sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Save cleaned dataset
df.to_csv(
    "data/cleaned_student_feedback.csv",
    index=False
)

print("\nCleaned dataset saved to:")
print("data/cleaned_student_feedback.csv")