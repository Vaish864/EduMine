import pandas as pd


# Load the TSV dataset
df = pd.read_csv(
    "data/student_feedback.tsv",
    sep="\t"
)

print("=" * 60)
print("Student Feedback Dataset Inspection")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate comments:")
print(df["StudentComments"].duplicated().sum())

print("\nSample comments:")
print(df[["StudentComments", "Rating", "Sentiment"]].head(10))