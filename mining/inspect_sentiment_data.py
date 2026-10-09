
import pandas as pd

df = pd.read_csv("data/cleaned_student_feedback.csv")
df = df.dropna(subset=["StudentComments", "Sentiment"]).copy()

df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
df = df[df["Sentiment"].isin(["positive", "neutral", "negative"])]

for label in ["neutral", "negative", "positive"]:
    print("\n" + "=" * 70)
    print(f"EXAMPLES LABELLED: {label.upper()}")
    print("=" * 70)

    sample = df[df["Sentiment"] == label].sample(
        n=min(10, (df["Sentiment"] == label).sum()),
        random_state=42
    )

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        print(f"\n{i}. {row.Sentiment}")
        print(str(row.StudentComments)[:500])