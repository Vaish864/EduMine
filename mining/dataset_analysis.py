import pandas as pd
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_student_feedback.csv"
)


def get_dataset_statistics():

    df = pd.read_csv(DATASET)

    total_feedback = len(df)

    sentiment_counts = df["Sentiment"].value_counts()

    positive_count = sentiment_counts.get("positive", 0)
    neutral_count = sentiment_counts.get("neutral", 0)
    negative_count = sentiment_counts.get("negative", 0)

    average_rating = df["Rating"].mean()

    return {
        "total_feedback": total_feedback,
        "average_rating": round(average_rating, 2),
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count
    }


if __name__ == "__main__":

    statistics = get_dataset_statistics()

    print("EduMine Dataset Statistics")
    print("=" * 40)

    for key, value in statistics.items():
        print(key, ":", value)