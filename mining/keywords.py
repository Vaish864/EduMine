import pandas as pd
from collections import Counter
import re


DATASET = "data/cleaned_student_feedback.csv"


def get_top_keywords(limit=10):

    df = pd.read_csv(DATASET)

    text = " ".join(
        df["StudentComments"].astype(str)
    ).lower()

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    stop_words = {
    "the", "and", "was", "were", "this",
    "that", "with", "for", "are", "very",
    "good", "teacher", "class", "classes",
    "not", "but", "have", "has", "had",
    "they", "their", "from", "about",
    "you", "our", "all", "can", "too",
    "more", "also", "its", "been",
    "she", "he", "his", "her", "him",
    "sir", "madam", "faculty", "students",
    "student", "well", "best","one", "like", "much", "person", "aiub"
}

    words = [
        word for word in words
        if word not in stop_words
    ]

    keyword_counts = Counter(words)

    return keyword_counts.most_common(limit)


if __name__ == "__main__":

    keywords = get_top_keywords()

    print("Top Keywords")
    print("=" * 40)

    for word, count in keywords:
        print(word, ":", count)