
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import re
from collections import Counter

from mining.predictor import predict_sentiment
from mining.dataset_analysis import get_dataset_statistics
from mining.keywords import get_top_keywords


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)
from database.init_db import *
import seed_demo_accounts
app.secret_key = "edumine-development-key"

DATABASE = "database/edumine.db"


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


# --------------------------------------------------
# Text analysis helpers for faculty feedback
# --------------------------------------------------

STOP_WORDS = {
    "the", "and", "for", "are", "was", "were", "with", "this",
    "that", "have", "has", "had", "but", "not", "you", "your",
    "our", "their", "they", "them", "his", "her", "its", "from",
    "into", "about", "there", "here", "when", "what", "which",
    "who", "will", "would", "could", "should", "can", "may",
    "might", "been", "being", "very", "more", "most", "much",
    "some", "such", "than", "then", "also", "only", "just",
    "because", "while", "where", "how", "all", "any", "too",
    "really", "good", "great", "nice", "excellent", "bad",
    "poor", "okay", "ok", "overall", "class", "classes",
    "teacher", "faculty", "student", "students", "subject",
    "teaching", "feedback", "lecture", "lectures"
}


def extract_common_terms(comments, limit=10):
    """Extract frequent meaningful words from a list of comments."""
    words = []

    for comment in comments:
        if not comment:
            continue

        tokens = re.findall(r"[a-zA-Z]{3,}", comment.lower())

        for word in tokens:
            if word not in STOP_WORDS:
                words.append(word)

    return Counter(words).most_common(limit)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Login page
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        connection = get_db_connection()

        if role == "student":
            user = connection.execute(
                """
                SELECT * FROM students
                WHERE email = ? AND password = ?
                """,
                (email, password)
            ).fetchone()

            connection.close()

            if user:
                session["user_id"] = user["id"]
                session["role"] = "student"
                session["name"] = user["name"]

                return redirect(url_for("student_dashboard"))

        elif role == "faculty":
            user = connection.execute(
                """
                SELECT * FROM faculty
                WHERE email = ? AND password = ?
                """,
                (email, password)
            ).fetchone()

            connection.close()

            if user:
                session["user_id"] = user["id"]
                session["role"] = "faculty"
                session["name"] = user["name"]

                return redirect(url_for("faculty_dashboard"))

        elif role == "admin":
            connection.close()

            # Demo credentials for the college project
            if (
                email == "admin@edumine.com"
                and password == "admin123"
            ):
                session["role"] = "admin"
                session["name"] = "Administrator"

                return redirect(url_for("admin_dashboard"))

        else:
            connection.close()

        return "Invalid login credentials"

    return render_template("login.html")



# --------------------------------------------------
# Submit feedback
# --------------------------------------------------


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    connection = get_db_connection()

    try:
        rows = connection.execute(
            "SELECT id, name, subject FROM faculty ORDER BY subject, name"
        ).fetchall()

        faculty_rows = [dict(row) for row in rows]
        subjects = sorted({
            row["subject"] for row in faculty_rows if row["subject"]
        })

        if request.method == "POST":
            subject = request.form.get("subject", "").strip()
            faculty_id = request.form.get("faculty_id", "").strip()
            rating_input = request.form.get("rating", "").strip()
            comment = request.form.get("comment", "").strip()

            if not subject or not faculty_id:
                return "Please select a subject and faculty.", 400

            if rating_input not in {"1", "2", "3", "4", "5"}:
                return "Please select a valid rating.", 400

            if not comment:
                return "Please enter a comment.", 400

            faculty = connection.execute(
                "SELECT id, subject FROM faculty WHERE id = ?",
                (faculty_id,)
            ).fetchone()

            if not faculty or faculty["subject"] != subject:
                return "Invalid faculty selection for this subject.", 400

            existing = connection.execute(
                """
                SELECT id FROM feedback
                WHERE student_id = ? AND subject = ? AND faculty_id = ?
                """,
                (session["user_id"], subject, faculty_id)
            ).fetchone()

            if existing:
                return (
                    "You have already submitted feedback for this subject "
                    "and faculty.",
                    409
                )

            rating = int(rating_input)
            sentiment = predict_sentiment(comment)

            connection.execute(
                """
                INSERT INTO feedback
                (student_id, faculty_id, subject, rating, comment, sentiment)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session["user_id"],
                    faculty_id,
                    subject,
                    rating,
                    comment,
                    sentiment
                )
            )
            connection.commit()

            return render_template(
                "feedback_success.html",
                message="Feedback submitted successfully!"
            )

        return render_template(
            "feedback.html",
            subjects=subjects,
            faculty_rows=faculty_rows
        )

    finally:
        connection.close()


# --------------------------------------------------
# Student dashboard
# --------------------------------------------------

@app.route("/student")
def student_dashboard():

    if session.get("role") != "student":
        return redirect(url_for("login"))

    return render_template(
        "student_dashboard.html",
        name=session.get("name")
    )


# --------------------------------------------------
# Student feedback history
# --------------------------------------------------

@app.route("/student/history")
def student_feedback_history():

    if session.get("role") != "student":
        return redirect(url_for("login"))

    connection = get_db_connection()

    feedback_records = connection.execute(
        """
        SELECT
            feedback.id,
            feedback.subject,
            feedback.rating,
            feedback.comment,
            feedback.sentiment,
            feedback.created_at,
            faculty.name AS faculty_name
        FROM feedback
        LEFT JOIN faculty
            ON feedback.faculty_id = faculty.id
        WHERE feedback.student_id = ?
        ORDER BY feedback.created_at DESC
        """,
        (session.get("user_id"),)
    ).fetchall()

    connection.close()

    return render_template(
        "student_history.html",
        name=session.get("name"),
        feedback_records=feedback_records
    )


# --------------------------------------------------
# Faculty dashboard and admin analytics
# --------------------------------------------------

@app.route("/faculty")
def faculty_dashboard():

    role = session.get("role")

    if role not in ("faculty", "admin"):
        return redirect(url_for("login"))

    connection = get_db_connection()

    # Overall preloaded dataset analytics are used by the admin.
    # They are displayed only on the admin analytics view.
    if role == "admin":
        dataset_stats = get_dataset_statistics()
        dataset_keywords = get_top_keywords()
    else:
        dataset_stats = {
            "total_feedback": 0,
            "average_rating": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0
        }
        dataset_keywords = []

    # Get live feedback records for the appropriate role.
    if role == "faculty":
        feedback_records = connection.execute(
            """
            SELECT
                id,
                subject,
                rating,
                comment,
                sentiment,
                created_at
            FROM feedback
            WHERE faculty_id = ?
            ORDER BY created_at DESC
            """,
            (session.get("user_id"),)
        ).fetchall()

    else:
        feedback_records = connection.execute(
            """
            SELECT
                id,
                subject,
                rating,
                comment,
                sentiment,
                created_at
            FROM feedback
            ORDER BY created_at DESC
            """
        ).fetchall()

    # Live statistics from the same set of feedback records.
    live_stats = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            COALESCE(AVG(rating), 0) AS average,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'positive'
                THEN 1 ELSE 0 END
            ) AS positive,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'neutral'
                THEN 1 ELSE 0 END
            ) AS neutral,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'negative'
                THEN 1 ELSE 0 END
            ) AS negative
        FROM feedback
        """
        if role == "admin"
        else
        """
        SELECT
            COUNT(*) AS total,
            COALESCE(AVG(rating), 0) AS average,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'positive'
                THEN 1 ELSE 0 END
            ) AS positive,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'neutral'
                THEN 1 ELSE 0 END
            ) AS neutral,
            SUM(
                CASE WHEN LOWER(COALESCE(sentiment, '')) = 'negative'
                THEN 1 ELSE 0 END
            ) AS negative
        FROM feedback
        WHERE faculty_id = ?
        """,
        () if role == "admin" else (session.get("user_id"),)
    ).fetchone()

    connection.close()

    # Build faculty-specific keyword and complaint-term lists.
    positive_comments = [
        row["comment"] for row in feedback_records
        if (row["sentiment"] or "").lower() == "positive"
    ]

    negative_comments = [
        row["comment"] for row in feedback_records
        if (row["sentiment"] or "").lower() == "negative"
    ]

    faculty_keywords = extract_common_terms(
        [row["comment"] for row in feedback_records]
    )

    complaint_keywords = extract_common_terms(negative_comments)

    return render_template(
        "faculty_dashboard.html",
        name=session.get("name"),
        role=role,

        # Overall preloaded dataset analytics for admin
        total_feedback=dataset_stats["total_feedback"],
        average_rating=dataset_stats["average_rating"],
        positive_count=dataset_stats["positive_count"],
        neutral_count=dataset_stats["neutral_count"],
        negative_count=dataset_stats["negative_count"],
        keywords=dataset_keywords,

        # Live statistics for the current faculty or all feedback for admin
        live_total=live_stats["total"] or 0,
        live_average=round(live_stats["average"] or 0, 2),
        live_positive=live_stats["positive"] or 0,
        live_neutral=live_stats["neutral"] or 0,
        live_negative=live_stats["negative"] or 0,

        # Faculty-specific chart data
        faculty_keywords=faculty_keywords,
        complaint_keywords=complaint_keywords
    )


# --------------------------------------------------
# Faculty anonymous feedback
# --------------------------------------------------

@app.route("/faculty/feedback")
def faculty_feedback():

    if session.get("role") != "faculty":
        return redirect(url_for("login"))

    connection = get_db_connection()

    # IMPORTANT: only this faculty member's feedback is returned.
    feedback_records = connection.execute(
        """
        SELECT
            id,
            subject,
            rating,
            comment,
            sentiment,
            created_at
        FROM feedback
        WHERE faculty_id = ?
        ORDER BY created_at DESC
        """,
        (session.get("user_id"),)
    ).fetchall()

    connection.close()

    return render_template(
        "faculty_feedback.html",
        name=session.get("name"),
        feedback_records=feedback_records
    )



# --------------------------------------------------
# Admin dashboard + Faculty Performance Insights
# --------------------------------------------------

@app.route("/admin")
def admin_dashboard():

    if session.get("role") != "admin":
        return redirect(url_for("login"))

    connection = get_db_connection()

    try:
        student_count = connection.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        faculty_count = connection.execute(
            "SELECT COUNT(*) FROM faculty"
        ).fetchone()[0]

        feedback_count = connection.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        # Get faculty accounts without exposing student identities.
        faculty_rows = connection.execute(
            """
            SELECT id, name, subject
            FROM faculty
            ORDER BY name
            """
        ).fetchall()

        faculty_insights = []

        MIN_FEEDBACK_FOR_ASSESSMENT = 5

        for faculty in faculty_rows:

            records = connection.execute(
                """
                SELECT rating, sentiment, comment
                FROM feedback
                WHERE faculty_id = ?
                """,
                (faculty["id"],)
            ).fetchall()

            total = len(records)

            ratings = [
                float(row["rating"])
                for row in records
                if row["rating"] is not None
            ]

            average_rating = (
                round(sum(ratings) / len(ratings), 2)
                if ratings else 0
            )

            positive = sum(
                1 for row in records
                if (row["sentiment"] or "").lower() == "positive"
            )

            neutral = sum(
                1 for row in records
                if (row["sentiment"] or "").lower() == "neutral"
            )

            negative = sum(
                1 for row in records
                if (row["sentiment"] or "").lower() == "negative"
            )

            # Percentages are based on all submissions.
            positive_pct = round(positive / total * 100, 1) if total else 0
            neutral_pct = round(neutral / total * 100, 1) if total else 0
            negative_pct = round(negative / total * 100, 1) if total else 0

            # Extract frequent terms only from negative comments.
            negative_comments = [
                row["comment"]
                for row in records
                if (row["sentiment"] or "").lower() == "negative"
                and row["comment"]
            ]

            complaint_terms = extract_common_terms(
                negative_comments,
                limit=5
            )

            # Avoid drawing conclusions from a very small sample.
            if total < MIN_FEEDBACK_FOR_ASSESSMENT:
                performance_status = "Not enough feedback yet"

            elif average_rating < 3.5 or negative_pct >= 30:
                performance_status = "Potential areas for improvement"

            else:
                performance_status = "Generally positive feedback"

            faculty_insights.append({
                "name": faculty["name"],
                "subject": faculty["subject"],
                "total": total,
                "average_rating": average_rating,
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "positive_pct": positive_pct,
                "neutral_pct": neutral_pct,
                "negative_pct": negative_pct,
                "complaint_terms": complaint_terms,
                "performance_status": performance_status
            })

    finally:
        connection.close()

    return render_template(
        "admin_dashboard.html",
        name=session.get("name"),
        student_count=student_count,
        faculty_count=faculty_count,
        feedback_count=feedback_count,
        faculty_insights=faculty_insights
    )


# --------------------------------------------------
# Logout
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)