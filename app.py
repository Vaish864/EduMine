from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3

from mining.predictor import predict_sentiment
from mining.dataset_analysis import get_dataset_statistics
from mining.keywords import get_top_keywords

# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

# Secret key for sessions
app.secret_key = "edumine-development-key"


# --------------------------------------------------
# Database connection
# --------------------------------------------------

DATABASE = "database/edumine.db"


def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


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

                return redirect(
                    url_for("student_dashboard")
                )

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

                return redirect(
                    url_for("faculty_dashboard")
                )

        elif role == "admin":

            connection.close()

            # Simple admin login for college demonstration
            if (
                email == "admin@edumine.com"
                and password == "admin123"
            ):
                session["role"] = "admin"
                session["name"] = "Administrator"

                return redirect(
                    url_for("admin_dashboard")
                )

        return "Invalid login credentials"

    return render_template("login.html")
#feedback
@app.route("/feedback", methods=["GET", "POST"])
def feedback():

    if session.get("role") != "student":
        return redirect(url_for("login"))

    if request.method == "POST":

        subject = request.form["subject"]
        faculty_id = request.form["faculty_id"]
        rating = request.form["rating"]
        comment = request.form["comment"]
        sentiment = predict_sentiment(comment)

        connection = get_db_connection()

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
        connection.close()

        return "Feedback submitted successfully!"

    return render_template("feedback.html")
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
# Faculty dashboard
# --------------------------------------------------

@app.route("/faculty")
def faculty_dashboard():

    if session.get("role") != "faculty":
        return redirect(url_for("login"))

    dataset_stats = get_dataset_statistics()
    keywords = get_top_keywords()

    return render_template(
        "faculty_dashboard.html",
        name=session.get("name"),
        total_feedback=dataset_stats["total_feedback"],
        average_rating=dataset_stats["average_rating"],
        positive_count=dataset_stats["positive_count"],
        neutral_count=dataset_stats["neutral_count"],
        negative_count=dataset_stats["negative_count"],
        keywords=keywords,
        live_total=0,
        live_average=0,
        live_positive=0,
        live_neutral=0,
        live_negative=0
    )
# --------------------------------------------------
# Admin dashboard
# --------------------------------------------------

@app.route("/admin")
def admin_dashboard():

    if session.get("role") != "admin":
        return redirect(url_for("login"))

    return render_template(
        "admin_dashboard.html",
        name=session.get("name")
    )


# --------------------------------------------------
# Logout
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)