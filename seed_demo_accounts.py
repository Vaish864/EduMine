
import sqlite3

DATABASE = "database/edumine.db"

students = [
    ("Priya Sharma", "student2@edumine.com", "student123"),
    ("Rahul Patil", "student3@edumine.com", "student123"),
    ("Sneha Joshi", "student4@edumine.com", "student123"),
    ("Aditya Deshmukh", "student5@edumine.com", "student123"),
]

faculty = [
    ("Dr. Aditi Sharma", "faculty.cloud@edumine.com", "faculty123", "Cloud Computing"),
    ("Prof. Rohan Verma", "faculty.ml@edumine.com", "faculty123", "Machine Learning"),
    ("Dr. Neha Kulkarni", "faculty.dbms@edumine.com", "faculty123", "Database Management"),
    ("Prof. Arjun Deshmukh", "faculty.android@edumine.com", "faculty123", "Android Development"),
]

connection = sqlite3.connect(DATABASE)

connection.executemany(
    """
    INSERT OR IGNORE INTO students (name, email, password)
    VALUES (?, ?, ?)
    """,
    students
)

connection.executemany(
    """
    INSERT OR IGNORE INTO faculty (name, email, password, subject)
    VALUES (?, ?, ?, ?)
    """,
    faculty
)

connection.commit()

student_count = connection.execute(
    "SELECT COUNT(*) FROM students"
).fetchone()[0]

faculty_count = connection.execute(
    "SELECT COUNT(*) FROM faculty"
).fetchone()[0]

print("Demo accounts added successfully!")
print("Total students:", student_count)
print("Total faculty:", faculty_count)

connection.close()