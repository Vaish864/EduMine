import sqlite3
import os


# Database path
DB_PATH = "database/edumine.db"


# Make sure database folder exists
os.makedirs("database", exist_ok=True)


# Connect to SQLite database
connection = sqlite3.connect(DB_PATH)

cursor = connection.cursor()


# --------------------------------------------------
# Students table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")


# --------------------------------------------------
# Faculty table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    subject TEXT NOT NULL
)
""")


# --------------------------------------------------
# Feedback table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    faculty_id INTEGER,
    subject TEXT NOT NULL,
    rating REAL NOT NULL,
    comment TEXT NOT NULL,
    sentiment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (student_id)
        REFERENCES students(id),

    FOREIGN KEY (faculty_id)
        REFERENCES faculty(id)
)
""")


# Save changes
connection.commit()

# Close database
connection.close()


print("=" * 60)
print("EduMine Database")
print("=" * 60)
print("\nDatabase created successfully!")
print("Location:", DB_PATH)
print("\nTables created:")
print("- students")
print("- faculty")
print("- feedback")