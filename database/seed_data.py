import sqlite3

DATABASE = "database/edumine.db"

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

# Demo student
cursor.execute("""
INSERT OR IGNORE INTO students
(name, email, password)
VALUES (?, ?, ?)
""", (
    "Demo Student",
    "student@edumine.com",
    "student123"
))

# Demo faculty
cursor.execute("""
INSERT OR IGNORE INTO faculty
(name, email, password, subject)
VALUES (?, ?, ?, ?)
""", (
    "Dr. Demo Faculty",
    "faculty@edumine.com",
    "faculty123",
    "Data Mining"
))

connection.commit()
connection.close()

print("Demo users created successfully!")
print()
print("Student:")
print("Email: student@edumine.com")
print("Password: student123")
print()
print("Faculty:")
print("Email: faculty@edumine.com")
print("Password: faculty123")