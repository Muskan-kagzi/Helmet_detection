
import sqlite3
import os

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "database.db"
)


def get_connection():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS rider_violations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_filename TEXT NOT NULL,

            rider_id INTEGER NOT NULL,

            helmet_status TEXT NOT NULL,

            number_plate TEXT,

            number_plate_status TEXT,

            bike_confidence REAL,

            ocr_confidence REAL,

            output_image TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    conn.commit()
    conn.close()


def save_violation(
    image_filename,
    rider,
    output_image
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO rider_violations (
            image_filename,
            rider_id,
            helmet_status,
            number_plate,
            number_plate_status,
            bike_confidence,
            ocr_confidence,
            output_image
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_filename,
        rider["rider_id"],
        rider["helmet_status"],
        rider["number_plate"],
        rider["number_plate_status"],
        rider["bike_confidence"],
        rider["ocr_confidence"],
        output_image
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM rider_violations
        ORDER BY id DESC
    """).fetchall()

    history = [
        dict(row)
        for row in rows
    ]

    conn.close()

    return history


def get_violation_count():

    conn = get_connection()

    row = conn.execute("""
        SELECT COUNT(*) AS total
        FROM rider_violations
    """).fetchone()

    conn.close()

    return row["total"]
