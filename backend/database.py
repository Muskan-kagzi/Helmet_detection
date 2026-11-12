
import sqlite3
import os
import json

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "database.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            detections TEXT NOT NULL,
            output_image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_detection(filename, detections, output_image):
    conn = get_connection()

    conn.execute("""
        INSERT INTO detections
        (filename, detections, output_image)
        VALUES (?, ?, ?)
    """, (
        filename,
        json.dumps(detections),
        output_image
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()

    rows = conn.execute("""
        SELECT * FROM detections
        ORDER BY id DESC
    """).fetchall()

    history = [dict(row) for row in rows]

    conn.close()
    return history
