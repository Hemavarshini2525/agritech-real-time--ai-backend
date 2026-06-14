import sqlite3

DB_NAME = "advisory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_query TEXT,
            disease TEXT,
            fertilizer TEXT,
            irrigation TEXT,
            weather_info TEXT,
            advisory_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_advisory(data: dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history 
        (farmer_query, disease, fertilizer, irrigation, weather_info, advisory_text)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        data["farmer_query"],
        data["disease"],
        data["fertilizer"],
        data["irrigation"],
        data["weather_info"],
        data["advisory_text"]
    ))
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows