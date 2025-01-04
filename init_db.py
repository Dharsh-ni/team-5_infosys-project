import sqlite3

def init_db():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industry_data (
            industry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            industry_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            description TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_data (
            company_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            industry_id INTEGER,
            financial_data TEXT,
            FOREIGN KEY (industry_id) REFERENCES industry_data (industry_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            industry_id INTEGER,
            report_date DATE NOT NULL,
            report_data TEXT,
            FOREIGN KEY (industry_id) REFERENCES industry_data (industry_id)
        )
    """)

    conn.commit()  # Save changes
    conn.close()   # Close the connection
    print("Database 'user_data.db' initialized and tables created successfully.")
# Run the function to initialize the database
init_db()
