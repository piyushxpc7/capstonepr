"""Run once to create and seed data/telecom_ops.db from sql/ scripts."""
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "telecom_ops.db")
SQL_DIR = os.path.join(BASE_DIR, "sql")

TABLES = [
    "network_towers",
    "network_outages",
    "tower_performance",
    "open_incidents",
    "customer_subscriptions",
    "billing_accounts",
    "billing_charges",
    "billing_credits",
    "billing_disputes",
]


def run():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for filename in ("01_schema.sql", "02_seed_data.sql"):
        path = os.path.join(SQL_DIR, filename)
        with open(path) as f:
            sql = f.read()
        cur.executescript(sql)
        print(f"Executed {filename}")

    conn.commit()

    print("\nRow counts:")
    for table in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:<30} {count}")

    conn.close()
    print(f"\nDatabase ready at: {DB_PATH}")


if __name__ == "__main__":
    run()
