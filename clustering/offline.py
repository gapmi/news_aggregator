import os

import psycopg2
import psycopg2.extras


def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
    )


def main():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, embedding
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT 5
                """
            )
            rows = cur.fetchall()

            print(len(rows))
            if rows:
                print(rows[0]["id"])
                print(rows[0]["title"])
                print(type(rows[0]["embedding"]).__name__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()