import psycopg2
import os
import time
from scrapers.base import Article

class PGStorage:
    def init(self):
        self.conn = None
        db_url = os.getenv("DATABASE_URL")
        
        # Цикл: пытаемся подключиться, пока не получится
        while self.conn is None:
            try:
                if db_url:
                    print("Connecting to DB via URL...")
                    self.conn = psycopg2.connect(db_url)
                else:
                    print("Connecting to DB via components...")
                    self.conn = psycopg2.connect(
                        host=os.getenv("DB_HOST", "db"),
                        port=os.getenv("DB_PORT", "5432"),
                        dbname=os.getenv("DB_NAME", "news_db"),
                        user=os.getenv("DB_USER", "postgres"),
                        password=os.getenv("DB_PASSWORD", "_qg9_P__1WWpeffd")
                    )
                self._create_table()
                print("Database initialized successfully!")
            except Exception as e:
                print(f"Database not ready yet ({e}). Retrying in 5 seconds...")
                time.sleep(5)

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    url TEXT UNIQUE,
                    published TIMESTAMP,
                    source TEXT
                )
            """)
        self.conn.commit()

    def save(self, articles: list[Article]):
        if not self.conn:
            print("No database connection, cannot save articles.")
            return
            
        with self.conn.cursor() as cur:
            for a in articles:
                cur.execute("""
                    INSERT INTO articles (title, url, published, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                """, (a.title, a.url, a.published, a.source))
        self.conn.commit()