import psycopg2
import os
import time

from scrapers.base import Article

class PGStorage:
    def init(self):
        self.conn = None
        db_url = os.getenv("DATABASE_URL")
        
        retries = 5 # Пытаемся подключиться 5 раз
        while retries > 0:
            try:
                if db_url:
                    print(f"Connecting to DB via URL... ({retries} retries left)")
                    self.conn = psycopg2.connect(db_url)
                else:
                    print(f"Connecting to DB via components... ({retries} retries left)")
                    self.conn = psycopg2.connect(
                        host=os.getenv("DB_HOST", "db"),
                        port=os.getenv("DB_PORT", "5432"),
                        dbname=os.getenv("DB_NAME", "news_db"),
                        user=os.getenv("DB_USER", "postgres"),
                        password=os.getenv("DB_PASSWORD", "_qg9_P__1WWpeffd")
                    )
                self._create_table()
                print("Database initialized successfully!")
                break # Если подключились — выходим из цикла
            except Exception as e:
                print(f"FAILED TO CONNECT TO DB: {e}. Retrying in 5s...")
                time.sleep(5)
                retries -= 1
        
        if not self.conn:
            print("CRITICAL: Could not connect to database after retries.")
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
        with self.conn.cursor() as cur:
            for a in articles:
                cur.execute("""
                    INSERT INTO articles (title, url, published, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                """, (a.title, a.url, a.published, a.source))
        self.conn.commit()