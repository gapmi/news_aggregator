import psycopg2
import os
from scrapers.base import Article

class PGStorage:
    def init(self):
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                self.conn = psycopg2.connect(db_url)
            else:
                self.conn = psycopg2.connect(
                    host=os.getenv("DB_HOST", "db"),
                    port=os.getenv("DB_PORT", "5432"),
                    dbname=os.getenv("DB_NAME", "news_db"),
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "_qg9_P__1WWpeffd")
                )
            self._create_table()

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