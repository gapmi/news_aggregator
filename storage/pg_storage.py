import os
import re
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from dateutil import parser as date_parser

from scrapers.base import Article


class PGStorage:
    def __init__(self):
        self.conn = None
        self.embedding_service = None

        while self.conn is None:
            try:
                self.conn = psycopg2.connect(
                    host=os.getenv("DB_HOST", "db"),
                    port=os.getenv("DB_PORT", "5432"),
                    dbname=os.getenv("DB_NAME", "news_db"),
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
                )
                register_vector(self.conn)
                self._create_table()
                print("Database initialized successfully!")
            except Exception as e:
                print(f"Waiting for DB... error: {e}")
                time.sleep(5)

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    url TEXT UNIQUE,
                    published TIMESTAMP,
                    source TEXT,
                    embedding vector(384)
                )
                """
            )
        self.conn.commit()

    def _get_embedding_service(self):
        if self.embedding_service is None:
            from processors.embeddings import EmbeddingService

            self.embedding_service = EmbeddingService()
        return self.embedding_service

    def _normalize_published(self, value):
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value

        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None

        cleaned = raw.replace("\u00a0", " ").strip()

        try:
            dt = date_parser.parse(cleaned, fuzzy=True)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            pass

        cleaned2 = re.sub(r"^[^\d]{1,20},\s*", "", cleaned).strip()

        try:
            dt = date_parser.parse(cleaned2, fuzzy=True)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def save(self, articles: list[Article]):
        saved_rows = []

        with self.conn.cursor() as cur:
            for a in articles:
                published = self._normalize_published(a.published)

                if a.published and published is None:
                    print(
                        f"WARNING: failed to parse published={a.published!r} "
                        f"source={a.source!r} url={a.url!r}"
                    )

                cur.execute(
                    """
                    INSERT INTO articles (title, url, published, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET title = EXCLUDED.title,
                        published = EXCLUDED.published,
                        source = EXCLUDED.source
                    RETURNING id, title
                    """,
                    (a.title, a.url, published, a.source),
                )

                row = cur.fetchone()
                saved_rows.append(
                    {
                        "id": row[0],
                        "title": row[1],
                    }
                )

        payload = [
            {"id": row["id"], "title": row["title"]}
            for row in saved_rows
            if row["title"]
        ]

        self.conn.commit()

        if not payload:
            return

        try:
            from processors.embeddings import ArticleText

            embedding_service = self._get_embedding_service()

            article_payload = [
                ArticleText(id=row["id"], title=row["title"], content=None)
                for row in payload
            ]

            vectors = embedding_service.encode_batch(article_payload, batch_size=8)

            update_rows = [
                (vector, article.id)
                for article, vector in zip(article_payload, vectors)
            ]

            with self.conn.cursor() as cur:
                execute_batch(
                    cur,
                    "UPDATE articles SET embedding = %s WHERE id = %s",
                    update_rows,
                    page_size=50,
                )

            id_with_vectors = [
                (article.id, vector)
                for article, vector in zip(article_payload, vectors)
            ]
            self.save_scales_for_articles(id_with_vectors)

            self.conn.commit()
            print(f"Embeddings updated: {len(update_rows)}")

        except Exception as e:
            print(f"Embedding generation skipped due to error: {e}")
            self.conn.rollback()

    def save_scales_for_articles(self, article_vectors: list[tuple[int, list[float]]]):
        from processors.scales.service import ScaleEmbeddingService

        scale_service = ScaleEmbeddingService(self._get_embedding_service())

        with self.conn.cursor() as cur:
            for article_id, vector in article_vectors:
                scales = scale_service.score_article_embedding(vector)

                cur.execute(
                    "DELETE FROM article_scales WHERE article_id = %s",
                    (article_id,),
                )

                execute_batch(
                    cur,
                    """
                    INSERT INTO article_scales (article_id, scale_id, score, strength)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (article_id, s["scale_id"], s["score"], s["strength"])
                        for s in scales
                    ],
                    page_size=20,
                )

                primary = max(scales, key=lambda s: s["strength"])
                cur.execute(
                    "UPDATE articles SET primary_scale_id = %s WHERE id = %s",
                    (primary["scale_id"], article_id),
                )