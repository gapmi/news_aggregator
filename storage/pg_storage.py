import logging
import os
import re
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from dateutil import parser as date_parser

from scrapers.base import Article

logger = logging.getLogger("news.collector")


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
            except Exception as e:
                logger.warning("waiting for DB connection failed: %s", e)
                time.sleep(5)

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
                    logger.warning(
                        "failed to parse published=%r source=%r url=%r",
                        a.published,
                        a.source,
                        a.url,
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
            logger.warning("storage.save finished with empty payload")
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

            # scales temporarily disabled: only update embeddings
            self.conn.commit()
            logger.warning("embeddings updated: ", len(update_rows))

        except Exception:
            logger.exception("embedding generation skipped due to error")
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