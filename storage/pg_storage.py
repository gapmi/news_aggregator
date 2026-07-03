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
        logger.warning("storage.save prepared rows: %s", len(saved_rows))
        self.conn.commit()

        if not payload:
            logger.warning("storage.save finished with empty payload")
            return

        logger.warning("embedding step: start, payload_size=%s", len(payload))

        try:
            from processors.embeddings import ArticleText
            logger.warning("embedding step: ArticleText imported")
        except Exception:
            logger.exception("embedding step failed: ArticleText import")
            return

        try:
            embedding_service = self._get_embedding_service()
            logger.warning("embedding step: service initialized: %s", type(embedding_service).__name__)
        except Exception:
            logger.exception("embedding step failed: service init")
            return

        try:
            article_payload = [
                ArticleText(id=row["id"], title=row["title"], content=None)
                for row in payload
            ]
            logger.warning(
                "embedding step: payload prepared, article_payload_size=%s, sample_id=%s",
                len(article_payload),
                article_payload[0].id if article_payload else None,
            )
        except Exception:
            logger.exception("embedding step failed: payload preparation")
            return

        try:
            vectors = embedding_service.encode_batch(article_payload, batch_size=8)
            logger.warning(
                "embedding step: encode ok, vectors_count=%s, first_vector_len=%s",
                len(vectors) if vectors is not None else None,
                len(vectors[0]) if vectors and vectors[0] is not None else None,
            )
        except Exception:
            logger.exception("embedding encode failed")
            self.conn.rollback()
            return

        try:
            if len(vectors) != len(article_payload):
                logger.warning(
                    "embedding step: vectors/article mismatch, vectors=%s, payload=%s",
                    len(vectors),
                    len(article_payload),
                )

            update_rows = [
                (vector, article.id)
                for article, vector in zip(article_payload, vectors)
                if vector is not None
            ]

            logger.warning(
                "embedding step: update_rows prepared, count=%s, sample_vector_len=%s",
                len(update_rows),
                len(update_rows[0][0]) if update_rows and update_rows[0][0] is not None else None,
            )
        except Exception as e:
            import traceback

            logger.exception("embedding generation skipped due to error")
            try:
                from api import run_logs
                run_logs.append(f"ERROR in embeddings: {e}")
                run_logs.extend(traceback.format_exc().splitlines()[-40:])
            except Exception:
                pass

            self.conn.rollback()
            return

        if not update_rows:
            logger.warning("embedding step: no update_rows, skip DB update")
            return

        try:
            with self.conn.cursor() as cur:
                execute_batch(
                    cur,
                    "UPDATE articles SET embedding = %s WHERE id = %s",
                    update_rows,
                    page_size=50,
                )
            logger.warning("embedding step: execute_batch done")
        except Exception:
            logger.exception("embedding db update failed")
            self.conn.rollback()
            return

        try:
            self.conn.commit()
            logger.warning("embeddings updated: %s", len(update_rows))
        except Exception:
            logger.exception("embedding commit failed")
            self.conn.rollback()
            return

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