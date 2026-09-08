import logging
import os
import re
import time
from datetime import datetime, timezone

import psycopg2
from dateutil import parser as date_parser
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_batch

from scrapers.base import Article


logger = logging.getLogger("news.collector")

INGEST_EMBEDDING_BATCH_SIZE = 1


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
                    connect_timeout=15,
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

        cleaned = re.sub(r"^[^\d]{1,20},\s*", "", cleaned).strip()

        try:
            dt = date_parser.parse(cleaned, fuzzy=True)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def save(self, articles: list[Article]):
        if not articles:
            logger.warning("storage.save: no articles")
            return

        saved_rows = []

        try:
            with self.conn.cursor() as cur:
                for article in articles:
                    published = self._normalize_published(article.published)

                    if article.published and published is None:
                        logger.warning(
                            "failed to parse published=%r source=%r url=%r",
                            article.published,
                            article.source,
                            article.url,
                        )

                    cur.execute(
                        """
                        INSERT INTO articles (
                            title,
                            url,
                            published,
                            source,
                            rss_description
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE
                        SET
                            title = EXCLUDED.title,
                            published = EXCLUDED.published,
                            source = EXCLUDED.source,
                            rss_description = EXCLUDED.rss_description
                        RETURNING id, title, rss_description
                        """,
                        (
                            article.title,
                            article.url,
                            published,
                            article.source,
                            article.description,
                        ),
                    )

                    row = cur.fetchone()
                    saved_rows.append(
                        {
                            "id": row[0],
                            "title": row[1],
                            "description": row[2],
                        }
                    )

            logger.warning("storage.save prepared rows: %s", len(saved_rows))

            from processors.embeddings import ArticleText

            article_payload = [
                ArticleText(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"],
                )
                for row in saved_rows
                if row["title"] or row["description"]
            ]

            if not article_payload:
                self.conn.commit()
                logger.warning("storage.save: no usable text, saved without embeddings")
                return

            embedding_service = self._get_embedding_service()

            texts = [
                embedding_service.build_text(article)
                for article in article_payload
            ]

            logger.warning(
                "embedding step: start, payload_size=%s, batch_size=%s",
                len(article_payload),
                INGEST_EMBEDDING_BATCH_SIZE,
            )

            vectors = embedding_service.encode_batch(
                article_payload,
                batch_size=INGEST_EMBEDDING_BATCH_SIZE,
            )

            if len(vectors) != len(article_payload):
                raise ValueError(
                    "vectors/article mismatch: "
                    f"vectors={len(vectors)}, articles={len(article_payload)}"
                )

            update_rows = [
                (text, vector, article.id)
                for article, text, vector in zip(
                    article_payload,
                    texts,
                    vectors,
                )
                if vector is not None
            ]

            if len(update_rows) != len(article_payload):
                raise ValueError(
                    "some embeddings are missing: "
                    f"updated={len(update_rows)}, expected={len(article_payload)}"
                )

            with self.conn.cursor() as cur:
                execute_batch(
                    cur,
                    """
                    UPDATE articles
                    SET
                        embedding_text = %s,
                        embedding = %s
                    WHERE id = %s
                    """,
                    update_rows,
                    page_size=25,
                )

            self.conn.commit()

            logger.warning(
                "storage.save committed: articles=%s embeddings=%s",
                len(saved_rows),
                len(update_rows),
            )

        except Exception as e:
            logger.exception("storage.save failed; transaction rolled back")

            try:
                self.conn.rollback()
            except Exception:
                logger.exception("storage.save rollback failed")

            try:
                from api import run_logs

                run_logs.append(f"ERROR in storage.save: {e}")
            except Exception:
                pass

            raise

    def save_scales_for_articles(
        self,
        article_vectors: list[tuple[int, list[float]]],
    ):
        from processors.scales.service import ScaleEmbeddingService

        scale_service = ScaleEmbeddingService(
            self._get_embedding_service()
        )

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
                    INSERT INTO article_scales (
                        article_id,
                        scale_id,
                        score,
                        strength
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            article_id,
                            scale["scale_id"],
                            scale["score"],
                            scale["strength"],
                        )
                        for scale in scales
                    ],
                    page_size=20,
                )

                primary = max(
                    scales,
                    key=lambda scale: scale["strength"],
                )

                cur.execute(
                    """
                    UPDATE articles
                    SET primary_scale_id = %s
                    WHERE id = %s
                    """,
                    (primary["scale_id"], article_id),
                )