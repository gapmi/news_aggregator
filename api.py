from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
import os
import secrets
import threading
import logging
import numpy as np


run_logs: list[str] = []


class LogCapture(logging.Handler):
    def emit(self, record):
        run_logs.append(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"[{record.levelname}] {record.getMessage()}"
        )


log_capture = LogCapture()
logging.getLogger().addHandler(log_capture)


active_tokens: set[str] = set()
collection_status = {"running": False, "last_run": None}
scheduler = BackgroundScheduler()
security = HTTPBearer()

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "secret123")

SEARCH_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_search_model = None
_search_model_lock = threading.Lock()


def get_search_model():
    global _search_model
    if _search_model is None:
        with _search_model_lock:
            if _search_model is None:
                _search_model = SentenceTransformer(SEARCH_MODEL_NAME)
    return _search_model


def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = psycopg2.connect(db_url)
        register_vector(conn)
        return conn

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
    )
    register_vector(conn)
    return conn


def init_sources_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL CHECK (type IN ('rss', 'html'))
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def load_sources_from_db():
    from config import RSSSource, HTMLSource

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, url, type
                FROM sources
                ORDER BY type, name
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    rss_sources = []
    html_sources = []

    for row in rows:
        if row["type"] == "rss":
            rss_sources.append(
                RSSSource(
                    name=row["name"],
                    url=row["url"],
                )
            )
        elif row["type"] == "html":
            if row["name"] == "Hacker News":
                html_sources.append(
                    HTMLSource(
                        name=row["name"],
                        url=row["url"],
                        article_selector=".athing",
                        title_selector=".titleline > a",
                        link_selector=".titleline > a",
                        description_selector="",
                    )
                )

    return rss_sources, html_sources


def start_collection_job():
    if collection_status["running"]:
        return

    thread = threading.Thread(target=run_collection, daemon=True)
    thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sources_table()

    scheduler.add_job(
        start_collection_job,
        "interval",
        hours=1,
        id="news_collection_job",
        replace_existing=True,
    )
    scheduler.start()

    try:
        yield
    finally:
        if scheduler.running:
            scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class SourceCreate(BaseModel):
    name: str
    url: str
    type: str


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in active_tokens:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


def enrich_articles_with_scales_and_badges(cur, articles):
    article_ids = [article["id"] for article in articles]
    scales_by_article = defaultdict(list)
    badges_by_article = defaultdict(list)

    if article_ids:
        scales_query = """
            SELECT article_id, scale_id, score, strength
            FROM article_scales
            WHERE article_id = ANY(%s)
            ORDER BY article_id, scale_id
        """
        cur.execute(scales_query, (article_ids,))
        scale_rows = cur.fetchall()

        for row in scale_rows:
            scales_by_article[row["article_id"]].append(
                {
                    "scale_id": row["scale_id"],
                    "score": row["score"],
                    "strength": row["strength"],
                }
            )

        badges_query = """
            SELECT article_id, tag_text, tag_kind, score
            FROM article_tags
            WHERE article_id = ANY(%s)
            ORDER BY article_id, score DESC NULLS LAST, tag_text
        """
        cur.execute(badges_query, (article_ids,))
        badge_rows = cur.fetchall()

        for row in badge_rows:
            badges_by_article[row["article_id"]].append(
                {
                    "tag": row["tag_text"],
                    "kind": row["tag_kind"],
                    "score": row["score"],
                }
            )

    for article in articles:
        article["semantic_scales"] = scales_by_article.get(article["id"], [])
        article["badges"] = badges_by_article.get(article["id"], [])


@app.post("/admin/login")
def login(body: LoginRequest):
    if body.username != ADMIN_USER or body.password != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_hex(32)
    active_tokens.add(token)
    return {"token": token}


@app.post("/admin/logout")
def logout(token: str = Depends(require_auth)):
    active_tokens.discard(token)
    return {"ok": True}


@app.get("/articles")
def get_articles(
    source: str = Query(None),
    search: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=300),
):
    offset = (page - 1) * page_size

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            where_clauses = ["1=1"]
            params = []

            if source:
                where_clauses.append("source = %s")
                params.append(source)

            if search:
                where_clauses.append("title ILIKE %s")
                params.append(f"%{search}%")

            where_sql = " AND ".join(where_clauses)

            count_query = f"""
                SELECT COUNT(*) AS total
                FROM articles
                WHERE {where_sql}
            """
            cur.execute(count_query, params)
            total = cur.fetchone()["total"]

            articles_query = f"""
                SELECT id, title, url, published, source, primary_scale_id
                FROM articles
                WHERE {where_sql}
                ORDER BY published DESC NULLS LAST
                LIMIT %s OFFSET %s
            """
            cur.execute(articles_query, params + [page_size, offset])
            articles = cur.fetchall()

            enrich_articles_with_scales_and_badges(cur, articles)

    finally:
        conn.close()

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return {
        "articles": articles,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/search")
def semantic_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
):
    model = get_search_model()
    query_embedding = model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, url, published, source, primary_scale_id
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, limit),
            )

            articles = cur.fetchall()
            enrich_articles_with_scales_and_badges(cur, articles)
    finally:
        conn.close()

    return {
        "query": q,
        "total": len(articles),
        "articles": articles,
    }


@app.get("/sources")
def get_sources_public():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, type
                FROM sources
                ORDER BY type, name
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    names = [row["name"] for row in rows]

    return {
        "sources": names,
        "items": rows,
    }


@app.get("/admin/stats")
def get_stats(_: str = Depends(require_auth)):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT source, COUNT(*) AS count
                FROM articles
                GROUP BY source
                ORDER BY count DESC
                """
            )
            stats = cur.fetchall()

            cur.execute("SELECT COUNT(*) AS total FROM articles")
            total = cur.fetchone()["total"]
    finally:
        conn.close()

    return {"stats": stats, "total": total}


@app.get("/admin/logs")
def get_logs(_: str = Depends(require_auth)):
    return {"logs": run_logs[-100:]}


@app.get("/topics")
def get_topics():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                WITH latest_run AS (
                    SELECT id, started_at, finished_at
                    FROM clustering_runs
                    WHERE status = 'success'
                    ORDER BY started_at DESC
                    LIMIT 1
                )
                SELECT
                    c.id AS cluster_id,
                    c.run_id,
                    c.label,
                    c.size,
                    c.representative_article_id,
                    c.representative_title,
                    lr.started_at,
                    lr.finished_at
                FROM clusters c
                JOIN latest_run lr ON lr.id = c.run_id
                ORDER BY c.size DESC, c.id DESC
                """
            )
            rows = cur.fetchall()

        return {
            "topics": rows,
            "total": len(rows),
        }
    finally:
        conn.close()


@app.get("/topics/{cluster_id}")
def get_topic_detail(cluster_id: int):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.id AS cluster_id,
                    c.run_id,
                    c.label,
                    c.size,
                    c.representative_article_id,
                    c.representative_title,
                    r.started_at,
                    r.finished_at
                FROM clusters c
                JOIN clustering_runs r ON r.id = c.run_id
                WHERE c.id = %s
                LIMIT 1
                """,
                (cluster_id,),
            )
            topic = cur.fetchone()

            if not topic:
                raise HTTPException(status_code=404, detail="Topic not found")

            cur.execute(
                """
                SELECT
                    a.id,
                    a.title,
                    a.url,
                    a.published,
                    a.source,
                    a.primary_scale_id
                FROM cluster_articles ca
                JOIN articles a ON a.id = ca.article_id
                WHERE ca.cluster_id = %s
                ORDER BY a.published DESC NULLS LAST, a.id DESC
                """,
                (cluster_id,),
            )
            articles = cur.fetchall()

            enrich_articles_with_scales_and_badges(cur, articles)

        return {
            "topic": topic,
            "articles": articles,
            "total": len(articles),
        }
    finally:
        conn.close()


def run_collection():
    global run_logs
    run_logs = []
    collection_status["running"] = True

    try:
        from config import Config
        from scrapers import RSSScraper, HTMLScraper
        from processors import deduplicate
        from storage.pg_storage import PGStorage

        cfg = Config()
        all_articles = []

        rss_sources, html_sources = load_sources_from_db()

        for src in rss_sources:
            scraper = RSSScraper(
                src,
                timeout=cfg.request_timeout,
                user_agent=cfg.user_agent,
            )
            all_articles.extend(scraper.fetch())

        for src in html_sources:
            scraper = HTMLScraper(
                src,
                timeout=cfg.request_timeout,
                user_agent=cfg.user_agent,
            )
            all_articles.extend(scraper.fetch())

        all_articles = deduplicate(all_articles)

        storage = PGStorage()
        storage.save(all_articles)

    except Exception as e:
        run_logs.append(f"ERROR: {e}")

    finally:
        collection_status["running"] = False
        collection_status["last_run"] = datetime.now().isoformat()


@app.post("/admin/collect")
def start_collection(_: str = Depends(require_auth)):
    if collection_status["running"]:
        raise HTTPException(status_code=409, detail="Collection already running")

    thread = threading.Thread(target=run_collection, daemon=True)
    thread.start()
    return {"ok": True, "message": "Collection started"}


@app.get("/admin/collect/status")
def collection_status_endpoint(_: str = Depends(require_auth)):
    return collection_status


@app.get("/admin/sources")
def list_sources(_: str = Depends(require_auth)):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sources ORDER BY type, name")
            sources = cur.fetchall()
    finally:
        conn.close()

    return {"sources": sources}


@app.post("/admin/sources")
def add_source(body: SourceCreate, _: str = Depends(require_auth)):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (name, url, type)
                VALUES (%s, %s, %s)
                """,
                (body.name, body.url, body.type),
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail="Source already exists")
    finally:
        conn.close()

    return {"ok": True}


@app.delete("/admin/sources/{source_id}")
def delete_source(source_id: int, _: str = Depends(require_auth)):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        conn.commit()
    finally:
        conn.close()

    return {"ok": True}