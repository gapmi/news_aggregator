from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from collections import defaultdict
import psycopg2
import psycopg2.extras
import os
import secrets
import threading
import logging
import time
from datetime import datetime

# --- Логгер для хранения логов запуска ---
run_logs: list[str] = []

class LogCapture(logging.Handler):
    def emit(self, record):
        run_logs.append(f"{datetime.now().strftime('%H:%M:%S')} [{record.levelname}] {record.getMessage()}")

log_capture = LogCapture()
logging.getLogger().addHandler(log_capture)

# --- Хранилище токенов ---
active_tokens: set[str] = set()

# --- FastAPI ---
app = FastAPI()
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Разрешаем запросы ото всех для теста
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "secret123")

# --- DB ---
def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd")
    )

def init_sources_table():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL CHECK (type IN ('rss', 'html'))
            )
        """)
    conn.commit()
    conn.close()
    if __name__ == "main":
        print("Waiting for database connection...")
        retries = 5
        while retries > 0:
            try:
                init_sources_table()
                print("Database initialized successfully!")
                break
            except Exception as e:
                print(f"Database not ready yet, retrying in 5s... ({retries} left)")
                time.sleep(5)
                retries -= 1
    init_sources_table()

# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in active_tokens:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials

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

# --- Публичные эндпоинты ---
@app.get("/articles")
def get_articles(
    source: str = Query(None),
    search: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=300),
):
    offset = (page - 1) * page_size

    conn = get_conn()
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
            SELECT *
            FROM articles
            WHERE {where_sql}
            ORDER BY published DESC NULLS LAST
            LIMIT %s OFFSET %s
        """
        cur.execute(articles_query, params + [page_size, offset])
        articles = cur.fetchall()

        article_ids = [article["id"] for article in articles]

        scales_by_article = defaultdict(list)

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
                scales_by_article[row["article_id"]].append({
                    "scale_id": row["scale_id"],
                    "score": row["score"],
                    "strength": row["strength"],
                })

        for article in articles:
            article["semantic_scales"] = scales_by_article.get(article["id"], [])

    conn.close()

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return {
        "articles": articles,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }

@app.get("/sources")
def get_sources_public():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM articles ORDER BY source")
        sources = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"sources": sources}

# --- Admin: статистика ---
@app.get("/admin/stats")
def get_stats(_: str = Depends(require_auth)):
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT source, COUNT(*) as count
            FROM articles
            GROUP BY source
            ORDER BY count DESC
        """)
        stats = cur.fetchall()
        cur.execute("SELECT COUNT(*) as total FROM articles")
        total = cur.fetchone()["total"]
    conn.close()
    return {"stats": stats, "total": total}

# --- Admin: логи ---
@app.get("/admin/logs")
def get_logs(_: str = Depends(require_auth)):
    return {"logs": run_logs[-100:]}  # последние 100 строк

# --- Admin: запуск сбора ---
collection_status = {"running": False, "last_run": None}

def run_collection():
    global run_logs
    run_logs = []  # очищаем логи перед запуском
    collection_status["running"] = True
    try:
        from config import Config
        from scrapers import RSSScraper, HTMLScraper
        from processors import deduplicate
        from storage.pg_storage import PGStorage

        cfg = Config()
        all_articles = []

        for src in cfg.rss_sources:
            scraper = RSSScraper(src, timeout=cfg.request_timeout, user_agent=cfg.user_agent)
            all_articles.extend(scraper.fetch())

        for src in cfg.html_sources:
            scraper = HTMLScraper(src, timeout=cfg.request_timeout, user_agent=cfg.user_agent)
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

# --- Admin: управление источниками ---
class SourceCreate(BaseModel):
    name: str
    url: str
    type: str  # 'rss' or 'html'

@app.get("/admin/sources")
def list_sources(_: str = Depends(require_auth)):
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM sources ORDER BY type, name")
        sources = cur.fetchall()
    conn.close()
    return {"sources": sources}

@app.post("/admin/sources")
def add_source(body: SourceCreate, _: str = Depends(require_auth)):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sources (name, url, type) VALUES (%s, %s, %s)",
                (body.name, body.url, body.type)
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Source already exists")
    finally:
        conn.close()
    return {"ok": True}

@app.delete("/admin/sources/{source_id}")
def delete_source(source_id: int, _: str = Depends(require_auth)):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))
    conn.commit()
    conn.close()
    return {"ok": True}