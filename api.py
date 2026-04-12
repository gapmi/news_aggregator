from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import psycopg2.extras
import os

app = FastAPI()

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # адрес Vite
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5433),
        dbname=os.getenv("DB_NAME", "news"),
        user=os.getenv("DB_USER", "myuser"),
        password=os.getenv("DB_PASSWORD", "mypassword"),
    )

@app.get("/articles")
def get_articles(
    source: str = Query(None, description="Фильтр по источнику"),
    search: str = Query(None, description="Поиск по заголовку"),
    limit: int = Query(200, le=200),
    offset: int = Query(0),
):
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        query = "SELECT * FROM articles WHERE 1=1"
        params = []

        if source:
            query += " AND source = %s"
            params.append(source)

        if search:
            query += " AND title ILIKE %s"
            params.append(f"%{search}%")

        query += " ORDER BY published DESC NULLS LAST LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cur.execute(query, params)
        articles = cur.fetchall()

    conn.close()
    return {"articles": articles, "total": len(articles)}

@app.get("/sources")
def get_sources():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM articles ORDER BY source")
        sources = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"sources": sources}