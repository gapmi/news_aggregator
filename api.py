from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from collections import defaultdict
from datetime import datetime
import time
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
logger = logging.getLogger("news.collector")


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


def init_articles_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
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
    init_articles_table()

    scheduler.add_job(
        start_collection_job,
        "interval",
        hours=1,
        id="news_collection_job",
        replace_existing=True,
    )
    scheduler.start()

    yield

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


class PageMeta(BaseModel):
    limit: int
    offset: int
    total: int
    hasNext: bool


class LineageEdgeResponse(BaseModel):
    edgeId: int
    parentRunId: int
    childRunId: int
    parentClusterId: int
    childClusterId: int
    sourceNodeId: str
    targetNodeId: str
    centroidSimilarity: float = Field(ge=-1.0, le=1.0)
    articleOverlapRatio: float = Field(ge=0.0, le=1.0)
    articleOverlapCount: int = Field(ge=0)
    parentSize: int = Field(gt=0)
    childSize: int = Field(gt=0)
    score: float = Field(ge=0.0, le=1.0)
    matchedAt: datetime


class LineageEdgesResponse(BaseModel):
    items: list[LineageEdgeResponse]
    page: PageMeta


class RunSummaryResponse(BaseModel):
    runId: int
    startedAt: datetime
    finishedAt: datetime | None
    status: str
    windowHours: int
    minClusterSize: int
    minSamples: int
    articleCount: int
    clusterCount: int
    noiseCount: int
    largestClusterSize: int
    largestClusterRatio: float
    noiseRatio: float
    maxPerSource: int | None
    parentLineageEdgeCount: int
    childLineageEdgeCount: int


class RunsResponse(BaseModel):
    items: list[RunSummaryResponse]
    page: PageMeta


class PipelineRunResponse(BaseModel):
    id: int
    jobType: str
    status: str
    startedAt: datetime
    finishedAt: datetime | None
    relatedRunId: int | None
    error: str | None
    meta: dict


class PipelineRunsResponse(BaseModel):
    items: list[PipelineRunResponse]
    page: PageMeta


class SankeyNodeResponse(BaseModel):
    id: str
    label: str | None
    runId: int
    clusterId: int
    clusterLabel: int
    size: int
    depth: int
    meta: dict


class SankeyLinkResponse(BaseModel):
    id: str
    edgeId: int
    source: str
    target: str
    value: float
    score: float
    overlapRatio: float
    overlapCount: int
    similarity: float


class SankeyStatsResponse(BaseModel):
    nodeCount: int
    linkCount: int
    runCount: int
    truncated: bool


class SankeyResponse(BaseModel):
    nodes: list[SankeyNodeResponse]
    links: list[SankeyLinkResponse]
    stats: SankeyStatsResponse


class EulerClusterResponse(BaseModel):
    id: str
    runId: int
    clusterId: int
    clusterLabel: int
    label: str | None
    size: int


class EulerOverlapResponse(BaseModel):
    count: int
    ratio: float
    parentCoverage: float
    childCoverage: float
    unionSize: int
    jaccard: float


class EulerMetricsResponse(BaseModel):
    similarity: float
    score: float


class EulerCirclesResponse(BaseModel):
    parentArea: int
    childArea: int
    intersectionArea: int


class EulerLabelsResponse(BaseModel):
    title: str
    subtitle: str
    explanation: str


class EulerPairDetailResponse(BaseModel):
    edgeId: int
    parent: EulerClusterResponse
    child: EulerClusterResponse
    overlap: EulerOverlapResponse
    metrics: EulerMetricsResponse
    circles: EulerCirclesResponse
    labels: EulerLabelsResponse


class GraphPositionHintResponse(BaseModel):
    lane: int
    rank: int
    x: float
    y: float


class GraphNodeResponse(BaseModel):
    id: str
    type: Literal["cluster"]
    runId: int
    clusterId: int
    clusterLabel: int
    label: str | None
    size: int
    group: str
    positionHint: GraphPositionHintResponse
    styleHints: dict
    meta: dict


class GraphEdgeResponse(BaseModel):
    id: str
    type: Literal["lineage"]
    edgeId: int
    source: str
    target: str
    label: str
    score: float
    similarity: float
    overlapRatio: float
    overlapCount: int
    styleHints: dict


class GraphGroupResponse(BaseModel):
    id: str
    type: Literal["run"]
    label: str
    runId: int
    nodeCount: int


class GraphStatsResponse(BaseModel):
    nodeCount: int
    edgeCount: int
    truncated: bool


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    groups: list[GraphGroupResponse]
    stats: GraphStatsResponse


class ArticlePreviewResponse(BaseModel):
    id: int
    title: str | None
    url: str | None
    published: datetime | None
    source: str | None


class ClusterDetailResponse(BaseModel):
    id: str
    clusterId: int
    runId: int
    clusterLabel: int
    displayName: str | None = None
    size: int
    representativeArticleId: int | None
    representativeTitle: str | None
    createdAt: datetime
    incomingEdgeCount: int
    outgoingEdgeCount: int
    nameShort: str | None = None
    nameTitle: str | None = None
    languageCode: str | None = None
    tags: list[str] | None = None
    concepts: list[str] | None = None
    articles: list[ArticlePreviewResponse]


class ClustersResponse(BaseModel):
    items: list[ClusterDetailResponse]
    page: PageMeta


def make_cluster_node_id(run_id: int, cluster_id: int) -> str:
    return f"run:{run_id}:cluster:{cluster_id}"


def resolve_cluster_display_name(
    *,
    cluster_id: int,
    tags: list[str] | None = None,
    name_title: str | None = None,
    name_short: str | None = None,
) -> str:
    if tags:
        cleaned_tags = [tag.strip() for tag in tags if tag and tag.strip()]
        if cleaned_tags:
            return ", ".join(cleaned_tags)

    if name_title and name_title.strip():
        return name_title.strip()

    if name_short and name_short.strip():
        return name_short.strip()

    return f"Cluster {cluster_id}"


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


@app.get("/v1/clustering/lineage/edges", response_model=LineageEdgesResponse)
def list_lineage_edges(
    parent_run_id: int | None = Query(None),
    child_run_id: int | None = Query(None),
    parent_cluster_id: int | None = Query(None),
    child_cluster_id: int | None = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    min_similarity: float | None = Query(None, ge=-1.0, le=1.0),
    min_overlap_ratio: float | None = Query(None, ge=0.0, le=1.0),
    min_overlap_count: int | None = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: Literal[
        "score_desc",
        "score_asc",
        "similarity_desc",
        "overlap_desc",
        "matched_at_desc",
    ] = Query("score_desc"),
):
    where_clauses = ["score >= %s"]
    params: list[object] = [min_score]

    if parent_run_id is not None:
        where_clauses.append("parent_run_id = %s")
        params.append(parent_run_id)

    if child_run_id is not None:
        where_clauses.append("child_run_id = %s")
        params.append(child_run_id)

    if parent_cluster_id is not None:
        where_clauses.append("parent_cluster_id = %s")
        params.append(parent_cluster_id)

    if child_cluster_id is not None:
        where_clauses.append("child_cluster_id = %s")
        params.append(child_cluster_id)

    if min_similarity is not None:
        where_clauses.append("centroid_similarity >= %s")
        params.append(min_similarity)

    if min_overlap_ratio is not None:
        where_clauses.append("article_overlap_ratio >= %s")
        params.append(min_overlap_ratio)

    if min_overlap_count is not None:
        where_clauses.append("article_overlap_count >= %s")
        params.append(min_overlap_count)

    order_sql = {
        "score_desc": "score DESC, article_overlap_count DESC, id ASC",
        "score_asc": "score ASC, id ASC",
        "similarity_desc": "centroid_similarity DESC, score DESC, id ASC",
        "overlap_desc": "article_overlap_count DESC, score DESC, id ASC",
        "matched_at_desc": "matched_at DESC, id DESC",
    }[sort]

    where_sql = " AND ".join(where_clauses)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM cluster_lineage
                WHERE {where_sql}
                """,
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT
                    id,
                    parent_run_id,
                    child_run_id,
                    parent_cluster_id,
                    child_cluster_id,
                    centroid_similarity,
                    article_overlap_ratio,
                    article_overlap_count,
                    parent_size,
                    child_size,
                    score,
                    matched_at
                FROM cluster_lineage
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = [
        LineageEdgeResponse(
            edgeId=row["id"],
            parentRunId=row["parent_run_id"],
            childRunId=row["child_run_id"],
            parentClusterId=row["parent_cluster_id"],
            childClusterId=row["child_cluster_id"],
            sourceNodeId=make_cluster_node_id(
                row["parent_run_id"],
                row["parent_cluster_id"],
            ),
            targetNodeId=make_cluster_node_id(
                row["child_run_id"],
                row["child_cluster_id"],
            ),
            centroidSimilarity=row["centroid_similarity"],
            articleOverlapRatio=row["article_overlap_ratio"],
            articleOverlapCount=row["article_overlap_count"],
            parentSize=row["parent_size"],
            childSize=row["child_size"],
            score=row["score"],
            matchedAt=row["matched_at"],
        )
        for row in rows
    ]

    return LineageEdgesResponse(
        items=items,
        page=PageMeta(
            limit=limit,
            offset=offset,
            total=total,
            hasNext=offset + limit < total,
        ),
    )


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

    if collection_status.get("running"):
        logger.warning("collection skipped: already running")
        return

    run_logs = []
    collection_status["running"] = True
    run_started_at = time.perf_counter()

    logger.warning("collection started")

    try:
        from config import Config
        from scrapers import RSSScraper, HTMLScraper
        from processors import deduplicate
        from storage.pg_storage import PGStorage

        cfg = Config()
        all_articles = []

        rss_sources, html_sources = load_sources_from_db()
        logger.warning(
            "sources loaded: rss=%s, html=%s",
            len(rss_sources),
            len(html_sources),
        )

        for src in rss_sources:
            source_started_at = time.perf_counter()
            try:
                logger.warning("rss start: %s", src.name)
                scraper = RSSScraper(
                    src,
                    timeout=cfg.request_timeout,
                    user_agent=cfg.user_agent,
                )
                items = scraper.fetch()
                elapsed = time.perf_counter() - source_started_at
                logger.warning(
                    "rss done: %s, items=%s, duration=%.2fs",
                    src.name,
                    len(items),
                    elapsed,
                )
                all_articles.extend(items)
            except Exception:
                elapsed = time.perf_counter() - source_started_at
                logger.exception(
                    "rss failed: %s, duration=%.2fs",
                    src.name,
                    elapsed,
                )

        for src in html_sources:
            source_started_at = time.perf_counter()
            try:
                logger.warning("html start: %s", src.name)
                scraper = HTMLScraper(
                    src,
                    timeout=cfg.request_timeout,
                    user_agent=cfg.user_agent,
                )
                items = scraper.fetch()
                elapsed = time.perf_counter() - source_started_at
                logger.warning(
                    "html done: %s, items=%s, duration=%.2fs",
                    src.name,
                    len(items),
                    elapsed,
                )
                all_articles.extend(items)
            except Exception:
                elapsed = time.perf_counter() - source_started_at
                logger.exception(
                    "html failed: %s, duration=%.2fs",
                    src.name,
                    elapsed,
                )

        logger.warning("before deduplicate: %s", len(all_articles))
        all_articles = deduplicate(all_articles)
        logger.warning("after deduplicate: %s", len(all_articles))

        storage = PGStorage()
        logger.warning("storage.save start")
        storage.save(all_articles)
        logger.warning("storage.save done, saved_articles=%s", len(all_articles))

    except Exception as e:
        import traceback

        run_logs.append(f"ERROR: {e}")
        run_logs.extend(traceback.format_exc().splitlines()[-30:])
        logger.exception("collection failed")

    finally:
        collection_status["running"] = False
        collection_status["last_run"] = datetime.now().isoformat()
        total_elapsed = time.perf_counter() - run_started_at
        logger.warning("collection finished, duration=%.2fs", total_elapsed)


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


@app.get("/v1/clustering/runs", response_model=RunsResponse)
def list_clustering_runs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order: Literal["asc", "desc"] = Query("desc"),
):
    where_clauses = ["1=1"]
    params: list[object] = []

    if status is not None:
        where_clauses.append("r.status = %s")
        params.append(status)

    where_sql = " AND ".join(where_clauses)
    order_sql = "r.started_at ASC, r.id ASC" if order == "asc" else "r.started_at DESC, r.id DESC"

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM clustering_runs r
                WHERE {where_sql}
                """,
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                WITH parent_counts AS (
                    SELECT parent_run_id AS run_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY parent_run_id
                ),
                child_counts AS (
                    SELECT child_run_id AS run_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY child_run_id
                )
                SELECT
                    r.id,
                    r.started_at,
                    r.finished_at,
                    r.status,
                    r.window_hours,
                    r.min_cluster_size,
                    r.min_samples,
                    r.article_count,
                    r.cluster_count,
                    r.noise_count,
                    COALESCE(r.largest_cluster_size, 0) AS largest_cluster_size,
                    COALESCE(r.largest_cluster_ratio, 0.0) AS largest_cluster_ratio,
                    COALESCE(r.noise_ratio, 0.0) AS noise_ratio,
                    r.max_per_source,
                    COALESCE(pc.edge_count, 0) AS parent_lineage_edge_count,
                    COALESCE(cc.edge_count, 0) AS child_lineage_edge_count
                FROM clustering_runs r
                LEFT JOIN parent_counts pc ON pc.run_id = r.id
                LEFT JOIN child_counts cc ON cc.run_id = r.id
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = [
        RunSummaryResponse(
            runId=row["id"],
            startedAt=row["started_at"],
            finishedAt=row["finished_at"],
            status=row["status"],
            windowHours=row["window_hours"],
            minClusterSize=row["min_cluster_size"],
            minSamples=row["min_samples"],
            articleCount=row["article_count"],
            clusterCount=row["cluster_count"],
            noiseCount=row["noise_count"],
            largestClusterSize=row["largest_cluster_size"],
            largestClusterRatio=row["largest_cluster_ratio"],
            noiseRatio=row["noise_ratio"],
            maxPerSource=row["max_per_source"],
            parentLineageEdgeCount=row["parent_lineage_edge_count"],
            childLineageEdgeCount=row["child_lineage_edge_count"],
        )
        for row in rows
    ]

    return RunsResponse(
        items=items,
        page=PageMeta(
            limit=limit,
            offset=offset,
            total=total,
            hasNext=offset + limit < total,
        ),
    )


@app.get("/v1/clustering/views/sankey", response_model=SankeyResponse)
def get_sankey_view(
    start_run_id: int = Query(..., ge=1),
    end_run_id: int = Query(..., ge=1),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    min_similarity: float | None = Query(None, ge=-1.0, le=1.0),
    min_overlap_ratio: float | None = Query(None, ge=0.0, le=1.0),
    min_overlap_count: int | None = Query(None, ge=0),
    link_value: Literal["overlap_count", "score", "child_size", "parent_size"] = Query("overlap_count"),
):
    if start_run_id >= end_run_id:
        raise HTTPException(
            status_code=422,
            detail="start_run_id must be less than end_run_id",
        )

    where_clauses = [
        "cl.parent_run_id >= %s",
        "cl.child_run_id <= %s",
        "cl.score >= %s",
    ]
    params: list[object] = [start_run_id, end_run_id, min_score]

    if min_similarity is not None:
        where_clauses.append("cl.centroid_similarity >= %s")
        params.append(min_similarity)

    if min_overlap_ratio is not None:
        where_clauses.append("cl.article_overlap_ratio >= %s")
        params.append(min_overlap_ratio)

    if min_overlap_count is not None:
        where_clauses.append("cl.article_overlap_count >= %s")
        params.append(min_overlap_count)

    where_sql = " AND ".join(where_clauses)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    cl.id AS edge_id,
                    cl.parent_run_id,
                    cl.child_run_id,
                    cl.parent_cluster_id,
                    cl.child_cluster_id,
                    cl.centroid_similarity,
                    cl.article_overlap_ratio,
                    cl.article_overlap_count,
                    cl.parent_size,
                    cl.child_size,
                    cl.score,

                    pc.label AS parent_label,
                    pc.size AS parent_cluster_size,
                    pc.representative_article_id AS parent_representative_article_id,
                    pc.representative_title AS parent_representative_title,
                    pcn.name_short AS parent_name_short,
                    pcn.name_title AS parent_name_title,
                    pcn.tags AS parent_tags,

                    cc.label AS child_label,
                    cc.size AS child_cluster_size,
                    cc.representative_article_id AS child_representative_article_id,
                    cc.representative_title AS child_representative_title,
                    ccn.name_short AS child_name_short,
                    ccn.name_title AS child_name_title,
                    ccn.tags AS child_tags
                FROM cluster_lineage cl
                JOIN clusters pc ON pc.id = cl.parent_cluster_id
                JOIN clusters cc ON cc.id = cl.child_cluster_id
                LEFT JOIN cluster_names pcn ON pcn.cluster_id = pc.id
                LEFT JOIN cluster_names ccn ON ccn.cluster_id = cc.id
                WHERE {where_sql}
                ORDER BY cl.parent_run_id ASC, cl.child_run_id ASC, cl.score DESC, cl.id ASC
                """,
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    node_map: dict[str, SankeyNodeResponse] = {}
    links: list[SankeyLinkResponse] = []

    for row in rows:
        source_id = make_cluster_node_id(row["parent_run_id"], row["parent_cluster_id"])
        target_id = make_cluster_node_id(row["child_run_id"], row["child_cluster_id"])

        parent_label_text = resolve_cluster_display_name(
            cluster_id=row["parent_cluster_id"],
            tags=row["parent_tags"],
            name_title=row["parent_name_title"],
            name_short=row["parent_name_short"],
        )
        child_label_text = resolve_cluster_display_name(
            cluster_id=row["child_cluster_id"],
            tags=row["child_tags"],
            name_title=row["child_name_title"],
            name_short=row["child_name_short"],
        )

        if source_id not in node_map:
            node_map[source_id] = SankeyNodeResponse(
                id=source_id,
                label=parent_label_text,
                runId=row["parent_run_id"],
                clusterId=row["parent_cluster_id"],
                clusterLabel=row["parent_label"],
                size=row["parent_cluster_size"],
                depth=row["parent_run_id"] - start_run_id,
                meta={
                    "representativeArticleId": row["parent_representative_article_id"],
                    "nameShort": row["parent_name_short"],
                    "nameTitle": row["parent_name_title"],
                    "tags": row["parent_tags"],
                },
            )

        if target_id not in node_map:
            node_map[target_id] = SankeyNodeResponse(
                id=target_id,
                label=child_label_text,
                runId=row["child_run_id"],
                clusterId=row["child_cluster_id"],
                clusterLabel=row["child_label"],
                size=row["child_cluster_size"],
                depth=row["child_run_id"] - start_run_id,
                meta={
                    "representativeArticleId": row["child_representative_article_id"],
                    "nameShort": row["child_name_short"],
                    "nameTitle": row["child_name_title"],
                    "tags": row["child_tags"],
                },
            )

        if link_value == "score":
            value = float(row["score"])
        elif link_value == "child_size":
            value = float(row["child_size"])
        elif link_value == "parent_size":
            value = float(row["parent_size"])
        else:
            value = float(row["article_overlap_count"])

        links.append(
            SankeyLinkResponse(
                id=f"edge:{row['edge_id']}",
                edgeId=row["edge_id"],
                source=source_id,
                target=target_id,
                value=value,
                score=row["score"],
                overlapRatio=row["article_overlap_ratio"],
                overlapCount=row["article_overlap_count"],
                similarity=row["centroid_similarity"],
            )
        )

    run_ids = {
        node.runId
        for node in node_map.values()
    }

    return SankeyResponse(
        nodes=list(node_map.values()),
        links=links,
        stats=SankeyStatsResponse(
            nodeCount=len(node_map),
            linkCount=len(links),
            runCount=len(run_ids),
            truncated=False,
        ),
    )


@app.get("/v1/clustering/views/euler/pair/{edge_id}", response_model=EulerPairDetailResponse)
def get_euler_pair_detail(edge_id: int):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    cl.id AS edge_id,
                    cl.parent_run_id,
                    cl.child_run_id,
                    cl.parent_cluster_id,
                    cl.child_cluster_id,
                    cl.centroid_similarity,
                    cl.article_overlap_ratio,
                    cl.article_overlap_count,
                    cl.parent_size,
                    cl.child_size,
                    cl.score,

                    pc.label AS parent_label,
                    pc.size AS parent_cluster_size,
                    pc.representative_article_id AS parent_representative_article_id,
                    pc.representative_title AS parent_representative_title,
                    pcn.name_short AS parent_name_short,
                    pcn.name_title AS parent_name_title,
                    pcn.tags AS parent_tags,

                    cc.label AS child_label,
                    cc.size AS child_cluster_size,
                    cc.representative_article_id AS child_representative_article_id,
                    cc.representative_title AS child_representative_title,
                    ccn.name_short AS child_name_short,
                    ccn.name_title AS child_name_title,
                    ccn.tags AS child_tags
                FROM cluster_lineage cl
                JOIN clusters pc ON pc.id = cl.parent_cluster_id
                JOIN clusters cc ON cc.id = cl.child_cluster_id
                LEFT JOIN cluster_names pcn ON pcn.cluster_id = pc.id
                LEFT JOIN cluster_names ccn ON ccn.cluster_id = cc.id
                WHERE cl.id = %s
                LIMIT 1
                """,
                (edge_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Lineage edge not found")

    parent_size = row["parent_size"]
    child_size = row["child_size"]
    overlap_count = row["article_overlap_count"]

    parent_coverage = overlap_count / parent_size if parent_size > 0 else 0.0
    child_coverage = overlap_count / child_size if child_size > 0 else 0.0
    union_size = parent_size + child_size - overlap_count
    jaccard = overlap_count / union_size if union_size > 0 else 0.0

    parent_title = resolve_cluster_display_name(
        cluster_id=row["parent_cluster_id"],
        tags=row["parent_tags"],
        name_title=row["parent_name_title"],
        name_short=row["parent_name_short"],
    )
    child_title = resolve_cluster_display_name(
        cluster_id=row["child_cluster_id"],
        tags=row["child_tags"],
        name_title=row["child_name_title"],
        name_short=row["child_name_short"],
    )

    return EulerPairDetailResponse(
        edgeId=row["edge_id"],
        parent=EulerClusterResponse(
            id=make_cluster_node_id(row["parent_run_id"], row["parent_cluster_id"]),
            runId=row["parent_run_id"],
            clusterId=row["parent_cluster_id"],
            clusterLabel=row["parent_label"],
            label=parent_title,
            size=parent_size,
        ),
        child=EulerClusterResponse(
            id=make_cluster_node_id(row["child_run_id"], row["child_cluster_id"]),
            runId=row["child_run_id"],
            clusterId=row["child_cluster_id"],
            clusterLabel=row["child_label"],
            label=child_title,
            size=child_size,
        ),
        overlap=EulerOverlapResponse(
            count=overlap_count,
            ratio=row["article_overlap_ratio"],
            parentCoverage=round(parent_coverage, 6),
            childCoverage=round(child_coverage, 6),
            unionSize=union_size,
            jaccard=round(jaccard, 6),
        ),
        metrics=EulerMetricsResponse(
            similarity=row["centroid_similarity"],
            score=row["score"],
        ),
        circles=EulerCirclesResponse(
            parentArea=parent_size,
            childArea=child_size,
            intersectionArea=overlap_count,
        ),
        labels=EulerLabelsResponse(
            title=f"{parent_title} → {child_title}",
            subtitle=f"{overlap_count} shared articles · score {row['score']:.2f}",
            explanation=(
                f"Child keeps {parent_coverage * 100:.1f}% of parent articles "
                f"and adds {child_size - overlap_count} new articles."
            ),
        ),
    )

@app.get("/v1/clustering/views/graph", response_model=GraphResponse)
def get_graph_view(
    start_run_id: int = Query(..., ge=1),
    end_run_id: int = Query(..., ge=1),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    min_similarity: float | None = Query(None, ge=-1.0, le=1.0),
    min_overlap_ratio: float | None = Query(None, ge=0.0, le=1.0),
    min_overlap_count: int | None = Query(None, ge=0),
    max_nodes: int = Query(1000, ge=1, le=5000),
    max_edges: int = Query(3000, ge=1, le=10000),
):
    if start_run_id >= end_run_id:
        raise HTTPException(
            status_code=422,
            detail="start_run_id must be less than end_run_id",
        )

    where_clauses = [
        "cl.parent_run_id >= %s",
        "cl.child_run_id <= %s",
        "cl.score >= %s",
    ]
    params: list[object] = [start_run_id, end_run_id, min_score]

    if min_similarity is not None:
        where_clauses.append("cl.centroid_similarity >= %s")
        params.append(min_similarity)

    if min_overlap_ratio is not None:
        where_clauses.append("cl.article_overlap_ratio >= %s")
        params.append(min_overlap_ratio)

    if min_overlap_count is not None:
        where_clauses.append("cl.article_overlap_count >= %s")
        params.append(min_overlap_count)

    where_sql = " AND ".join(where_clauses)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    cl.id AS edge_id,
                    cl.parent_run_id,
                    cl.child_run_id,
                    cl.parent_cluster_id,
                    cl.child_cluster_id,
                    cl.centroid_similarity,
                    cl.article_overlap_ratio,
                    cl.article_overlap_count,
                    cl.parent_size,
                    cl.child_size,
                    cl.score,

                    pc.label AS parent_label,
                    pc.size AS parent_cluster_size,
                    pc.representative_article_id AS parent_representative_article_id,
                    pc.representative_title AS parent_representative_title,
                    pcn.name_short AS parent_name_short,
                    pcn.name_title AS parent_name_title,
                    pcn.tags AS parent_tags,

                    cc.label AS child_label,
                    cc.size AS child_cluster_size,
                    cc.representative_article_id AS child_representative_article_id,
                    cc.representative_title AS child_representative_title,
                    ccn.name_short AS child_name_short,
                    ccn.name_title AS child_name_title,
                    ccn.tags AS child_tags
                FROM cluster_lineage cl
                JOIN clusters pc ON pc.id = cl.parent_cluster_id
                JOIN clusters cc ON cc.id = cl.child_cluster_id
                LEFT JOIN cluster_names pcn ON pcn.cluster_id = pc.id
                LEFT JOIN cluster_names ccn ON ccn.cluster_id = cc.id
                WHERE {where_sql}
                ORDER BY cl.parent_run_id ASC, cl.child_run_id ASC, cl.score DESC, cl.id ASC
                LIMIT %s
                """,
                params + [max_edges + 1],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    truncated = len(rows) > max_edges
    rows = rows[:max_edges]

    node_map: dict[str, GraphNodeResponse] = {}
    edges: list[GraphEdgeResponse] = []
    run_node_counts: dict[int, int] = {}

    def add_node(
        *,
        run_id: int,
        cluster_id: int,
        cluster_label: int,
        size: int,
        representative_article_id: int | None,
        representative_title: str | None,
        name_short: str | None,
        name_title: str | None,
        tags: list[str] | None,
    ) -> None:
        node_id = make_cluster_node_id(run_id, cluster_id)
        if node_id in node_map:
            return

        display_name = resolve_cluster_display_name(
            cluster_id=cluster_id,
            tags=tags,
            name_title=name_title,
            name_short=name_short,
        )

        lane = run_id - start_run_id
        rank = run_node_counts.get(run_id, 0)
        run_node_counts[run_id] = rank + 1

        radius = max(8.0, min(28.0, 8.0 + size ** 0.5))
        weight = min(1.0, size / 100.0)

        node_map[node_id] = GraphNodeResponse(
            id=node_id,
            type="cluster",
            runId=run_id,
            clusterId=cluster_id,
            clusterLabel=cluster_label,
            label=display_name,
            size=size,
            group=f"run:{run_id}",
            positionHint=GraphPositionHintResponse(
                lane=lane,
                rank=rank,
                x=lane * 420,
                y=rank * 90,
            ),
            styleHints={
                "radius": radius,
                "weight": weight,
                "colorKey": f"run:{run_id}",
            },
            meta={
                "representativeArticleId": representative_article_id,
                "representativeTitle": representative_title,
                "nameShort": name_short,
                "nameTitle": name_title,
                "tags": tags,
                "displayName": display_name,
            },
        )

    for row in rows:
        add_node(
            run_id=row["parent_run_id"],
            cluster_id=row["parent_cluster_id"],
            cluster_label=row["parent_label"],
            size=row["parent_cluster_size"],
            representative_article_id=row["parent_representative_article_id"],
            representative_title=row["parent_representative_title"],
            name_short=row["parent_name_short"],
            name_title=row["parent_name_title"],
            tags=row["parent_tags"],
        )

        add_node(
            run_id=row["child_run_id"],
            cluster_id=row["child_cluster_id"],
            cluster_label=row["child_label"],
            size=row["child_cluster_size"],
            representative_article_id=row["child_representative_article_id"],
            representative_title=row["child_representative_title"],
            name_short=row["child_name_short"],
            name_title=row["child_name_title"],
            tags=row["child_tags"],
        )

        source_id = make_cluster_node_id(row["parent_run_id"], row["parent_cluster_id"])
        target_id = make_cluster_node_id(row["child_run_id"], row["child_cluster_id"])

        width = max(1.0, min(6.0, 1.0 + row["score"] * 5.0))
        opacity = max(0.25, min(1.0, row["score"]))

        edges.append(
            GraphEdgeResponse(
                id=f"edge:{row['edge_id']}",
                type="lineage",
                edgeId=row["edge_id"],
                source=source_id,
                target=target_id,
                label=f"score {row['score']:.2f} · overlap {row['article_overlap_count']}",
                score=row["score"],
                similarity=row["centroid_similarity"],
                overlapRatio=row["article_overlap_ratio"],
                overlapCount=row["article_overlap_count"],
                styleHints={
                    "width": width,
                    "opacity": opacity,
                },
            )
        )

    if len(node_map) > max_nodes:
        truncated = True
        allowed_node_ids = set(list(node_map.keys())[:max_nodes])
        node_map = {
            node_id: node
            for node_id, node in node_map.items()
            if node_id in allowed_node_ids
        }
        edges = [
            edge
            for edge in edges
            if edge.source in allowed_node_ids and edge.target in allowed_node_ids
        ]

    groups = [
        GraphGroupResponse(
            id=f"run:{run_id}",
            type="run",
            label=f"Run {run_id}",
            runId=run_id,
            nodeCount=sum(1 for node in node_map.values() if node.runId == run_id),
        )
        for run_id in range(start_run_id, end_run_id + 1)
    ]

    return GraphResponse(
        nodes=list(node_map.values()),
        edges=edges,
        groups=groups,
        stats=GraphStatsResponse(
            nodeCount=len(node_map),
            edgeCount=len(edges),
            truncated=truncated,
        ),
    )

@app.get("/v1/clustering/clusters/{cluster_id}", response_model=ClusterDetailResponse)
def get_cluster_detail_v1(
    cluster_id: int,
    include_articles: bool = Query(False),
    articles_limit: int = Query(30, ge=1, le=200),
):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.run_id,
                    c.label,
                    c.size,
                    c.representative_article_id,
                    c.representative_title,
                    c.created_at,
                    COALESCE(in_edges.edge_count, 0) AS incoming_edge_count,
                    COALESCE(out_edges.edge_count, 0) AS outgoing_edge_count,
                    cn.name_short,
                    cn.name_title,
                    cn.language_code,
                    cn.tags,
                    cn.concepts
                FROM clusters c
                LEFT JOIN (
                    SELECT child_cluster_id AS cluster_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY child_cluster_id
                ) in_edges ON in_edges.cluster_id = c.id
                LEFT JOIN (
                    SELECT parent_cluster_id AS cluster_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY parent_cluster_id
                ) out_edges ON out_edges.cluster_id = c.id
                LEFT JOIN cluster_names cn ON cn.cluster_id = c.id
                WHERE c.id = %s
                LIMIT 1
                """,
                (cluster_id,),
            )
            cluster = cur.fetchone()

            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            articles: list[dict] = []
            if include_articles:
                cur.execute(
                    """
                    SELECT
                        a.id,
                        a.title,
                        a.url,
                        a.published,
                        a.source
                    FROM cluster_articles ca
                    JOIN articles a ON a.id = ca.article_id
                    WHERE ca.cluster_id = %s
                    ORDER BY a.published DESC NULLS LAST, a.id DESC
                    LIMIT %s
                    """,
                    (cluster_id, articles_limit),
                )
                articles = cur.fetchall()
    finally:
        conn.close()

    display_name = resolve_cluster_display_name(
        cluster_id=cluster["id"],
        tags=cluster["tags"],
        name_title=cluster["name_title"],
        name_short=cluster["name_short"],
    )

    return ClusterDetailResponse(
        id=make_cluster_node_id(cluster["run_id"], cluster["id"]),
        clusterId=cluster["id"],
        runId=cluster["run_id"],
        clusterLabel=cluster["label"],
        displayName=display_name,
        size=cluster["size"],
        representativeArticleId=cluster["representative_article_id"],
        representativeTitle=cluster["representative_title"],
        createdAt=cluster["created_at"],
        incomingEdgeCount=cluster["incoming_edge_count"],
        outgoingEdgeCount=cluster["outgoing_edge_count"],
        nameShort=cluster["name_short"],
        nameTitle=cluster["name_title"],
        languageCode=cluster["language_code"],
        tags=cluster["tags"],
        concepts=cluster["concepts"],
        articles=[
            ArticlePreviewResponse(
                id=row["id"],
                title=row["title"],
                url=row["url"],
                published=row["published"],
                source=row["source"],
            )
            for row in articles
        ],
    )


@app.get("/v1/clustering/clusters", response_model=ClustersResponse)
def list_clusters_v1(
    run_id: int | None = Query(None),
    min_size: int | None = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    where_clauses = ["1=1"]
    params: list[object] = []

    if run_id is not None:
        where_clauses.append("c.run_id = %s")
        params.append(run_id)

    if min_size is not None:
        where_clauses.append("c.size >= %s")
        params.append(min_size)

    where_sql = " AND ".join(where_clauses)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM clusters c
                WHERE {where_sql}
                """,
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT
                    c.id,
                    c.run_id,
                    c.label,
                    c.size,
                    c.representative_article_id,
                    c.representative_title,
                    c.created_at,
                    COALESCE(in_edges.edge_count, 0) AS incoming_edge_count,
                    COALESCE(out_edges.edge_count, 0) AS outgoing_edge_count,
                    cn.name_short,
                    cn.name_title,
                    cn.language_code,
                    cn.tags,
                    cn.concepts
                FROM clusters c
                LEFT JOIN (
                    SELECT child_cluster_id AS cluster_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY child_cluster_id
                ) in_edges ON in_edges.cluster_id = c.id
                LEFT JOIN (
                    SELECT parent_cluster_id AS cluster_id, COUNT(*)::int AS edge_count
                    FROM cluster_lineage
                    GROUP BY parent_cluster_id
                ) out_edges ON out_edges.cluster_id = c.id
                LEFT JOIN cluster_names cn ON cn.cluster_id = c.id
                WHERE {where_sql}
                ORDER BY c.size DESC, c.id DESC
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = [
        ClusterDetailResponse(
            id=make_cluster_node_id(row["run_id"], row["id"]),
            clusterId=row["id"],
            runId=row["run_id"],
            clusterLabel=row["label"],
            size=row["size"],
            representativeArticleId=row["representative_article_id"],
            representativeTitle=resolve_cluster_display_name(
                cluster_id=row["id"],
                tags=row["tags"],
                name_title=row["name_title"],
                name_short=row["name_short"],
            ),
            createdAt=row["created_at"],
            incomingEdgeCount=row["incoming_edge_count"],
            outgoingEdgeCount=row["outgoing_edge_count"],
            nameShort=row["name_short"],
            nameTitle=row["name_title"],
            languageCode=row["language_code"],
            tags=row["tags"],
            concepts=row["concepts"],
            articles=[],
        )
        for row in rows
    ]

    return ClustersResponse(
        items=items,
        page=PageMeta(
            limit=limit,
            offset=offset,
            total=total,
            hasNext=offset + limit < total,
        ),
    )


@app.get("/v1/clustering/pipeline/runs", response_model=PipelineRunsResponse)
def get_pipeline_runs(
    job_type: str | None = None,
    status: str | None = None,
    related_run_id: int | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    filters = []
    params: list = []

    if job_type is not None:
        filters.append("job_type = %s")
        params.append(job_type)

    if status is not None:
        filters.append("status = %s")
        params.append(status)

    if related_run_id is not None:
        filters.append("related_run_id = %s")
        params.append(related_run_id)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM pipeline_runs
                {where_sql}
                """,
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT
                    id,
                    job_type,
                    status,
                    started_at,
                    finished_at,
                    related_run_id,
                    error,
                    meta
                FROM pipeline_runs
                {where_sql}
                ORDER BY started_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()

        items = [
            PipelineRunResponse(
                id=row["id"],
                jobType=row["job_type"],
                status=row["status"],
                startedAt=row["started_at"],
                finishedAt=row["finished_at"],
                relatedRunId=row["related_run_id"],
                error=row["error"],
                meta=row["meta"] or {},
            )
            for row in rows
        ]

        return PipelineRunsResponse(
            items=items,
            page=PageMeta(
                limit=limit,
                offset=offset,
                total=total,
                hasNext=offset + limit < total,
            ),
        )
    finally:
        conn.close()