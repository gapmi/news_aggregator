from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import psycopg2.extras

from clustering.offline import parse_embedding

DEFAULT_TARGET_DURATION_SECONDS = 120
DEFAULT_MAX_TOPICS = 6
DEFAULT_EVIDENCE_ARTICLES_PER_TOPIC = 5
DEFAULT_MAX_NEW_TOPICS = 2
DEFAULT_MAX_REFRAMED_TOPICS = 2
DEFAULT_MAX_CONTINUING_TOPICS = 5

MIN_EDITORIAL_EVIDENCE_ARTICLES = 2
MIN_CENTRAL_EVIDENCE_SIMILARITY = 0.72
MAX_CANDIDATE_EVIDENCE_ARTICLES = 40


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalize_whitespace(value: str | None) -> str:
    return " ".join((value or "").split())


def _safe_topic_name(row: dict[str, Any]) -> str:
    return (
        row.get("name_title")
        or row.get("name_short")
        or row.get("representative_title")
        or f"topic-{row['id']}"
    )


def _public_topic_title(
    representative_article: dict[str, Any] | None,
    fallback_title: str | None,
) -> str:
    if representative_article is not None:
        title = _normalize_whitespace(
            representative_article.get("title")
        )

        if title:
            return title[:220]

    title = _normalize_whitespace(fallback_title)

    if title:
        return title[:220]

    return "Latest reports in the selected news topic"


def _normalize_topic_name(value: str) -> str:
    return " ".join(
        value.lower()
        .replace(",", " ")
        .replace(":", " ")
        .replace("-", " ")
        .split()
    )


def _normalize_source_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = _normalize_whitespace(value)

    if not normalized:
        return None

    return normalized[:160]


def _cosine_similarity(
    article_embedding: Any,
    centroid: Any,
) -> float | None:
    """
    Return cosine similarity between an article and a topic centroid.

    pgvector values arrive through psycopg2 as Vector objects. The existing
    parse_embedding() from offline.py handles Vector and list-like variants.
    """
    try:
        article_vector = parse_embedding(article_embedding)
        centroid_vector = parse_embedding(centroid)
    except Exception:
        return None

    if article_vector is None or centroid_vector is None:
        return None

    article_vector = np.asarray(article_vector, dtype=np.float32).reshape(-1)
    centroid_vector = np.asarray(centroid_vector, dtype=np.float32).reshape(-1)

    if article_vector.size == 0 or centroid_vector.size == 0:
        return None

    if article_vector.shape != centroid_vector.shape:
        return None

    if not np.isfinite(article_vector).all():
        return None

    if not np.isfinite(centroid_vector).all():
        return None

    article_norm = float(np.linalg.norm(article_vector))
    centroid_norm = float(np.linalg.norm(centroid_vector))

    if article_norm == 0.0 or centroid_norm == 0.0:
        return None

    similarity = float(
        np.dot(article_vector, centroid_vector)
        / (article_norm * centroid_norm)
    )

    return max(min(similarity, 1.0), -1.0)


def get_previous_completed_run_id(
    conn,
    child_run_id: int,
) -> int | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id
            FROM clustering_runs
            WHERE id < %s
              AND status IN ('success', 'completed', 'degraded')
              AND finished_at IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (child_run_id,),
        )
        row = cur.fetchone()

    return int(row["id"]) if row else None


def _load_run(conn, run_id: int) -> dict[str, Any]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                started_at,
                finished_at,
                status,
                window_hours,
                article_count,
                cluster_count,
                noise_count,
                noise_ratio,
                largest_cluster_size,
                largest_cluster_ratio,
                emergent_topic_count,
                emergent_assigned_article_count
            FROM clustering_runs
            WHERE id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"clustering run {run_id} was not found")

    if row["status"] not in {"success", "completed", "degraded"}:
        raise ValueError(
            f"clustering run {run_id} is not completed: "
            f"status={row['status']!r}"
        )

    if row["finished_at"] is None:
        raise ValueError(f"clustering run {run_id} has no finished_at value")

    return {
        "run_id": int(row["id"]),
        "status": row["status"],
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
        "window_hours": int(row["window_hours"]),
        "article_count": int(row["article_count"]),
        "cluster_count": int(row["cluster_count"]),
        "noise_count": int(row["noise_count"]),
        "noise_ratio": float(row["noise_ratio"] or 0.0),
        "largest_cluster_size": int(row["largest_cluster_size"] or 0),
        "largest_cluster_ratio": float(
            row["largest_cluster_ratio"] or 0.0
        ),
        "emergent_topic_count": int(row["emergent_topic_count"] or 0),
        "emergent_assigned_article_count": int(
            row["emergent_assigned_article_count"] or 0
        ),
    }


def _load_central_evidence_articles(
    conn,
    *,
    cluster_id: int,
    centroid: Any,
    representative_article_id: int | None,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Select source-attributed evidence near the topic centroid.

    The query first retrieves up to MAX_CANDIDATE_EVIDENCE_ARTICLES from the
    topic. Python computes cosine similarity because no distance is persisted
    in cluster_articles. The model sees only final source/title/date/URL data,
    not embeddings, similarity scores, cluster IDs, or other internal values.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                a.id,
                a.title,
                a.url,
                a.published,
                a.source,
                a.embedding
            FROM cluster_articles ca
            JOIN articles a
              ON a.id = ca.article_id
            WHERE ca.cluster_id = %s
              AND a.title IS NOT NULL
              AND BTRIM(a.title) <> ''
              AND a.embedding IS NOT NULL
            ORDER BY a.published DESC NULLS LAST, a.id DESC
            LIMIT %s
            """,
            (cluster_id, MAX_CANDIDATE_EVIDENCE_ARTICLES),
        )
        rows = cur.fetchall()

    candidates: list[dict[str, Any]] = []

    for row in rows:
        title = _normalize_whitespace(row["title"])
        source_name = _normalize_source_name(row["source"])

        if not title or source_name is None:
            continue

        similarity = _cosine_similarity(
            row["embedding"],
            centroid,
        )

        if similarity is None:
            continue

        candidates.append(
            {
                "article_id": int(row["id"]),
                "source_name": source_name,
                "title": title[:500],
                "published_at": _iso(row["published"]),
                "url": row["url"],
                "is_representative": bool(
                    representative_article_id is not None
                    and int(row["id"]) == representative_article_id
                ),
                "_similarity": similarity,
            }
        )

    candidates.sort(
        key=lambda article: (
            not article["is_representative"],
            -float(article["_similarity"]),
            article["published_at"] or "",
            article["article_id"],
        )
    )

    representative_article = next(
        (
            article
            for article in candidates
            if article["is_representative"]
        ),
        None,
    )

    central_candidates = [
        article
        for article in candidates
        if (
            article["is_representative"]
            or article["_similarity"] >= MIN_CENTRAL_EVIDENCE_SIMILARITY
        )
    ]

    selected: list[dict[str, Any]] = []
    seen_article_ids: set[int] = set()
    seen_title_keys: set[str] = set()
    seen_sources: set[str] = set()

    def add(article: dict[str, Any]) -> bool:
        article_id = int(article["article_id"])
        title_key = str(article["title"]).casefold()

        if article_id in seen_article_ids or title_key in seen_title_keys:
            return False

        cleaned = {
            key: value
            for key, value in article.items()
            if key != "_similarity"
        }

        selected.append(cleaned)
        seen_article_ids.add(article_id)
        seen_title_keys.add(title_key)
        seen_sources.add(str(article["source_name"]).casefold())
        return True

    if representative_article is not None:
        add(representative_article)

    for article in central_candidates:
        if len(selected) >= limit:
            break

        source_key = str(article["source_name"]).casefold()

        if source_key in seen_sources:
            continue

        add(article)

    for article in central_candidates:
        if len(selected) >= limit:
            break

        add(article)

    cleaned_representative = None

    if representative_article is not None:
        cleaned_representative = {
            key: value
            for key, value in representative_article.items()
            if key != "_similarity"
        }

    return cleaned_representative, selected


def _load_clusters_with_evidence(
    conn,
    *,
    run_id: int,
    evidence_articles_per_topic: int,
) -> list[dict[str, Any]]:
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
                c.centroid,
                c.origin_type,
                c.quality_score,
                c.is_oversized,
                c.first_seen_at,
                c.last_seen_at,
                cn.name_short,
                cn.name_title,
                cn.tags,
                cn.language_code
            FROM clusters c
            LEFT JOIN cluster_names cn
              ON cn.cluster_id = c.id
            WHERE c.run_id = %s
            ORDER BY
                c.size DESC,
                c.quality_score DESC NULLS LAST,
                c.id
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    result: list[dict[str, Any]] = []

    for row in rows:
        cluster_id = int(row["id"])
        representative_article_id = row["representative_article_id"]

        if representative_article_id is not None:
            representative_article_id = int(representative_article_id)

        representative_article, evidence_articles = (
            _load_central_evidence_articles(
                conn,
                cluster_id=cluster_id,
                centroid=row["centroid"],
                representative_article_id=representative_article_id,
                limit=evidence_articles_per_topic,
            )
        )

        result.append(
            {
                "cluster_id": cluster_id,
                "run_id": int(row["run_id"]),
                "internal_topic_reference": _safe_topic_name(row),
                "public_topic_title": _public_topic_title(
                    representative_article,
                    row["representative_title"],
                ),
                "size": int(row["size"]),
                "representative_article": representative_article,
                "evidence_articles": evidence_articles,
                "quality_score": (
                    float(row["quality_score"])
                    if row["quality_score"] is not None
                    else None
                ),
            }
        )

    return result


def _load_lineage_edges(
    conn,
    parent_run_id: int,
    child_run_id: int,
) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                parent_cluster_id,
                child_cluster_id,
                centroid_similarity,
                article_overlap_ratio,
                article_overlap_count,
                parent_size,
                child_size,
                score,
                link_type
            FROM cluster_lineage
            WHERE parent_run_id = %s
              AND child_run_id = %s
            ORDER BY
                score DESC,
                article_overlap_count DESC,
                parent_cluster_id,
                child_cluster_id
            """,
            (parent_run_id, child_run_id),
        )
        rows = cur.fetchall()

    return [
        {
            "lineage_id": int(row["id"]),
            "parent_cluster_id": int(row["parent_cluster_id"]),
            "child_cluster_id": int(row["child_cluster_id"]),
            "parent_size": int(row["parent_size"]),
            "child_size": int(row["child_size"]),
            "article_overlap_count": int(row["article_overlap_count"]),
            "score": float(row["score"]),
        }
        for row in rows
    ]


def _trend_from_sizes(parent_size: int, child_size: int) -> str:
    if child_size > parent_size:
        return "growing"

    if child_size < parent_size:
        return "declining"

    return "stable"


def _coverage_change_percent(
    parent_size: int | None,
    child_size: int | None,
) -> float | None:
    if parent_size is None or child_size is None or parent_size <= 0:
        return None

    return round(((child_size - parent_size) / parent_size) * 100, 1)


def _coverage_momentum(
    *,
    trend: str,
    parent_size: int | None,
    child_size: int | None,
    analysis_period_hours: int,
) -> dict[str, Any]:
    return {
        "analysis_period_hours": analysis_period_hours,
        "coverage_change_percent": _coverage_change_percent(
            parent_size,
            child_size,
        ),
        "coverage_direction": trend,
    }


def _build_transitions(
    *,
    parent_clusters: list[dict[str, Any]],
    child_clusters: list[dict[str, Any]],
    lineage_edges: list[dict[str, Any]],
    analysis_period_hours: int,
) -> list[dict[str, Any]]:
    parent_by_id = {
        item["cluster_id"]: item
        for item in parent_clusters
    }

    incoming_by_child: dict[int, list[dict[str, Any]]] = defaultdict(list)
    outgoing_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for edge in lineage_edges:
        incoming_by_child[edge["child_cluster_id"]].append(edge)
        outgoing_by_parent[edge["parent_cluster_id"]].append(edge)

    transitions: list[dict[str, Any]] = []

    for child in child_clusters:
        incoming = incoming_by_child.get(child["cluster_id"], [])

        if not incoming:
            transitions.append(
                {
                    "transition_type": "new",
                    "trend": "new",
                    "topic_reference": child["internal_topic_reference"],
                    "public_topic_title": child["public_topic_title"],
                    "representative_article": child["representative_article"],
                    "evidence_articles": child["evidence_articles"],
                    "coverage_momentum": _coverage_momentum(
                        trend="new",
                        parent_size=None,
                        child_size=child["size"],
                        analysis_period_hours=analysis_period_hours,
                    ),
                    "_current_size": child["size"],
                    "_previous_size": 0,
                    "_quality_score": child["quality_score"],
                }
            )
            continue

        edge = max(
            incoming,
            key=lambda item: (
                item["score"],
                item["article_overlap_count"],
                item["parent_size"],
                -item["parent_cluster_id"],
            ),
        )

        parent = parent_by_id.get(edge["parent_cluster_id"])

        if parent is None:
            continue

        changed_name = (
            _normalize_topic_name(parent["internal_topic_reference"])
            != _normalize_topic_name(child["internal_topic_reference"])
        )

        transition_type = "reframed" if changed_name else "continuation"
        trend = _trend_from_sizes(
            edge["parent_size"],
            edge["child_size"],
        )

        transitions.append(
            {
                "transition_type": transition_type,
                "trend": trend,
                "topic_reference": child["internal_topic_reference"],
                "public_topic_title": child["public_topic_title"],
                "representative_article": child["representative_article"],
                "evidence_articles": child["evidence_articles"],
                "coverage_momentum": _coverage_momentum(
                    trend=trend,
                    parent_size=edge["parent_size"],
                    child_size=edge["child_size"],
                    analysis_period_hours=analysis_period_hours,
                ),
                "_current_size": edge["child_size"],
                "_previous_size": edge["parent_size"],
                "_quality_score": child["quality_score"],
            }
        )

    return transitions


def _select_editorial_topics(
    *,
    transitions: list[dict[str, Any]],
    max_topics: int,
) -> list[dict[str, Any]]:
    max_topics = min(max_topics, DEFAULT_MAX_TOPICS)

    def evidence_count(item: dict[str, Any]) -> int:
        return len(item.get("evidence_articles") or [])

    def source_count(item: dict[str, Any]) -> int:
        return len(
            {
                str(article.get("source_name", "")).casefold()
                for article in item.get("evidence_articles") or []
                if isinstance(article, dict)
                and str(article.get("source_name", "")).strip()
            }
        )

    def is_eligible(item: dict[str, Any]) -> bool:
        if evidence_count(item) < MIN_EDITORIAL_EVIDENCE_ARTICLES:
            return False

        if source_count(item) < 1:
            return False

        if not _normalize_whitespace(item.get("public_topic_title")):
            return False

        return int(item.get("_current_size", 0)) >= 8

    def rank(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
        current_size = int(item.get("_current_size", 0))
        previous_size = int(item.get("_previous_size", 0))
        change = abs(current_size - previous_size)

        type_bonus = {
            "continuation": 3,
            "new": 2,
            "reframed": 1,
        }.get(item["transition_type"], 0)

        return (
            type_bonus,
            current_size + change,
            source_count(item),
            float(item.get("_quality_score") or 0.0),
            item["topic_reference"],
        )

    selected: list[dict[str, Any]] = []
    used_references: set[str] = set()

    for item in sorted(
        [item for item in transitions if is_eligible(item)],
        key=rank,
        reverse=True,
    ):
        if len(selected) >= max_topics:
            break

        reference = item["topic_reference"]

        if reference in used_references:
            continue

        selected.append(
            {
                "topic_reference": reference,
                "public_topic_title": item["public_topic_title"],
                "transition_type": item["transition_type"],
                "trend": item["trend"],
                "coverage_momentum": item["coverage_momentum"],
                "representative_article": item["representative_article"],
                "evidence_articles": item["evidence_articles"],
            }
        )
        used_references.add(reference)

    return selected


def _build_agenda_summary(
    *,
    current_run: dict[str, Any],
    editorial_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    def change_value(item: dict[str, Any]) -> float:
        value = item["coverage_momentum"].get("coverage_change_percent")
        return float(value) if value is not None else 0.0

    growing = [
        item
        for item in editorial_topics
        if item["trend"] == "growing"
    ]

    declining = [
        item
        for item in editorial_topics
        if item["trend"] == "declining"
    ]

    newly_visible = [
        item
        for item in editorial_topics
        if item["transition_type"] == "new"
    ]

    reframed = [
        item
        for item in editorial_topics
        if item["transition_type"] == "reframed"
    ]

    def public_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "topic_reference": item["topic_reference"],
            "public_topic_title": item["public_topic_title"],
            "coverage_change_percent": item["coverage_momentum"].get(
                "coverage_change_percent"
            ),
        }

    return {
        "analysis_period_hours": current_run["window_hours"],
        "top_growing_topics": [
            public_item(item)
            for item in sorted(
                growing,
                key=change_value,
                reverse=True,
            )[:3]
        ],
        "top_declining_topics": [
            public_item(item)
            for item in sorted(
                declining,
                key=change_value,
            )[:3]
        ],
        "notable_new_topics": [
            public_item(item)
            for item in newly_visible[:3]
        ],
        "reframed_topics": [
            public_item(item)
            for item in reframed[:3]
        ],
    }


def build_mistral_video_payload(
    conn,
    child_run_id: int,
    target_duration_seconds: int = DEFAULT_TARGET_DURATION_SECONDS,
    max_topics: int = DEFAULT_MAX_TOPICS,
    headlines_per_topic: int = DEFAULT_EVIDENCE_ARTICLES_PER_TOPIC,
) -> dict[str, Any]:
    if target_duration_seconds < 30:
        raise ValueError("target_duration_seconds must be at least 30")

    if max_topics < 1:
        raise ValueError("max_topics must be at least 1")

    if headlines_per_topic < MIN_EDITORIAL_EVIDENCE_ARTICLES:
        raise ValueError(
            "headlines_per_topic must be at least "
            f"{MIN_EDITORIAL_EVIDENCE_ARTICLES}"
        )

    parent_run_id = get_previous_completed_run_id(conn, child_run_id)
    child_run = _load_run(conn, child_run_id)

    video_requirements = {
        "target_duration_seconds": target_duration_seconds,
        "language": "en",
        "accent": "American English",
        "platform": "youtube",
        "format": "source-attributed news agenda recap",
        "voice_style": "calm, professional American news narrator",
        "audience": "international English-speaking audience",
        "aspect_ratio": "16:9",
        "visual_style": "cinematic editorial news documentary",
    }

    constraints = {
        "must_use_only_input_data": True,
        "do_not_invent_facts": True,
        "topic_references_are_internal_only": True,
        "use_only_editorial_topics_for_main_stories": True,
        "attribute_single_source_claims_to_that_source": True,
        "use_exact_numbers_only_when_present_in_evidence_title": True,
        "show_source_disagreement_without_merging_values": True,
        "use_percentage_for_change_in_monitored_publication_volume": True,
        "never_expose_internal_analysis_terms_in_narration": True,
    }

    if parent_run_id is None:
        return {
            "project": {
                "project_name": "news_aggregator",
                "language": "en",
            },
            "video_requirements": video_requirements,
            "analysis_context": {
                "comparison_available": False,
                "analysis_period_hours": child_run["window_hours"],
                "reason": "No previous completed period is available.",
            },
            "previous_run": None,
            "current_run": {
                "period_ended_at": child_run["finished_at"],
            },
            "agenda_summary": None,
            "editorial_topics": [],
            "constraints": constraints,
        }

    parent_run = _load_run(conn, parent_run_id)

    parent_clusters = _load_clusters_with_evidence(
        conn,
        run_id=parent_run_id,
        evidence_articles_per_topic=headlines_per_topic,
    )

    child_clusters = _load_clusters_with_evidence(
        conn,
        run_id=child_run_id,
        evidence_articles_per_topic=headlines_per_topic,
    )

    lineage_edges = _load_lineage_edges(
        conn,
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
    )

    transitions = _build_transitions(
        parent_clusters=parent_clusters,
        child_clusters=child_clusters,
        lineage_edges=lineage_edges,
        analysis_period_hours=child_run["window_hours"],
    )

    editorial_topics = _select_editorial_topics(
        transitions=transitions,
        max_topics=max_topics,
    )

    agenda_summary = _build_agenda_summary(
        current_run=child_run,
        editorial_topics=editorial_topics,
    )

    return {
        "project": {
            "project_name": "news_aggregator",
            "language": "en",
        },
        "video_requirements": video_requirements,
        "analysis_context": {
            "comparison_available": True,
            "analysis_period_hours": child_run["window_hours"],
            "comparison_description": (
                "The current period is compared with the immediately "
                "preceding completed period."
            ),
        },
        "previous_run": {
            "period_ended_at": parent_run["finished_at"],
        },
        "current_run": {
            "period_ended_at": child_run["finished_at"],
        },
        "agenda_summary": agenda_summary,
        "editorial_topics": editorial_topics,
        "constraints": constraints,
    }