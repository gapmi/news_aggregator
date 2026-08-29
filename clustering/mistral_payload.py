from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import psycopg2.extras

DEFAULT_TARGET_DURATION_SECONDS = 120
DEFAULT_MAX_TOPICS = 6
DEFAULT_EVIDENCE_ARTICLES_PER_TOPIC = 5
DEFAULT_MAX_NEW_TOPICS = 2
DEFAULT_MAX_REFRAMED_TOPICS = 2
DEFAULT_MAX_CONTINUING_TOPICS = 5

MIN_EDITORIAL_EVIDENCE_ARTICLES = 2


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalize_whitespace(value: str | None) -> str:
    return " ".join((value or "").split())


def _safe_topic_name(row: dict[str, Any]) -> str:
    """
    Return an internal topic reference.

    This is not a public news headline and must not be spoken by the narrator.
    Public editorial meaning comes from evidence article titles.
    """
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
    """
    Build a public-facing editorial title from an actual article title.

    Topic labels produced by keyword extraction can be fragmentary. The
    representative article is closest to the topic centroid, so its title is
    the safest available public summary of the story.
    """
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
    """
    Return a publication/source name suitable for on-air attribution.

    articles.source is the available source field in the current schema.
    Empty values remain None and are not offered as evidence to Mistral.
    """
    if not isinstance(value, str):
        return None

    normalized = _normalize_whitespace(value)

    if not normalized:
        return None

    return normalized[:160]


def get_previous_completed_run_id(
    conn,
    child_run_id: int,
) -> int | None:
    """Return the nearest earlier completed clustering run."""
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


def _load_evidence_articles(
    conn,
    *,
    cluster_id: int,
    representative_article_id: int | None,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Load a centroid-nearest representative article plus diverse supporting items.

    The representative article ID was selected during clustering as the article
    closest to the cluster centroid. Supporting evidence is selected from the
    same topic with a preference for source diversity and recency.

    The current database has no persisted per-article centroid distance, so
    only representative_article_id is guaranteed centroid-near.
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
                CASE
                    WHEN a.id = %s THEN 0
                    ELSE 1
                END AS representative_rank
            FROM cluster_articles ca
            JOIN articles a
              ON a.id = ca.article_id
            WHERE ca.cluster_id = %s
              AND a.title IS NOT NULL
              AND BTRIM(a.title) <> ''
            ORDER BY
                representative_rank ASC,
                a.published DESC NULLS LAST,
                a.id DESC
            """,
            (representative_article_id, cluster_id),
        )
        rows = cur.fetchall()

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        title = _normalize_whitespace(row["title"])
        source_name = _normalize_source_name(row["source"])

        if not title or source_name is None:
            continue

        normalized_rows.append(
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
            }
        )

    representative_article = next(
        (
            article
            for article in normalized_rows
            if article["is_representative"]
        ),
        None,
    )

    selected: list[dict[str, Any]] = []
    seen_article_ids: set[int] = set()
    seen_title_keys: set[str] = set()
    seen_sources: set[str] = set()

    def add_article(article: dict[str, Any]) -> bool:
        article_id = article["article_id"]
        title_key = article["title"].casefold()

        if article_id in seen_article_ids or title_key in seen_title_keys:
            return False

        selected.append(article)
        seen_article_ids.add(article_id)
        seen_title_keys.add(title_key)
        seen_sources.add(article["source_name"].casefold())
        return True

    if representative_article is not None:
        add_article(representative_article)

    for article in normalized_rows:
        if len(selected) >= limit:
            break

        if article["source_name"].casefold() in seen_sources:
            continue

        add_article(article)

    for article in normalized_rows:
        if len(selected) >= limit:
            break

        add_article(article)

    return representative_article, selected


def _load_clusters_with_evidence(
    conn,
    *,
    run_id: int,
    evidence_articles_per_topic: int,
) -> list[dict[str, Any]]:
    """
    Load cluster metadata and source-attributed evidence for a run.

    Internal labels are retained only as stable references. The public editorial
    description is derived from the representative article and evidence list.
    """
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

    clusters: list[dict[str, Any]] = []

    for row in rows:
        cluster_id = int(row["id"])
        representative_article_id = row["representative_article_id"]

        if representative_article_id is not None:
            representative_article_id = int(representative_article_id)

        representative_article, evidence_articles = _load_evidence_articles(
            conn,
            cluster_id=cluster_id,
            representative_article_id=representative_article_id,
            limit=evidence_articles_per_topic,
        )

        internal_topic_reference = _safe_topic_name(row)
        public_title = _public_topic_title(
            representative_article,
            row["representative_title"],
        )

        clusters.append(
            {
                "cluster_id": cluster_id,
                "run_id": int(row["run_id"]),
                "internal_topic_reference": internal_topic_reference,
                "public_topic_title": public_title,
                "size": int(row["size"]),
                "representative_article": representative_article,
                "evidence_articles": evidence_articles,
                "tags": list(row["tags"] or []),
                "language_code": row["language_code"],
                "origin_type": row["origin_type"],
                "quality_score": (
                    float(row["quality_score"])
                    if row["quality_score"] is not None
                    else None
                ),
                "is_oversized": bool(row["is_oversized"]),
                "first_seen_at": _iso(row["first_seen_at"]),
                "last_seen_at": _iso(row["last_seen_at"]),
            }
        )

    return clusters


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
            "link_type": row["link_type"],
            "centroid_similarity": float(row["centroid_similarity"]),
            "article_overlap_ratio": float(row["article_overlap_ratio"]),
            "article_overlap_count": int(row["article_overlap_count"]),
            "parent_size": int(row["parent_size"]),
            "child_size": int(row["child_size"]),
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
    """
    Calculate the change in number of monitored publications.

    The percentage describes only volume in this news-monitoring dataset, not
    the real-world scale or importance of the event.
    """
    if parent_size is None or child_size is None or parent_size <= 0:
        return None

    return round(((child_size - parent_size) / parent_size) * 100, 1)


def _public_momentum(
    *,
    trend: str,
    parent_size: int | None,
    child_size: int | None,
    analysis_period_hours: int,
) -> dict[str, Any]:
    return {
        "analysis_period_hours": analysis_period_hours,
        "previous_publication_count": parent_size,
        "current_publication_count": child_size,
        "coverage_change_percent": _coverage_change_percent(
            parent_size,
            child_size,
        ),
        "coverage_direction": trend,
        "editorial_interpretation": (
            "This measures change in the number of monitored publications, "
            "not the real-world scale or importance of the event."
        ),
    }


def _build_transitions(
    *,
    parent_clusters: list[dict[str, Any]],
    child_clusters: list[dict[str, Any]],
    lineage_edges: list[dict[str, Any]],
    analysis_period_hours: int,
) -> list[dict[str, Any]]:
    parent_by_id = {
        cluster["cluster_id"]: cluster
        for cluster in parent_clusters
    }

    incoming_by_child: dict[int, list[dict[str, Any]]] = defaultdict(list)
    outgoing_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for edge in lineage_edges:
        incoming_by_child[edge["child_cluster_id"]].append(edge)
        outgoing_by_parent[edge["parent_cluster_id"]].append(edge)

    transitions: list[dict[str, Any]] = []

    for child in child_clusters:
        child_id = child["cluster_id"]
        incoming = incoming_by_child.get(child_id, [])

        if not incoming:
            transitions.append(
                {
                    "transition_type": "new",
                    "trend": "new",
                    "topic_reference": child["internal_topic_reference"],
                    "public_topic_title": child["public_topic_title"],
                    "child_cluster_id": child_id,
                    "evidence_articles": child["evidence_articles"],
                    "representative_article": child["representative_article"],
                    "coverage_momentum": _public_momentum(
                        trend="new",
                        parent_size=None,
                        child_size=child["size"],
                        analysis_period_hours=analysis_period_hours,
                    ),
                    "quality_score": child["quality_score"],
                }
            )
            continue

        primary_edge = max(
            incoming,
            key=lambda edge: (
                edge["score"],
                edge["article_overlap_count"],
                edge["parent_size"],
                -edge["parent_cluster_id"],
            ),
        )

        parent = parent_by_id.get(primary_edge["parent_cluster_id"])

        if parent is None:
            raise ValueError(
                "Lineage edge references a parent topic outside the selected "
                f"parent run: parent_cluster_id={primary_edge['parent_cluster_id']}"
            )

        name_changed = (
            _normalize_topic_name(parent["internal_topic_reference"])
            != _normalize_topic_name(child["internal_topic_reference"])
        )

        transition_type = "reframed" if name_changed else "continuation"
        trend = _trend_from_sizes(
            primary_edge["parent_size"],
            primary_edge["child_size"],
        )

        transitions.append(
            {
                "transition_type": transition_type,
                "trend": trend,
                "topic_reference": child["internal_topic_reference"],
                "public_topic_title": child["public_topic_title"],
                "parent_public_topic_title": parent["public_topic_title"],
                "parent_cluster_id": parent["cluster_id"],
                "child_cluster_id": child_id,
                "evidence_articles": child["evidence_articles"],
                "representative_article": child["representative_article"],
                "coverage_momentum": _public_momentum(
                    trend=trend,
                    parent_size=primary_edge["parent_size"],
                    child_size=primary_edge["child_size"],
                    analysis_period_hours=analysis_period_hours,
                ),
                "quality_score": child["quality_score"],
            }
        )

    for parent in parent_clusters:
        parent_id = parent["cluster_id"]

        if outgoing_by_parent.get(parent_id):
            continue

        transitions.append(
            {
                "transition_type": "disappeared",
                "trend": "disappeared",
                "topic_reference": parent["internal_topic_reference"],
                "public_topic_title": parent["public_topic_title"],
                "parent_cluster_id": parent_id,
                "evidence_articles": parent["evidence_articles"],
                "representative_article": parent["representative_article"],
                "coverage_momentum": _public_momentum(
                    trend="disappeared",
                    parent_size=parent["size"],
                    child_size=None,
                    analysis_period_hours=analysis_period_hours,
                ),
                "quality_score": parent["quality_score"],
            }
        )

    sort_priority = {
        "new": 0,
        "reframed": 1,
        "continuation": 2,
        "disappeared": 3,
    }

    return sorted(
        transitions,
        key=lambda item: (
            sort_priority[item["transition_type"]],
            -int(
                item["coverage_momentum"].get(
                    "current_publication_count"
                )
                or item["coverage_momentum"].get(
                    "previous_publication_count"
                )
                or 0
            ),
            item["topic_reference"],
        ),
    )


def _select_editorial_topics(
    *,
    transitions: list[dict[str, Any]],
    max_topics: int,
) -> list[dict[str, Any]]:
    """
    Select coherent, source-attributed topics eligible for spoken narration.

    A topic requires at least two evidence articles with non-empty source names.
    Internal labels remain references only; narration should be built from
    public_topic_title and evidence articles.
    """
    max_topics = min(max_topics, DEFAULT_MAX_TOPICS)

    def current_size(item: dict[str, Any]) -> int:
        return int(
            item["coverage_momentum"].get(
                "current_publication_count"
            )
            or 0
        )

    def previous_size(item: dict[str, Any]) -> int:
        return int(
            item["coverage_momentum"].get(
                "previous_publication_count"
            )
            or 0
        )

    def absolute_change(item: dict[str, Any]) -> int:
        return abs(current_size(item) - previous_size(item))

    def evidence_count(item: dict[str, Any]) -> int:
        return len(item.get("evidence_articles") or [])

    def source_count(item: dict[str, Any]) -> int:
        return len(
            {
                article["source_name"].casefold()
                for article in item.get("evidence_articles") or []
                if isinstance(article, dict)
                and isinstance(article.get("source_name"), str)
                and article["source_name"].strip()
            }
        )

    def quality_score(item: dict[str, Any]) -> float:
        return float(item.get("quality_score") or 0.0)

    def eligible(item: dict[str, Any]) -> bool:
        transition_type = item["transition_type"]

        if transition_type == "disappeared":
            return False

        if evidence_count(item) < MIN_EDITORIAL_EVIDENCE_ARTICLES:
            return False

        if source_count(item) < 1:
            return False

        if not _normalize_whitespace(item.get("public_topic_title")):
            return False

        size = current_size(item)

        if transition_type == "new":
            return size >= 10

        if transition_type == "reframed":
            return size >= 12

        return size >= 10 or absolute_change(item) >= 4

    def score(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
        type_bonus = {
            "continuation": 3,
            "new": 2,
            "reframed": 1,
        }.get(item["transition_type"], 0)

        return (
            type_bonus,
            current_size(item) + absolute_change(item),
            source_count(item),
            quality_score(item),
            item["topic_reference"],
        )

    eligible_topics = [
        item
        for item in transitions
        if eligible(item)
    ]

    continuations = sorted(
        [
            item
            for item in eligible_topics
            if item["transition_type"] == "continuation"
        ],
        key=score,
        reverse=True,
    )

    new_topics = sorted(
        [
            item
            for item in eligible_topics
            if item["transition_type"] == "new"
        ],
        key=score,
        reverse=True,
    )

    reframed_topics = sorted(
        [
            item
            for item in eligible_topics
            if item["transition_type"] == "reframed"
        ],
        key=score,
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_references: set[str] = set()

    def add(items: list[dict[str, Any]], limit: int) -> None:
        added = 0

        for item in items:
            if len(selected) >= max_topics or added >= limit:
                return

            reference = item["topic_reference"]

            if reference in selected_references:
                continue

            selected.append(item)
            selected_references.add(reference)
            added += 1

    add(continuations, DEFAULT_MAX_CONTINUING_TOPICS)
    add(new_topics, DEFAULT_MAX_NEW_TOPICS)
    add(reframed_topics, DEFAULT_MAX_REFRAMED_TOPICS)

    remaining = sorted(
        eligible_topics,
        key=score,
        reverse=True,
    )

    for item in remaining:
        if len(selected) >= max_topics:
            break

        reference = item["topic_reference"]

        if reference in selected_references:
            continue

        selected.append(item)
        selected_references.add(reference)

    return selected


def _build_agenda_summary(
    *,
    previous_run: dict[str, Any],
    current_run: dict[str, Any],
    transitions: list[dict[str, Any]],
    editorial_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build concise audience-facing trend context.

    Internal terms stay out of narration. Counts are retained for backend
    grounding, but narration is instructed to report percentage changes.
    """
    relevant = [
        item
        for item in transitions
        if item["transition_type"] != "disappeared"
    ]

    growing = [
        item
        for item in relevant
        if item["trend"] == "growing"
    ]

    declining = [
        item
        for item in relevant
        if item["trend"] == "declining"
    ]

    new_topics = [
        item
        for item in relevant
        if item["transition_type"] == "new"
    ]

    reframed = [
        item
        for item in relevant
        if item["transition_type"] == "reframed"
    ]

    def sort_by_change(item: dict[str, Any]) -> tuple[float, int, str]:
        momentum = item["coverage_momentum"]
        percent = momentum.get("coverage_change_percent")

        return (
            abs(float(percent)) if percent is not None else -1.0,
            int(momentum.get("current_publication_count") or 0),
            item["topic_reference"],
        )

    return {
        "analysis_period_hours": current_run["window_hours"],
        "editorial_topic_references": [
            item["topic_reference"]
            for item in editorial_topics
        ],
        "editorial_topic_titles": [
            item["public_topic_title"]
            for item in editorial_topics
        ],
        "top_growing_topics": [
            {
                "topic_reference": item["topic_reference"],
                "public_topic_title": item["public_topic_title"],
                "coverage_change_percent": item["coverage_momentum"].get(
                    "coverage_change_percent"
                ),
            }
            for item in sorted(
                growing,
                key=sort_by_change,
                reverse=True,
            )[:3]
        ],
        "top_declining_topics": [
            {
                "topic_reference": item["topic_reference"],
                "public_topic_title": item["public_topic_title"],
                "coverage_change_percent": item["coverage_momentum"].get(
                    "coverage_change_percent"
                ),
            }
            for item in sorted(
                declining,
                key=sort_by_change,
                reverse=True,
            )[:3]
        ],
        "notable_new_topics": [
            {
                "topic_reference": item["topic_reference"],
                "public_topic_title": item["public_topic_title"],
            }
            for item in sorted(
                new_topics,
                key=lambda item: (
                    int(
                        item["coverage_momentum"].get(
                            "current_publication_count"
                        )
                        or 0
                    ),
                    item["topic_reference"],
                ),
                reverse=True,
            )[:3]
        ],
        "reframed_topics": [
            {
                "topic_reference": item["topic_reference"],
                "public_topic_title": item["public_topic_title"],
            }
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
    """
    Build source-aware editorial input for Mistral.

    `headlines_per_topic` is retained as a compatibility parameter but now
    controls the number of source-attributed evidence articles per topic.
    """
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
                "child_run_id": child_run_id,
                "reason": "No previous completed period is available.",
            },
            "previous_run": None,
            "current_run": child_run,
            "agenda_summary": None,
            "editorial_topics": [],
            "constraints": constraints,
        }

    parent_run = _load_run(conn, parent_run_id)

    analysis_period_hours = child_run["window_hours"]

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
        analysis_period_hours=analysis_period_hours,
    )

    editorial_topics = _select_editorial_topics(
        transitions=transitions,
        max_topics=max_topics,
    )

    agenda_summary = _build_agenda_summary(
        previous_run=parent_run,
        current_run=child_run,
        transitions=transitions,
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
            "analysis_period_hours": analysis_period_hours,
            "comparison_description": (
                "The current period is compared with the immediately "
                "preceding completed period of the same monitoring workflow."
            ),
        },
        "previous_run": {
            "period_ended_at": parent_run["finished_at"],
            "article_count": parent_run["article_count"],
        },
        "current_run": {
            "period_ended_at": child_run["finished_at"],
            "article_count": child_run["article_count"],
        },
        "agenda_summary": agenda_summary,
        "editorial_topics": editorial_topics,
        "constraints": constraints,
    }