from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import psycopg2.extras


DEFAULT_TARGET_DURATION_SECONDS = 120
DEFAULT_MAX_TOPICS = 8
DEFAULT_HEADLINES_PER_TOPIC = 3


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _safe_topic_name(row: dict[str, Any]) -> str:
    return (
        row.get("name_title")
        or row.get("name_short")
        or row.get("representative_title")
        or f"Cluster {row['id']}"
    )


def _normalize_topic_name(value: str) -> str:
    return " ".join(
        value.lower()
        .replace(",", " ")
        .replace(":", " ")
        .replace("-", " ")
        .split()
    )


def get_previous_completed_run_id(
    conn,
    child_run_id: int,
) -> int | None:
    """
    Returns the immediately preceding completed run by numeric run ID.
    """
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
            f"clustering run {run_id} is not completed: status={row['status']!r}"
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
        "largest_cluster_ratio": float(row["largest_cluster_ratio"] or 0.0),
        "emergent_topic_count": int(row["emergent_topic_count"] or 0),
        "emergent_assigned_article_count": int(
            row["emergent_assigned_article_count"] or 0
        ),
    }


def _load_clusters_with_headlines(
    conn,
    run_id: int,
    headlines_per_topic: int,
) -> list[dict[str, Any]]:
    sql = """
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
        cn.language_code,
        COALESCE(
            headlines.headline_samples,
            ARRAY[]::text[]
        ) AS headline_samples
    FROM clusters c
    LEFT JOIN cluster_names cn
      ON cn.cluster_id = c.id
    LEFT JOIN LATERAL (
        SELECT ARRAY_AGG(
            sample.title
            ORDER BY sample.published DESC NULLS LAST
        ) AS headline_samples
        FROM (
            SELECT
                a.title,
                a.published
            FROM cluster_articles ca
            JOIN articles a
              ON a.id = ca.article_id
            WHERE ca.cluster_id = c.id
              AND a.title IS NOT NULL
              AND BTRIM(a.title) <> ''
            ORDER BY a.published DESC NULLS LAST, a.id DESC
            LIMIT %s
        ) sample
    ) headlines ON TRUE
    WHERE c.run_id = %s
    ORDER BY
        c.size DESC,
        c.quality_score DESC NULLS LAST,
        c.id
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (headlines_per_topic, run_id))
        rows = cur.fetchall()

    result: list[dict[str, Any]] = []

    for row in rows:
        result.append(
            {
                "cluster_id": int(row["id"]),
                "run_id": int(row["run_id"]),
                "label": int(row["label"]),
                "topic_name": _safe_topic_name(row),
                "size": int(row["size"]),
                "representative_title": row["representative_title"],
                "headline_samples": list(row["headline_samples"] or []),
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


def _build_transitions(
    parent_clusters: list[dict[str, Any]],
    child_clusters: list[dict[str, Any]],
    lineage_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_by_id = {cluster["cluster_id"]: cluster for cluster in parent_clusters}

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
                    "topic_name": child["topic_name"],
                    "child_cluster_id": child_id,
                    "child_size": child["size"],
                    "parent_cluster_ids": [],
                    "parent_topic_names": [],
                    "headline_samples": child["headline_samples"],
                    "tags": child["tags"],
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
                "Lineage edge references a parent cluster outside the selected parent run: "
                f"parent_cluster_id={primary_edge['parent_cluster_id']}"
            )

        parent_name = parent["topic_name"]
        child_name = child["topic_name"]
        name_changed = (
            _normalize_topic_name(parent_name)
            != _normalize_topic_name(child_name)
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
                "topic_name": child_name,
                "parent_topic_name": parent_name,
                "parent_cluster_id": parent["cluster_id"],
                "child_cluster_id": child_id,
                "parent_size": primary_edge["parent_size"],
                "child_size": primary_edge["child_size"],
                "size_delta": (
                    primary_edge["child_size"] - primary_edge["parent_size"]
                ),
                "centroid_similarity": primary_edge["centroid_similarity"],
                "article_overlap_ratio": primary_edge["article_overlap_ratio"],
                "article_overlap_count": primary_edge["article_overlap_count"],
                "lineage_score": primary_edge["score"],
                "link_type": primary_edge["link_type"],
                "headline_samples": child["headline_samples"],
                "tags": child["tags"],
                "quality_score": child["quality_score"],
                "incoming_lineage_count": len(incoming),
                "outgoing_lineage_count": len(
                    outgoing_by_parent.get(parent["cluster_id"], [])
                ),
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
                "topic_name": parent["topic_name"],
                "parent_cluster_id": parent_id,
                "parent_size": parent["size"],
                "child_cluster_ids": [],
                "headline_samples": parent["headline_samples"],
                "tags": parent["tags"],
                "quality_score": parent["quality_score"],
            }
        )

    priority = {
        "new": 0,
        "reframed": 1,
        "continuation": 2,
        "disappeared": 3,
    }

    return sorted(
        transitions,
        key=lambda item: (
            priority[item["transition_type"]],
            -int(item.get("child_size", item.get("parent_size", 0))),
            item["topic_name"],
        ),
    )


def _select_editorial_topics(
    transitions: list[dict[str, Any]],
    max_topics: int,
) -> list[dict[str, Any]]:
    """
    Возвращает небольшой редакционный набор тем для короткого видео.

    Цель: не позволять всем маленьким new-кластерам вытеснить крупные
    продолжения, по которым есть заметная динамика и содержательные заголовки.
    """
    max_topics = min(max_topics, DEFAULT_MAX_TOPICS)

    def topic_size(item: dict[str, Any]) -> int:
        return int(item.get("child_size", item.get("parent_size", 0)))

    def size_delta_abs(item: dict[str, Any]) -> int:
        return abs(int(item.get("size_delta", 0)))

    def headline_count(item: dict[str, Any]) -> int:
        return len(item.get("headline_samples") or [])

    def is_eligible(item: dict[str, Any]) -> bool:
        transition_type = item["transition_type"]
        size = topic_size(item)

        if headline_count(item) < 2:
            return False

        if transition_type == "new":
            return size >= 10

        if transition_type == "reframed":
            return size >= 10

        if transition_type == "disappeared":
            return size >= 12

        if transition_type == "continuation":
            return size >= 12 or size_delta_abs(item) >= 5

        return False

    def score(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
        transition_type = item["transition_type"]
        size = topic_size(item)
        delta = size_delta_abs(item)
        quality_score = float(item.get("quality_score") or 0.0)

        transition_bonus = {
            "new": 5,
            "reframed": 4,
            "disappeared": 3,
            "continuation": 2,
        }.get(transition_type, 0)

        return (
            size + delta + transition_bonus,
            delta,
            size,
            quality_score,
            item["topic_name"],
        )

    eligible = [item for item in transitions if is_eligible(item)]

    grouped: dict[str, list[dict[str, Any]]] = {
        "new": [],
        "reframed": [],
        "disappeared": [],
        "continuation": [],
    }

    for item in eligible:
        grouped[item["transition_type"]].append(item)

    for items in grouped.values():
        items.sort(key=score, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int | None, int | None]] = set()

    def identity(item: dict[str, Any]) -> tuple[str, int | None, int | None]:
        return (
            item["topic_name"],
            item.get("parent_cluster_id"),
            item.get("child_cluster_id"),
        )

    def add_from_group(items: list[dict[str, Any]], limit: int) -> None:
        added = 0

        for item in items:
            if len(selected) >= max_topics or added >= limit:
                return

            item_key = identity(item)

            if item_key in selected_keys:
                continue

            selected.append(item)
            selected_keys.add(item_key)
            added += 1

    # Сохраняем новизну, но не разрешаем ей заполнить весь ролик.
    add_from_group(grouped["new"], limit=2)
    add_from_group(grouped["reframed"], limit=2)
    add_from_group(grouped["disappeared"], limit=2)

    # Основная часть новостного обзора: устойчивые темы с масштабом и динамикой.
    add_from_group(grouped["continuation"], limit=5)

    # Если набор оказался короче лимита, добираем лучшие доступные темы.
    for item in sorted(eligible, key=score, reverse=True):
        if len(selected) >= max_topics:
            break

        item_key = identity(item)

        if item_key in selected_keys:
            continue

        selected.append(item)
        selected_keys.add(item_key)

    return selected


def build_mistral_video_payload(
    conn,
    child_run_id: int,
    target_duration_seconds: int = DEFAULT_TARGET_DURATION_SECONDS,
    max_topics: int = DEFAULT_MAX_TOPICS,
    headlines_per_topic: int = DEFAULT_HEADLINES_PER_TOPIC,
) -> dict[str, Any]:
    """
    Формирует фактологический входной payload для Mistral.

    Функция не вызывает Mistral API и не изменяет БД.

    Вызывать только после того, как:
    1. текущий run сохранён со статусом success/completed/degraded;
    2. cluster_names сохранены;
    3. lineage parent -> child сохранён;
    4. транзакция с lineage успешно закоммичена.
    """
    if target_duration_seconds < 30:
        raise ValueError("target_duration_seconds must be at least 30 seconds")

    if max_topics < 1:
        raise ValueError("max_topics must be at least 1")

    if headlines_per_topic < 1:
        raise ValueError("headlines_per_topic must be at least 1")

    parent_run_id = get_previous_completed_run_id(conn, child_run_id)
    child_run = _load_run(conn, child_run_id)

    video_requirements = {
        "target_duration_seconds": target_duration_seconds,
        "language": "en",
        "platform": "youtube",
        "format": "neutral news coverage recap",
        "voice_style": "calm professional news narrator",
        "aspect_ratio": "16:9",
        "visual_style": "cinematic editorial news documentary",
    }

    constraints = {
        "must_use_only_input_data": True,
        "do_not_invent_facts": True,
        "do_not_interpret_metrics_without_thresholds": True,
        "headlines_are_source_material_not_verified_claims": True,
        "max_topics_sent_to_model": min(max_topics, DEFAULT_MAX_TOPICS),
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
                "child_run_id": child_run_id,
                "reason": "No previous completed run exists for comparison.",
            },
            "previous_run": None,
            "current_run": child_run,
            "topic_transitions": [],
            "editorial_topics": [],
            "constraints": constraints,
        }

    parent_run = _load_run(conn, parent_run_id)

    parent_clusters = _load_clusters_with_headlines(
        conn=conn,
        run_id=parent_run_id,
        headlines_per_topic=headlines_per_topic,
    )
    child_clusters = _load_clusters_with_headlines(
        conn=conn,
        run_id=child_run_id,
        headlines_per_topic=headlines_per_topic,
    )
    lineage_edges = _load_lineage_edges(
        conn=conn,
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
    )

    transitions = _build_transitions(
        parent_clusters=parent_clusters,
        child_clusters=child_clusters,
        lineage_edges=lineage_edges,
    )
    editorial_topics = _select_editorial_topics(
        transitions=transitions,
        max_topics=max_topics,
    )

    return {
        "project": {
            "project_name": "news_aggregator",
            "language": "en",
        },
        "video_requirements": video_requirements,
        "analysis_context": {
            "comparison_available": True,
            "parent_run_id": parent_run_id,
            "child_run_id": child_run_id,
            "comparison_type": "adjacent_completed_runs",
            "lineage_edge_count": len(lineage_edges),
        },
        "previous_run": parent_run,
        "current_run": child_run,
        "topic_transitions": transitions,
        "editorial_topics": editorial_topics,
        "constraints": constraints,
    }