from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2.extras


DEFAULT_TARGET_DURATION_SECONDS = 120
DEFAULT_MAX_TOPICS = 12
DEFAULT_HEADLINES_PER_TOPIC = 3


@dataclass(frozen=True)
class RunPair:
    parent_run_id: int
    child_run_id: int


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
        SELECT ARRAY_AGG(sample.title ORDER BY sample.published DESC NULLS LAST)
            AS headline_samples
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
    child_by_id = {cluster["cluster_id"]: cluster for cluster in child_clusters}

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
        parent = parent_by_id[primary_edge["parent_cluster_id"]]

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
    Ограничиваем payload наиболее заметными темами.
    Новые, переформулированные и исчезнувшие темы получают приоритет;
    затем идут наиболее крупные продолжения.
    """
    priority = {
        "new": 0,
        "reframed": 1,
        "disappeared": 2,
        "continuation": 3,
    }

    ranked = sorted(
        transitions,
        key=lambda item: (
            priority[item["transition_type"]],
            -abs(int(item.get("size_delta", 0))),
            -int(item.get("child_size", item.get("parent_size", 0))),
            item["topic_name"],
        ),
    )

    return ranked[:max_topics]


def _select_editorial_topics(
    transitions: list[dict[str, Any]],
    max_topics: int,
) -> list[dict[str, Any]]:
    """
    Отбирает темы для короткого новостного видеосценария.

    Бизнес-правило: крупные продолжающиеся темы с заметной динамикой
    не должны вытесняться небольшими новыми кластерами. В output
    попадают только темы, для которых есть достаточно материала
    для нейтральной редакционной формулировки.
    """
    max_topics = min(max_topics, 8)

    def topic_size(item: dict[str, Any]) -> int:
        return int(item.get("child_size", item.get("parent_size", 0)))

    def headline_count(item: dict[str, Any]) -> int:
        return len(item.get("headline_samples") or [])

    def editorial_score(item: dict[str, Any]) -> tuple[int, int, int, str]:
        transition_type = item["transition_type"]
        size = topic_size(item)
        delta = abs(int(item.get("size_delta", 0)))

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
            item["topic_name"],
        )

    def is_eligible(item: dict[str, Any]) -> bool:
        transition_type = item["transition_type"]
        size = topic_size(item)

        if headline_count(item) < 2:
            return False

        if transition_type == "disappeared":
            return size >= 12

        if transition_type == "new":
            return size >= 10

        if transition_type == "reframed":
            return size >= 10

        if transition_type == "continuation":
            return size >= 12 or abs(int(item.get("size_delta", 0))) >= 5

        return False

    eligible = [item for item in transitions if is_eligible(item)]

    grouped: dict[str, list[dict[str, Any]]] = {
        "new": [],
        "reframed": [],
        "disappeared": [],
        "continuation": [],
    }

    for item in eligible:
        grouped.setdefault(item["transition_type"], []).append(item)

    for items in grouped.values():
        items.sort(key=editorial_score, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, int | None, int | None]] = set()

    def add_items(items: list[dict[str, Any]], limit: int) -> None:
        for item in items:
            if len(selected) >= max_topics:
                return

            item_key = (
                item["topic_name"],
                item.get("parent_cluster_id"),
                item.get("child_cluster_id"),
            )

            if item_key in selected_ids:
                continue

            selected.append(item)
            selected_ids.add(item_key)

            if sum(
                1
                for selected_item in selected
                if selected_item["transition_type"] == item["transition_type"]
            ) >= limit:
                return

    # Новизна важна, но ограничена: она не должна заполнить весь ролик.
    add_items(grouped["new"], limit=2)
    add_items(grouped["reframed"], limit=2)
    add_items(grouped["disappeared"], limit=2)

    # Основа ролика — крупнейшие и наиболее изменившиеся продолжения.
    add_items(grouped["continuation"], limit=5)

    # Если тематических категорий оказалось мало, добираем лучшие темы
    # из общего списка, сохраняя max_topics.
    remaining = sorted(eligible, key=editorial_score, reverse=True)

    for item in remaining:
        if len(selected) >= max_topics:
            break

        item_key = (
            item["topic_name"],
            item.get("parent_cluster_id"),
            item.get("child_cluster_id"),
        )

        if item_key in selected_ids:
            continue

        selected.append(item)
        selected_ids.add(item_key)

    return selected