from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import psycopg2.extras


DEFAULT_TARGET_DURATION_SECONDS = 120
DEFAULT_MAX_TOPICS = 6
DEFAULT_HEADLINES_PER_TOPIC = 3
DEFAULT_MAX_NEW_TOPICS = 2
DEFAULT_MAX_REFRAMED_TOPICS = 2
DEFAULT_MAX_CONTINUING_TOPICS = 5


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
    Возвращает ближайший предыдущий завершённый run.

    Для сравнения используется только соседняя пара run по ID:
    parent_run_id < child_run_id, status завершён, finished_at задан.
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
    """
    Загружает кластеры run, их отображаемые имена и последние headline samples.

    Headline samples — исходный редакционный материал для модели.
    Они не считаются подтверждёнными фактами сами по себе.
    """
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
            ORDER BY sample.published DESC NULLS LAST, sample.id DESC
        ) AS headline_samples
        FROM (
            SELECT
                a.id,
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
    """
    Вычисляет тип перехода в Python, а не поручает это LLM.

    Модель получает уже готовые, проверяемые категории:
    new, disappeared, continuation, reframed.
    """
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
                "Lineage edge references a parent cluster outside the selected "
                f"parent run: parent_cluster_id={primary_edge['parent_cluster_id']}"
            )

        parent_name = parent["topic_name"]
        child_name = child["topic_name"]

        name_changed = (
            _normalize_topic_name(parent_name)
            != _normalize_topic_name(child_name)
        )

        transition_type = "reframed" if name_changed else "continuation"

        transitions.append(
            {
                "transition_type": transition_type,
                "trend": _trend_from_sizes(
                    primary_edge["parent_size"],
                    primary_edge["child_size"],
                ),
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
            -int(item.get("child_size", item.get("parent_size", 0))),
            item["topic_name"],
        ),
    )


def _select_editorial_topics(
    transitions: list[dict[str, Any]],
    max_topics: int,
) -> list[dict[str, Any]]:
    """
    Возвращает набор тем для формата A: News agenda recap.

    Формат ориентирован на текущую новостную повестку, поэтому:
    - disappeared topics не включаются в видео-сценарий;
    - максимум две новые темы;
    - максимум две reframed темы;
    - основой служат крупные continuations с заметной динамикой;
    - короткий ролик ограничен максимум шестью темами.

    Полный набор transitions остаётся в payload для аналитики, но Mistral
    должен строить основной сценарий по editorial_topics и agenda_summary.
    """
    max_topics = min(max_topics, DEFAULT_MAX_TOPICS)

    def topic_size(item: dict[str, Any]) -> int:
        return int(item.get("child_size", item.get("parent_size", 0)))

    def absolute_delta(item: dict[str, Any]) -> int:
        return abs(int(item.get("size_delta", 0)))

    def headline_count(item: dict[str, Any]) -> int:
        return len(item.get("headline_samples") or [])

    def quality_score(item: dict[str, Any]) -> float:
        return float(item.get("quality_score") or 0.0)

    def eligible_for_agenda(item: dict[str, Any]) -> bool:
        transition_type = item["transition_type"]
        size = topic_size(item)

        if headline_count(item) < 2:
            return False

        if transition_type == "disappeared":
            return False

        if transition_type == "new":
            return size >= 12

        if transition_type == "reframed":
            return size >= 15

        if transition_type == "continuation":
            return size >= 12 or absolute_delta(item) >= 5

        return False

    def continuing_score(item: dict[str, Any]) -> tuple[int, int, float, str]:
        return (
            topic_size(item) + absolute_delta(item),
            absolute_delta(item),
            quality_score(item),
            item["topic_name"],
        )

    def new_score(item: dict[str, Any]) -> tuple[int, float, str]:
        return (
            topic_size(item),
            quality_score(item),
            item["topic_name"],
        )

    def reframed_score(item: dict[str, Any]) -> tuple[int, int, float, str]:
        return (
            topic_size(item) + absolute_delta(item),
            absolute_delta(item),
            quality_score(item),
            item["topic_name"],
        )

    eligible = [
        item
        for item in transitions
        if eligible_for_agenda(item)
    ]

    continuations = sorted(
        [
            item
            for item in eligible
            if item["transition_type"] == "continuation"
        ],
        key=continuing_score,
        reverse=True,
    )

    new_topics = sorted(
        [
            item
            for item in eligible
            if item["transition_type"] == "new"
        ],
        key=new_score,
        reverse=True,
    )

    reframed_topics = sorted(
        [
            item
            for item in eligible
            if item["transition_type"] == "reframed"
        ],
        key=reframed_score,
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int | None, int | None]] = set()

    def identity(item: dict[str, Any]) -> tuple[str, int | None, int | None]:
        return (
            item["topic_name"],
            item.get("parent_cluster_id"),
            item.get("child_cluster_id"),
        )

    def add_items(items: list[dict[str, Any]], limit: int) -> None:
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

    # Сначала ядро повестки: наиболее крупные актуальные продолжения.
    add_items(continuations, DEFAULT_MAX_CONTINUING_TOPICS)

    # Затем — ограниченное число заметных новых тем.
    add_items(new_topics, DEFAULT_MAX_NEW_TOPICS)

    # Затем — устойчивые темы, изменившие название/формулировку.
    add_items(reframed_topics, DEFAULT_MAX_REFRAMED_TOPICS)

    # Если набор ещё не заполнен, добираем оставшиеся темы по общему score.
    remaining = sorted(
        eligible,
        key=lambda item: (
            topic_size(item) + absolute_delta(item),
            absolute_delta(item),
            quality_score(item),
            item["topic_name"],
        ),
        reverse=True,
    )

    for item in remaining:
        if len(selected) >= max_topics:
            break

        item_key = identity(item)

        if item_key in selected_keys:
            continue

        selected.append(item)
        selected_keys.add(item_key)

    return selected


def _build_agenda_summary(
    previous_run: dict[str, Any],
    current_run: dict[str, Any],
    transitions: list[dict[str, Any]],
    editorial_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Формирует короткий структурированный контекст для сценарной LLM.

    Это не текст для диктора. Это уже рассчитанные backend-данные,
    на основе которых Mistral пишет нейтральный news agenda recap.
    """
    def transition_size(item: dict[str, Any]) -> int:
        return int(item.get("child_size", item.get("parent_size", 0)))

    continuing = [
        item
        for item in transitions
        if item["transition_type"] == "continuation"
    ]
    newly_visible = [
        item
        for item in transitions
        if item["transition_type"] == "new"
    ]
    reframed = [
        item
        for item in transitions
        if item["transition_type"] == "reframed"
    ]
    disappeared = [
        item
        for item in transitions
        if item["transition_type"] == "disappeared"
    ]

    growing = sorted(
        [
            item
            for item in continuing + reframed
            if item.get("trend") == "growing"
        ],
        key=lambda item: (
            int(item.get("size_delta", 0)),
            transition_size(item),
        ),
        reverse=True,
    )

    declining = sorted(
        [
            item
            for item in continuing + reframed
            if item.get("trend") == "declining"
        ],
        key=lambda item: (
            int(item.get("size_delta", 0)),
            -transition_size(item),
        ),
    )

    largest_current_topics = sorted(
        [
            item
            for item in transitions
            if item["transition_type"] != "disappeared"
        ],
        key=transition_size,
        reverse=True,
    )

    return {
        "previous_run_id": previous_run["run_id"],
        "current_run_id": current_run["run_id"],
        "previous_article_count": previous_run["article_count"],
        "current_article_count": current_run["article_count"],
        "previous_cluster_count": previous_run["cluster_count"],
        "current_cluster_count": current_run["cluster_count"],
        "article_count_delta": (
            current_run["article_count"] - previous_run["article_count"]
        ),
        "cluster_count_delta": (
            current_run["cluster_count"] - previous_run["cluster_count"]
        ),
        "previous_noise_ratio": previous_run["noise_ratio"],
        "current_noise_ratio": current_run["noise_ratio"],
        "noise_ratio_delta": (
            current_run["noise_ratio"] - previous_run["noise_ratio"]
        ),
        "continuing_topic_count": len(continuing),
        "newly_visible_topic_count": len(newly_visible),
        "reframed_topic_count": len(reframed),
        "disappeared_topic_count": len(disappeared),
        "top_growing_topics": [
            {
                "topic_name": item["topic_name"],
                "parent_topic_name": item.get("parent_topic_name"),
                "parent_size": item.get("parent_size"),
                "child_size": item.get("child_size"),
                "size_delta": item.get("size_delta"),
            }
            for item in growing[:3]
        ],
        "top_declining_topics": [
            {
                "topic_name": item["topic_name"],
                "parent_topic_name": item.get("parent_topic_name"),
                "parent_size": item.get("parent_size"),
                "child_size": item.get("child_size"),
                "size_delta": item.get("size_delta"),
            }
            for item in declining[:3]
        ],
        "largest_current_topics": [
            {
                "topic_name": item["topic_name"],
                "topic_size": transition_size(item),
                "transition_type": item["transition_type"],
                "trend": item["trend"],
            }
            for item in largest_current_topics[:5]
        ],
        "notable_new_topics": [
            {
                "topic_name": item["topic_name"],
                "topic_size": transition_size(item),
            }
            for item in newly_visible[:3]
        ],
        "editorial_topic_names": [
            item["topic_name"]
            for item in editorial_topics
        ],
    }


def build_mistral_video_payload(
    conn,
    child_run_id: int,
    target_duration_seconds: int = DEFAULT_TARGET_DURATION_SECONDS,
    max_topics: int = DEFAULT_MAX_TOPICS,
    headlines_per_topic: int = DEFAULT_HEADLINES_PER_TOPIC,
) -> dict[str, Any]:
    """
    Формирует входной payload для Mistral.

    Функция:
    - не вызывает Mistral API;
    - не записывает и не изменяет данные в БД;
    - должна вызываться после сохранения cluster_names и cluster_lineage;
    - должна вызываться после commit транзакции pipeline.

    Результат передаётся в функцию формирования Mistral HTTP request.
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
        "format": "news agenda recap",
        "voice_style": "calm professional news narrator",
        "audience": "international English-speaking audience",
        "aspect_ratio": "16:9",
        "visual_style": "cinematic editorial news documentary",
    }

    constraints = {
        "must_use_only_input_data": True,
        "do_not_invent_facts": True,
        "do_not_interpret_metrics_without_thresholds": True,
        "headlines_are_source_material_not_verified_claims": True,
        "use_editorial_topics_as_primary_script_source": True,
        "disappeared_topics_are_context_only": True,
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
            "agenda_summary": None,
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
            "parent_run_id": parent_run_id,
            "child_run_id": child_run_id,
            "comparison_type": "adjacent_completed_runs",
            "lineage_edge_count": len(lineage_edges),
        },
        "previous_run": parent_run,
        "current_run": child_run,
        "agenda_summary": agenda_summary,
        "topic_transitions": transitions,
        "editorial_topics": editorial_topics,
        "constraints": constraints,
    }