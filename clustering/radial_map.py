from __future__ import annotations


import math
from dataclasses import dataclass


import numpy as np
import psycopg2.extras
from psycopg2.extras import execute_values



RING_KEYS = ["core", "mid", "edge", "outlier_risk"]
RING_LABELS = {
    "core": "Core",
    "mid": "Mid",
    "edge": "Edge",
    "outlier_risk": "Outlier risk",
}

# Sentinel used for articles that are not assigned to any real subcluster
# (true HDBSCAN noise / outliers). Kept distinct from any real subcluster
# label (which are always >= 0) so it never shares a sector with a real
# subcluster labeled "0".
UNASSIGNED_LABEL = -1



@dataclass
class ClusterArticleEmbedding:
    article_id: int
    title: str | None
    url: str | None
    published: object | None
    source: str | None
    embedding: np.ndarray



def _to_numpy_vector(value) -> np.ndarray | None:
    if value is None:
        return None


    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)


    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)


    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=np.float32)


    if hasattr(value, "to_list"):
        return np.asarray(value.to_list(), dtype=np.float32)


    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parts = [p.strip() for p in stripped[1:-1].split(",") if p.strip()]
            return np.asarray([float(p) for p in parts], dtype=np.float32)


    return np.asarray(list(value), dtype=np.float32)



def _normalize_distances(distances: np.ndarray) -> np.ndarray:
    if len(distances) == 0:
        return distances


    d_min = float(np.min(distances))
    d_max = float(np.max(distances))


    if abs(d_max - d_min) < 1e-12:
        return np.zeros(len(distances), dtype=np.float32)


    return ((distances - d_min) / (d_max - d_min)).astype(np.float32)



def _distance_stats(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {
            "distance_min": None,
            "distance_max": None,
            "distance_mean": None,
            "distance_median": None,
        }


    return {
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_median": float(np.median(distances)),
    }



def _assign_ring_index(distance_quantile: float) -> int:
    if distance_quantile <= 0.25:
        return 0
    if distance_quantile <= 0.50:
        return 1
    if distance_quantile <= 0.75:
        return 2
    return 3



def _compute_sector_layout(
    article_rows: list[ClusterArticleEmbedding],
    assignments: dict[int, dict],
    subclusters: list[dict],
) -> list[dict]:
    """
    Single source of truth for sector geometry.

    Sector angular bounds are derived from the ACTUAL number of points
    that will be rendered in each sector (computed here, from
    article_rows + assignments), never from cluster_subclusters.size.
    This guarantees the drawn sector width and the number of points
    inside it are always proportional, even when:
      - some articles were filtered out for missing embeddings, or
      - some articles are not assigned to any subcluster (true outliers).

    Unassigned articles get their own dedicated "unassigned" sector
    instead of silently falling into subcluster label 0's slot.
    """
    sorted_subclusters = sorted(subclusters, key=lambda x: int(x["label"]))

    counts_by_label: dict[int, int] = {}
    for article in article_rows:
        assignment = assignments.get(article.article_id)
        label_int = (
            int(assignment["subcluster_label_int"])
            if assignment is not None
            else UNASSIGNED_LABEL
        )
        counts_by_label[label_int] = counts_by_label.get(label_int, 0) + 1

    subcluster_id_by_label = {
        int(row["label"]): row["id"] for row in sorted_subclusters
    }

    ordered_labels = [int(row["label"]) for row in sorted_subclusters]
    if UNASSIGNED_LABEL in counts_by_label:
        ordered_labels.append(UNASSIGNED_LABEL)

    # Drop labels with zero actually-rendered points so they don't eat
    # angular width for nothing (e.g. all their articles lacked embeddings).
    ordered_labels = [
        label for label in ordered_labels if counts_by_label.get(label, 0) > 0
    ]

    total = sum(counts_by_label.get(label, 0) for label in ordered_labels)

    layout: list[dict] = []
    start = 0.0

    for idx, label_int in enumerate(ordered_labels):
        count = counts_by_label.get(label_int, 0)
        span = 360.0 * count / total if total > 0 else 0.0
        end = 360.0 if idx == len(ordered_labels) - 1 else start + span

        if label_int == UNASSIGNED_LABEL:
            sector_key = "unassigned"
            label = "Unassigned"
            subcluster_id = None
        else:
            sector_key = f"subcluster_{label_int}"
            label = f"Subcluster {label_int}"
            subcluster_id = subcluster_id_by_label.get(label_int)

        layout.append(
            {
                "label_int": label_int,
                "sector_index": idx,
                "sector_key": sector_key,
                "label": label,
                "subcluster_id": subcluster_id,
                "start": start,
                "end": end,
                "count": count,
            }
        )
        start = end

    if not layout:
        layout.append(
            {
                "label_int": 0,
                "sector_index": 0,
                "sector_key": "subcluster_0",
                "label": "Subcluster 0",
                "subcluster_id": None,
                "start": 0.0,
                "end": 360.0,
                "count": 0,
            }
        )

    return layout



def _build_point_rows(
    cluster_id: int,
    radial_map_id: int,
    article_rows: list[ClusterArticleEmbedding],
    centroid: np.ndarray,
    assignments: dict[int, dict] | None = None,
    subclusters: list[dict] | None = None,
) -> tuple[list[tuple], dict, list[dict]]:
    if not article_rows:
        return (
            [],
            {
                "article_count": 0,
                "core_count": 0,
                "mid_count": 0,
                "edge_count": 0,
                "outlier_risk_count": 0,
                "questionable_count": 0,
                "outlier_count": 0,
                "subcluster_count": 0,
                "sector_count": 0,
                "unassigned_subcluster_count": 0,
                **_distance_stats(np.array([], dtype=np.float32)),
            },
            [],
        )


    assignments = assignments or {}
    subclusters = subclusters or []


    sector_layout = _compute_sector_layout(article_rows, assignments, subclusters)


    sector_index_by_label = {row["label_int"]: row["sector_index"] for row in sector_layout}
    sector_bounds_by_label = {row["label_int"]: (row["start"], row["end"]) for row in sector_layout}
    sector_key_by_label = {row["label_int"]: row["sector_key"] for row in sector_layout}


    vectors = np.vstack([row.embedding for row in article_rows]).astype(np.float32)
    distances = np.linalg.norm(vectors - centroid, axis=1).astype(np.float32)
    radius_values = _normalize_distances(distances)


    order = np.argsort(distances)
    distance_quantiles = np.zeros(len(article_rows), dtype=np.float32)


    if len(article_rows) == 1:
        distance_quantiles[0] = 0.0
    else:
        for rank, idx in enumerate(order):
            distance_quantiles[idx] = rank / (len(article_rows) - 1)


    articles_by_sector_label: dict[int, list[int]] = {}
    for idx, article in enumerate(article_rows):
        assignment = assignments.get(article.article_id)
        label_int = (
            int(assignment["subcluster_label_int"])
            if assignment is not None
            else UNASSIGNED_LABEL
        )
        articles_by_sector_label.setdefault(label_int, []).append(idx)


    position_in_sector: dict[int, tuple[int, int]] = {}
    for label_int, indices in articles_by_sector_label.items():
        for pos, article_idx in enumerate(indices):
            position_in_sector[article_idx] = (pos, len(indices))


    point_rows: list[tuple] = []


    core_count = 0
    mid_count = 0
    edge_count = 0
    outlier_risk_count = 0


    n = len(article_rows)


    for idx, article in enumerate(article_rows):
        distance_to_centroid = float(distances[idx])
        distance_quantile = float(distance_quantiles[idx])


        ring_index = _assign_ring_index(distance_quantile)
        ring_key = RING_KEYS[ring_index]


        is_core = ring_key == "core"
        is_edge = ring_key == "edge"
        is_outlier_risk = ring_key == "outlier_risk"
        is_outlier = False
        is_questionable = False


        if is_core:
            core_count += 1
        elif ring_key == "mid":
            mid_count += 1
        elif is_edge:
            edge_count += 1
        elif is_outlier_risk:
            outlier_risk_count += 1


        assignment = assignments.get(article.article_id)


        if assignment is not None:
            label_int = int(assignment["subcluster_label_int"])
            subcluster_id = str(assignment["subcluster_id"])
            subcluster_label = f"Subcluster {label_int}"
            membership_confidence = assignment["probability"]
            outlier_score = assignment["outlier_score"]
        else:
            label_int = UNASSIGNED_LABEL
            subcluster_id = None
            subcluster_label = None
            membership_confidence = None
            outlier_score = None


        sector_index = int(sector_index_by_label.get(label_int, 0))
        sector_key = sector_key_by_label.get(label_int, "subcluster_0")


        sector_start, sector_end = sector_bounds_by_label.get(label_int, (0.0, 360.0))
        sector_span = max(0.0, sector_end - sector_start)


        pos, total_in_sector = position_in_sector.get(idx, (0, 1))


        if total_in_sector <= 1:
            angle_deg = sector_start + sector_span / 2.0
        else:
            angle_deg = sector_start + sector_span * ((pos + 0.5) / total_in_sector)


        # -90 aligns with the frontend's "0deg = top, clockwise" SVG
        # convention (kept for consistency; the frontend now positions
        # points from radius + angle_deg only, but any other consumer
        # of raw x/y gets a geometrically correct value too).
        angle_rad = math.radians(angle_deg - 90.0)


        radius = float(radius_values[idx])
        x = float(radius * math.cos(angle_rad))
        y = float(radius * math.sin(angle_rad))


        point_rows.append(
            (
                radial_map_id,
                cluster_id,
                article.article_id,
                idx,
                x,
                y,
                radius,
                angle_deg,
                ring_index,
                ring_key,
                sector_index,
                sector_key,
                subcluster_id,
                subcluster_label,
                distance_to_centroid,
                distance_quantile,
                membership_confidence,
                outlier_score,
                None,
                None,
                None,
                is_core,
                is_edge,
                is_outlier_risk,
                is_outlier,
                is_questionable,
            )
        )


    assigned_article_count = sum(
        1 for article in article_rows
        if article.article_id in assignments
    )


    real_subcluster_ids = {
        assignments[article.article_id]["subcluster_id"]
        for article in article_rows
        if article.article_id in assignments
    }


    stats = {
        "article_count": n,
        "core_count": core_count,
        "mid_count": mid_count,
        "edge_count": edge_count,
        "outlier_risk_count": outlier_risk_count,
        "questionable_count": 0,
        "outlier_count": 0,
        "subcluster_count": len(real_subcluster_ids) if real_subcluster_ids else (1 if n > 0 else 0),
        "sector_count": len(sector_layout),
        "unassigned_subcluster_count": n - assigned_article_count if assignments else 0,
        **_distance_stats(distances),
    }


    return point_rows, stats, sector_layout



def _load_cluster_rows(conn, run_id: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.id,
                c.run_id,
                c.centroid
            FROM clusters c
            WHERE c.run_id = %s
            ORDER BY c.id
            """,
            (run_id,),
        )
        return cur.fetchall()



def _load_cluster_articles(conn, cluster_id: int) -> list[ClusterArticleEmbedding]:
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
            JOIN articles a ON a.id = ca.article_id
            WHERE ca.cluster_id = %s
              AND a.embedding IS NOT NULL
            ORDER BY a.published DESC NULLS LAST, a.id DESC
            """,
            (cluster_id,),
        )
        rows = cur.fetchall()


    result: list[ClusterArticleEmbedding] = []
    for row in rows:
        embedding = _to_numpy_vector(row["embedding"])
        if embedding is None:
            continue


        result.append(
            ClusterArticleEmbedding(
                article_id=row["id"],
                title=row["title"],
                url=row["url"],
                published=row["published"],
                source=row["source"],
                embedding=embedding,
            )
        )


    return result



def _load_subclusters(conn, cluster_id: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                label,
                size
            FROM cluster_subclusters
            WHERE cluster_id = %s
            ORDER BY label
            """,
            (cluster_id,),
        )
        return cur.fetchall()



def _load_subcluster_assignments(conn, cluster_id: int) -> dict[int, dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                csa.article_id,
                csa.subcluster_id,
                csa.probability,
                csa.outlier_score,
                cs.label AS subcluster_label_int
            FROM cluster_subcluster_articles csa
            JOIN cluster_subclusters cs ON cs.id = csa.subcluster_id
            WHERE csa.cluster_id = %s
            """,
            (cluster_id,),
        )
        rows = cur.fetchall()


    return {
        int(row["article_id"]): {
            "subcluster_id": int(row["subcluster_id"]),
            "subcluster_label_int": int(row["subcluster_label_int"]),
            "probability": row["probability"],
            "outlier_score": row["outlier_score"],
        }
        for row in rows
    }



def _delete_existing_radial_map(conn, cluster_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM cluster_radial_maps
            WHERE cluster_id = %s
            """,
            (cluster_id,),
        )



def _insert_radial_map(conn, cluster_id: int, run_id: int, stats: dict) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cluster_radial_maps (
                cluster_id,
                run_id,
                version,
                ring_mode,
                ring_count,
                sector_mode,
                sector_count,
                article_count,
                core_count,
                mid_count,
                edge_count,
                outlier_risk_count,
                questionable_count,
                outlier_count,
                subcluster_count,
                unassigned_subcluster_count,
                distance_min,
                distance_max,
                distance_mean,
                distance_median
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                cluster_id,
                run_id,
                1,
                "quantiles",
                4,
                "subclusters",
                stats["sector_count"],
                stats["article_count"],
                stats["core_count"],
                stats["mid_count"],
                stats["edge_count"],
                stats["outlier_risk_count"],
                stats["questionable_count"],
                stats["outlier_count"],
                stats["subcluster_count"],
                stats["unassigned_subcluster_count"],
                stats["distance_min"],
                stats["distance_max"],
                stats["distance_mean"],
                stats["distance_median"],
            ),
        )
        return cur.fetchone()[0]



def _insert_stub_rings(conn, radial_map_id: int, stats: dict) -> None:
    ring_counts = {
        "core": int(stats["core_count"]),
        "mid": int(stats["mid_count"]),
        "edge": int(stats["edge_count"]),
        "outlier_risk": int(stats["outlier_risk_count"]),
    }


    ring_rows = [
        (
            radial_map_id,
            0,
            "core",
            RING_LABELS["core"],
            0.00,
            0.25,
            0.00,
            0.25,
            ring_counts["core"],
        ),
        (
            radial_map_id,
            1,
            "mid",
            RING_LABELS["mid"],
            0.25,
            0.50,
            0.25,
            0.50,
            ring_counts["mid"],
        ),
        (
            radial_map_id,
            2,
            "edge",
            RING_LABELS["edge"],
            0.50,
            0.75,
            0.50,
            0.75,
            ring_counts["edge"],
        ),
        (
            radial_map_id,
            3,
            "outlier_risk",
            RING_LABELS["outlier_risk"],
            0.75,
            1.00,
            0.75,
            1.00,
            ring_counts["outlier_risk"],
        ),
    ]


    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO cluster_radial_rings (
                radial_map_id,
                ring_index,
                ring_key,
                label,
                quantile_start,
                quantile_end,
                radius_inner,
                radius_outer,
                article_count
            )
            VALUES %s
            """,
            ring_rows,
        )



def _insert_sectors_from_layout(
    conn, radial_map_id: int, sector_layout: list[dict]
) -> None:
    """
    Persist sectors using the SAME layout (bounds + counts) that was used
    to place the points, so the stored sector width is always exactly
    proportional to the number of points that landed inside it.
    """
    if not sector_layout:
        return


    rows = [
        (
            radial_map_id,
            row["sector_index"],
            row["sector_key"],
            row["label"],
            str(row["subcluster_id"]) if row["subcluster_id"] is not None else None,
            row["start"],
            row["end"],
            row["count"],
            None,
        )
        for row in sector_layout
    ]


    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO cluster_radial_sectors (
                radial_map_id,
                sector_index,
                sector_key,
                label,
                subcluster_id,
                start_angle_deg,
                end_angle_deg,
                article_count,
                color_key
            )
            VALUES %s
            """,
            rows,
        )



def _insert_point_rows(conn, point_rows: list[tuple]) -> None:
    if not point_rows:
        return


    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO cluster_radial_points (
                radial_map_id,
                cluster_id,
                article_id,
                article_index,
                x,
                y,
                radius,
                angle_deg,
                ring_index,
                ring_key,
                sector_index,
                sector_key,
                subcluster_id,
                subcluster_label,
                distance_to_centroid,
                distance_quantile,
                membership_confidence,
                outlier_score,
                nearest_neighbor_distance,
                nearest_alternative_cluster_id,
                nearest_alternative_cluster_distance,
                is_core,
                is_edge,
                is_outlier_risk,
                is_outlier,
                is_questionable
            )
            VALUES %s
            """,
            point_rows,
        )



def build_and_save_radial_map_for_cluster(conn, cluster_id: int, run_id: int, centroid_value) -> int:
    centroid = _to_numpy_vector(centroid_value)
    if centroid is None:
        _delete_existing_radial_map(conn, cluster_id)
        return 0


    article_rows = _load_cluster_articles(conn, cluster_id)
    subclusters = _load_subclusters(conn, cluster_id)
    assignments = _load_subcluster_assignments(conn, cluster_id) if subclusters else {}


    point_rows, stats, sector_layout = _build_point_rows(
        cluster_id=cluster_id,
        radial_map_id=-1,
        article_rows=article_rows,
        centroid=centroid,
        assignments=assignments,
        subclusters=subclusters,
    )


    _delete_existing_radial_map(conn, cluster_id)
    radial_map_id = _insert_radial_map(conn, cluster_id, run_id, stats)


    patched_point_rows = []
    for row in point_rows:
        row = list(row)
        row[0] = radial_map_id
        patched_point_rows.append(tuple(row))


    _insert_stub_rings(conn, radial_map_id, stats)
    _insert_sectors_from_layout(conn, radial_map_id, sector_layout)
    _insert_point_rows(conn, patched_point_rows)


    return stats["article_count"]



def build_and_save_radial_maps_for_run(conn, run_id: int) -> dict:
    cluster_rows = _load_cluster_rows(conn, run_id)


    cluster_count = 0
    point_count = 0


    for row in cluster_rows:
        cluster_count += 1
        point_count += build_and_save_radial_map_for_cluster(
            conn=conn,
            cluster_id=row["id"],
            run_id=row["run_id"],
            centroid_value=row["centroid"],
        )


    return {
        "cluster_count": cluster_count,
        "point_count": point_count,
    }