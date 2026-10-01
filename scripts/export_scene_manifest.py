from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import psycopg2.extras

from clustering.offline import get_conn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a validated Mistral video script from PostgreSQL "
            "to the Cartesia scene-manifest format."
        )
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_script(conn, run_id: int) -> dict[str, Any]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                run_id,
                status,
                validation_status,
                title,
                description,
                language,
                estimated_duration_seconds,
                editorial_angle,
                script_json
            FROM mistral_video_scripts
            WHERE run_id = %s
            LIMIT 1
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No Mistral video script exists for run_id={run_id}"
        )

    if row["status"] != "success":
        raise RuntimeError(
            "Mistral video script is not successful: "
            f"run_id={run_id}, status={row['status']!r}"
        )

    if row["validation_status"] != "passed":
        raise RuntimeError(
            "Mistral video script did not pass validation: "
            f"run_id={run_id}, validation_status={row['validation_status']!r}"
        )

    script = row["script_json"]

    if not isinstance(script, dict):
        raise RuntimeError(
            f"Mistral script_json is not an object for run_id={run_id}"
        )

    return dict(row)


def build_manifest(
    *,
    run_id: int,
    script_row: dict[str, Any],
) -> dict[str, Any]:
    script_json = script_row["script_json"]
    scenes = script_json.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError(
            f"Mistral script contains no scenes for run_id={run_id}"
        )

    ordered_scenes = sorted(
        scenes,
        key=lambda item: int(item["scene_number"]),
    )

    expected_numbers = list(range(1, len(ordered_scenes) + 1))
    actual_numbers = [
        int(scene["scene_number"])
        for scene in ordered_scenes
    ]

    if actual_numbers != expected_numbers:
        raise RuntimeError(
            "Invalid Mistral scene order: "
            f"expected={expected_numbers}, actual={actual_numbers}"
        )

    exported_scenes: list[dict[str, Any]] = []

    for scene in ordered_scenes:
        narration = str(scene.get("narration", "")).strip()
        duration = scene.get("duration_seconds")

        if not narration:
            raise RuntimeError(
                f"Scene {scene.get('scene_number')} has empty narration"
            )

        if not isinstance(duration, int) or duration <= 0:
            raise RuntimeError(
                "Scene has invalid duration_seconds: "
                f"scene={scene.get('scene_number')}, duration={duration!r}"
            )

        exported_scenes.append(
            {
                "scene_number": int(scene["scene_number"]),
                "target_duration_seconds": duration,
                "narration": narration,
                "visual_type": scene.get("visual_type"),
                "visual_prompt": scene.get("visual_prompt"),
                "topic_references": scene.get("topic_references", []),
            }
        )

    return {
        "run_id": run_id,
        "language": str(script_row["language"] or "en").strip() or "en",
        "title": script_row["title"],
        "description": script_row["description"],
        "editorial_angle": script_row["editorial_angle"],
        "estimated_duration_seconds": int(
            script_row["estimated_duration_seconds"]
        ),
        "scene_count": len(exported_scenes),
        "scenes": exported_scenes,
    }


def main() -> int:
    args = parse_args()

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")

    conn = get_conn()

    try:
        script_row = load_script(conn, args.run_id)
    finally:
        conn.close()

    manifest = build_manifest(
        run_id=args.run_id,
        script_row=script_row,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output": str(args.output),
                "scene_count": manifest["scene_count"],
                "estimated_duration_seconds": (
                    manifest["estimated_duration_seconds"]
                ),
                "title": manifest["title"],
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())