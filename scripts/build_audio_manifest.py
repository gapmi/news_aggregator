import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    args = parser.parse_args()

    items = []

    for metadata_path in sorted(
        args.audio_dir.glob("scene_*_tts.json")
    ):
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        items.append(item)

    items.sort(key=lambda item: item["scene_number"])

    expected_numbers = list(range(1, len(items) + 1))
    actual_numbers = [item["scene_number"] for item in items]

    if actual_numbers != expected_numbers:
        raise SystemExit(
            f"Unexpected scene sequence: {actual_numbers}"
        )

    total_actual = round(
        sum(item["actual_duration_seconds"] for item in items),
        3,
    )
    total_target = sum(
        item["target_duration_seconds"] for item in items
    )

    output = {
        "run_id": args.run_id,
        "scene_count": len(items),
        "total_target_duration_seconds": total_target,
        "total_actual_duration_seconds": total_actual,
        "total_delta_seconds": round(
            total_actual - total_target,
            3,
        ),
        "scenes": items,
    }

    if not items:
        raise SystemExit("No scene TTS metadata files were found")

    transcript_sections = [
        f"Run {args.run_id} — narration by scene"
    ]
    full_voiceover_parts = []

    for item in items:
        if item.get("run_id") != args.run_id:
            raise SystemExit(
                "TTS metadata run_id mismatch for scene "
                f"{item.get('scene_number')}: "
                f"{item.get('run_id')!r}"
            )

        narration = item.get("narration")

        if not isinstance(narration, str) or not narration.strip():
            raise SystemExit(
                "Missing narration for scene "
                f"{item.get('scene_number')}"
            )

        transcript_sections.append(
            f"Scene {item['scene_number']:03d}\n"
            f"Audio: {item['audio_file']}\n\n"
            f"{narration}"
        )
        full_voiceover_parts.append(narration)

    transcript_path = args.audio_dir / "narration_by_scene.txt"
    full_voiceover_path = args.audio_dir / "full_voiceover.txt"

    transcript_path.write_text(
        "\n\n".join(transcript_sections) + "\n",
        encoding="utf-8",
    )
    full_voiceover_path.write_text(
        " ".join(full_voiceover_parts),
        encoding="utf-8",
    )

    output["narration_by_scene_file"] = transcript_path.name
    output["full_voiceover_file"] = full_voiceover_path.name

    output_path = args.audio_dir / "audio_manifest.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "scene_count": len(items),
                "total_target_duration_seconds": total_target,
                "total_actual_duration_seconds": total_actual,
                "total_delta_seconds": output["total_delta_seconds"],
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())