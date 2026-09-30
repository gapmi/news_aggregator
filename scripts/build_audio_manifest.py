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