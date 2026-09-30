import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cartesia import Cartesia


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Cartesia TTS WAV file from a scene manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to scene_manifest_<run_id>.json",
    )
    parser.add_argument(
        "--scene-number",
        type=int,
        required=True,
        help="Positive scene number to synthesize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for WAV and metadata JSON",
    )
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def load_scene(
    manifest_path: Path,
    scene_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest was not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise RuntimeError("Manifest root must be a JSON object")

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("Manifest field scenes must be a JSON array")

    for scene in scenes:
        if (
            isinstance(scene, dict)
            and scene.get("scene_number") == scene_number
        ):
            return manifest, scene

    raise RuntimeError(f"Scene {scene_number} was not found in manifest")


def probe_duration_seconds(audio_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found in PATH")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    duration_text = result.stdout.strip()

    if not duration_text:
        raise RuntimeError("ffprobe returned an empty duration")

    return round(float(duration_text), 3)


def main() -> int:
    args = parse_args()

    if args.scene_number < 1:
        raise RuntimeError("--scene-number must be positive")

    api_key = required_env("CARTESIA_API_KEY")
    model_id = required_env("CARTESIA_TTS_MODEL")
    voice_id = required_env("CARTESIA_VOICE_ID")

    manifest, scene = load_scene(
        args.manifest,
        args.scene_number,
    )

    run_id = manifest.get("run_id")
    language = str(manifest.get("language", "en")).strip() or "en"
    narration = str(scene.get("narration", "")).strip()

    if not isinstance(run_id, int):
        raise RuntimeError("Manifest field run_id must be an integer")

    if not narration:
        raise RuntimeError("Scene narration is empty")

    target_duration = scene.get("target_duration_seconds")

    if not isinstance(target_duration, int) or target_duration <= 0:
        raise RuntimeError(
            "Scene target_duration_seconds must be a positive integer"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = args.output_dir / (
        f"scene_{args.scene_number:03d}.wav"
    )
    metadata_path = args.output_dir / (
        f"scene_{args.scene_number:03d}_tts.json"
    )

    client = Cartesia(api_key=api_key)

    response = client.tts.generate(
        model_id=model_id,
        transcript=narration,
        voice={
            "mode": "id",
            "id": voice_id,
        },
        output_format={
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 44100,
        },
        language=language,
    )

    audio_bytes = response.read()

    if not audio_bytes:
        raise RuntimeError("Cartesia returned an empty audio response")

    audio_path.write_bytes(audio_bytes)

    actual_duration = probe_duration_seconds(audio_path)
    duration_delta = round(
        actual_duration - target_duration,
        3,
    )

    metadata = {
        "run_id": run_id,
        "scene_number": args.scene_number,
        "language": language,
        "model_id": model_id,
        "voice_id": voice_id,
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": actual_duration,
        "duration_delta_seconds": duration_delta,
        "word_count": len(narration.split()),
        "audio_file": audio_path.name,
        "narration": narration,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "scene_number": args.scene_number,
                "target_duration_seconds": target_duration,
                "actual_duration_seconds": actual_duration,
                "duration_delta_seconds": duration_delta,
                "audio_file": str(audio_path),
                "metadata_file": str(metadata_path),
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)