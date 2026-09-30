from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


WIDTH = 1920
HEIGHT = 1080
FPS = 30
CRF = "20"
PRESET = "medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a clean 16:9 documentary-style news video from "
            "scene audio and selected real assets."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to render_manifest_<run_id>.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for scene clips and final MP4",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError("Manifest root must be an object")

    canvas = data.get("canvas")
    if canvas != {"width": WIDTH, "height": HEIGHT, "fps": FPS}:
        raise RuntimeError(
            f"Expected canvas {WIDTH}x{HEIGHT}@{FPS}, got {canvas!r}"
        )

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Manifest scenes must be a non-empty array")

    expected_numbers = list(range(1, len(scenes) + 1))
    actual_numbers = [scene.get("scene_number") for scene in scenes]

    if actual_numbers != expected_numbers:
        raise RuntimeError(
            f"Scene numbering mismatch: expected={expected_numbers}, "
            f"actual={actual_numbers}"
        )

    for scene in scenes:
        audio_path = Path(str(scene.get("audio_path", "")))
        assets = scene.get("assets")
        duration = scene.get("actual_duration_seconds")

        if not audio_path.is_file():
            raise RuntimeError(
                f"Scene {scene['scene_number']} audio not found: "
                f"{audio_path}"
            )

        if not isinstance(assets, list) or not assets:
            raise RuntimeError(
                f"Scene {scene['scene_number']} has no assets"
            )

        if not isinstance(duration, (int, float)) or duration <= 0:
            raise RuntimeError(
                f"Scene {scene['scene_number']} has invalid duration"
            )

        for asset in assets:
            if not Path(str(asset)).is_file():
                raise RuntimeError(
                    f"Scene {scene['scene_number']} asset not found: "
                    f"{asset}"
                )

    return data


def duration_frames(seconds: float) -> int:
    return max(1, round(seconds * FPS))


def still_filter(frame_count: int) -> str:
    return (
        "scale=2400:1350:force_original_aspect_ratio=increase,"
        "crop=2400:1350,"
        "zoompan="
        "z='min(zoom+0.00055,1.12)':"
        f"d={frame_count}:"
        "x='iw/2-(iw/zoom/2)':"
        "y='ih/2-(ih/zoom/2)':"
        f"s={WIDTH}x{HEIGHT}:"
        f"fps={FPS},"
        "format=yuv420p,"
        "setsar=1"
    )


def render_single_asset_scene(
    *,
    asset_path: Path,
    audio_path: Path,
    duration: float,
    output_path: Path,
) -> None:
    frames = duration_frames(duration)

    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(asset_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        f"[0:v]{still_filter(frames)}[v]",
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    run(command)


def render_multi_asset_video(
    *,
    assets: list[Path],
    duration: float,
    output_path: Path,
) -> None:
    asset_count = len(assets)
    segment_duration = duration / asset_count
    segment_frames = duration_frames(segment_duration)

    command = ["ffmpeg", "-y"]

    for asset in assets:
        command.extend(["-loop", "1", "-i", str(asset)])

    filters: list[str] = []

    for index in range(asset_count):
        filters.append(
            f"[{index}:v]{still_filter(segment_frames)}[v{index}]"
        )

    concat_inputs = "".join(f"[v{index}]" for index in range(asset_count))
    filters.append(
        f"{concat_inputs}concat=n={asset_count}:v=1:a=0,"
        f"trim=duration={duration:.6f},"
        "setpts=PTS-STARTPTS[vout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-frames:v",
            str(duration_frames(duration)),
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            CRF,
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    run(command)


def mux_scene_audio(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    run(command)


def render_scene(
    scene: dict[str, Any],
    scenes_dir: Path,
) -> Path:
    scene_number = int(scene["scene_number"])
    duration = float(scene["actual_duration_seconds"])
    audio_path = Path(scene["audio_path"])
    assets = [Path(asset) for asset in scene["assets"]]

    output_path = scenes_dir / f"scene_{scene_number:03d}.mp4"

    print()
    print(
        f"Rendering scene {scene_number}: "
        f"{duration:.3f}s, assets={len(assets)}"
    )

    if len(assets) == 1:
        render_single_asset_scene(
            asset_path=assets[0],
            audio_path=audio_path,
            duration=duration,
            output_path=output_path,
        )
        return output_path

    silent_video_path = scenes_dir / (
        f"scene_{scene_number:03d}_silent.mp4"
    )

    render_multi_asset_video(
        assets=assets,
        duration=duration,
        output_path=silent_video_path,
    )

    mux_scene_audio(
        video_path=silent_video_path,
        audio_path=audio_path,
        output_path=output_path,
    )

    silent_video_path.unlink(missing_ok=True)
    return output_path


def write_concat_file(scene_paths: list[Path], output_path: Path) -> None:
    lines = []

    for path in scene_paths:
        escaped = str(path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def concat_final_video(
    *,
    concat_file: Path,
    output_path: Path,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    run(command)


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    manifest = load_manifest(args.manifest)
    run_id = int(manifest["run_id"])

    render_dir = args.output_dir
    scenes_dir = render_dir / "scenes"

    render_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_paths = [
        render_scene(scene, scenes_dir)
        for scene in manifest["scenes"]
    ]

    concat_file = render_dir / "scenes_concat.txt"
    final_path = render_dir / f"final_{run_id}.mp4"

    write_concat_file(scene_paths, concat_file)
    concat_final_video(
        concat_file=concat_file,
        output_path=final_path,
    )

    report = {
        "run_id": run_id,
        "manifest": str(args.manifest),
        "final_video": str(final_path),
        "scene_files": [str(path) for path in scene_paths],
        "expected_duration_seconds": round(
            sum(
                float(scene["actual_duration_seconds"])
                for scene in manifest["scenes"]
            ),
            3,
        ),
        "final_probe": probe(final_path),
    }

    report_path = render_dir / "render_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
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