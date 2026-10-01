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
        description="Render a timestamp-safe news video from scene assets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH")


def duration_frames(seconds: float) -> int:
    return max(1, round(seconds * FPS))


def still_filter(frame_count: int) -> str:
    return (
        "scale=2400:1350:force_original_aspect_ratio=increase,"
        "crop=2400:1350,"
        f"fps={FPS},"
        "format=yuv420p,"
        "setsar=1,"
        "setpts=N/(30*TB)"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError("Manifest root must be an object")

    if data.get("canvas") != {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
    }:
        raise RuntimeError("Unexpected canvas settings")

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Manifest has no scenes")

    expected = list(range(1, len(scenes) + 1))
    actual = [scene.get("scene_number") for scene in scenes]

    if actual != expected:
        raise RuntimeError(
            f"Invalid scene order: expected={expected}, actual={actual}"
        )

    for scene in scenes:
        if not Path(str(scene.get("audio_path", ""))).is_file():
            raise RuntimeError(
                f"Audio missing for scene {scene.get('scene_number')}"
            )

        assets = scene.get("assets")
        if not isinstance(assets, list) or not assets:
            raise RuntimeError(
                f"No assets for scene {scene.get('scene_number')}"
            )

        for asset in assets:
            if not Path(str(asset)).is_file():
                raise RuntimeError(f"Asset missing: {asset}")

    return data


def render_asset_segment(
    *,
    asset_path: Path,
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
        "-filter_complex",
        f"[0:v]{still_filter(frames)}[v]",
        "-map",
        "[v]",
        "-frames:v",
        str(frames),
        "-r",
        str(FPS),
        "-video_track_timescale",
        "90000",
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

    run(command)


def write_concat_file(paths: list[Path], output_path: Path) -> None:
    lines = []

    for path in paths:
        escaped = str(path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def concat_video_segments(
    *,
    segment_paths: list[Path],
    output_path: Path,
    temp_dir: Path,
) -> None:
    concat_file = temp_dir / "segments_concat.txt"
    write_concat_file(segment_paths, concat_file)

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
        "-r",
        str(FPS),
        "-video_track_timescale",
        "90000",
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

    run(command)


def mux_audio(
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
        "-r",
        str(FPS),
        "-video_track_timescale",
        "90000",
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
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    run(command)


def render_scene(scene: dict[str, Any], scenes_dir: Path) -> Path:
    scene_number = int(scene["scene_number"])
    total_duration = float(scene["actual_duration_seconds"])
    audio_path = Path(scene["audio_path"])
    assets = [Path(str(item)) for item in scene["assets"]]

    scene_dir = scenes_dir / f"scene_{scene_number:03d}_parts"
    scene_dir.mkdir(parents=True, exist_ok=True)

    segment_duration = total_duration / len(assets)
    segment_paths: list[Path] = []

    print(
        f"Rendering scene={scene_number}, "
        f"duration={total_duration:.3f}, assets={len(assets)}",
        flush=True,
    )

    for index, asset_path in enumerate(assets, start=1):
        segment_path = scene_dir / f"part_{index:02d}.mp4"

        render_asset_segment(
            asset_path=asset_path,
            duration=segment_duration,
            output_path=segment_path,
        )

        segment_paths.append(segment_path)

    silent_path = scenes_dir / f"scene_{scene_number:03d}_silent.mp4"

    concat_video_segments(
        segment_paths=segment_paths,
        output_path=silent_path,
        temp_dir=scene_dir,
    )

    scene_path = scenes_dir / f"scene_{scene_number:03d}.mp4"

    mux_audio(
        video_path=silent_path,
        audio_path=audio_path,
        output_path=scene_path,
    )

    return scene_path


def concat_final(
    *,
    scene_paths: list[Path],
    output_path: Path,
    render_dir: Path,
) -> None:
    concat_file = render_dir / "scenes_concat.txt"
    write_concat_file(scene_paths, concat_file)

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
        "-r",
        str(FPS),
        "-video_track_timescale",
        "90000",
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
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:"
            "stream=codec_name,codec_type,width,height,"
            "r_frame_rate,time_base",
            "-of",
            "json",
            str(path),
        ],
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

    final_path = render_dir / f"final_{run_id}.mp4"

    concat_final(
        scene_paths=scene_paths,
        output_path=final_path,
        render_dir=render_dir,
    )

    expected_duration = round(
        sum(
            float(scene["actual_duration_seconds"])
            for scene in manifest["scenes"]
        ),
        3,
    )

    report = {
        "run_id": run_id,
        "expected_duration_seconds": expected_duration,
        "final_video": str(final_path),
        "final_probe": probe(final_path),
        "scene_files": [str(path) for path in scene_paths],
    }

    report_path = render_dir / "render_report_v2.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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