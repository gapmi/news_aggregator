import math
import argparse
import json
import shutil

import subprocess
import sys

from pathlib import Path
from typing import Any
from scripts.video_branding import build_branded_sequence
from scripts.video_profiles import (
    DEFAULT_VIDEO_PROFILE,
    VideoProfile,
    get_video_profile,
)

WIDTH = 2400
HEIGHT = 1350
FPS = 30
CRF = "20"
PRESET = "medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a timestamp-safe news video from scene assets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-branding",
        action="store_true",
        help="Render scenes without branding; intended for previews.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH")


def duration_frames(seconds: float) -> int:
    return max(1, round(seconds * FPS))


def still_filter(
    *,
    profile: VideoProfile,
    visual_type: str,
) -> str:
    width = profile.width
    height = profile.height

    if profile.name == "news_horizontal":
        return (
            f"[0:v]scale={width}:{height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={profile.fps},"
            "format=yuv420p,"
            "setsar=1,"
            f"setpts=N/({profile.fps}*TB)[v]"
        )

    if visual_type == "map":
        return (
            f"[0:v]scale={width}:{height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:"
            "color=0x0f1926,"
            f"fps={profile.fps},"
            "format=yuv420p,"
            "setsar=1,"
            f"setpts=N/({profile.fps}*TB)[v]"
        )

    return (
        "[0:v]split=2[background_source][foreground_source];"
        "[background_source]"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        "boxblur=20:2[background];"
        "[foreground_source]"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        "[foreground];"
        "[background][foreground]"
        "overlay=(W-w)/2:(H-h)/2,"
        f"fps={profile.fps},"
        "format=yuv420p,"
        "setsar=1,"
        f"setpts=N/({profile.fps}*TB)[v]"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError("Manifest root must be an object")

    profile = get_video_profile(
        data.get("profile", DEFAULT_VIDEO_PROFILE)
    )

    if data.get("canvas") != {
        "width": profile.width,
        "height": profile.height,
        "fps": profile.fps,
    }:
        raise RuntimeError(
            f"Unexpected canvas settings for profile={profile.name}"
        )

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
        duration = scene.get("actual_duration_seconds")

        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise RuntimeError(
                "Invalid duration for scene "
                f"{scene.get('scene_number')}: {duration!r}"
            )

        visual_type = scene.get("visual_type")

        if profile.name == "news_vertical":
            if visual_type not in {"photo", "map"}:
                raise RuntimeError(
                    "Vertical scene must have visual_type='photo' "
                    "or visual_type='map': "
                    f"scene={scene.get('scene_number')}"
                )
            
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
    frames: int,
    output_path: Path,
    profile: VideoProfile,
    visual_type: str,
) -> None:
    if frames < 1:
        raise ValueError("Segment must contain at least one frame")

    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(asset_path),
        "-filter_complex",
        still_filter(
            profile=profile,
            visual_type=visual_type,
        ),
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
    subtitles_path: Path,
    profile: VideoProfile,
) -> None:
    subtitle_file = (
        str(subtitles_path.resolve())
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "'\\''")
    )

    if profile.name == "news_vertical":
        subtitle_layout = (
            "MarginL=24,"
            "MarginR=24,"
            "MarginV=60"
        )
    else:
        subtitle_layout = (
            "MarginL=30,"
            "MarginR=30,"
            "MarginV=24"
        )

    subtitle_filter = (
        f"subtitles=filename='{subtitle_file}':"
        "force_style='"
        "FontName=DejaVu Sans,"
        "FontSize=20,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        f"{subtitle_layout}"
        "'"
    )

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
        "-vf",
        subtitle_filter,
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
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    run(command)


def write_scene_subtitles(
    *,
    narration: str,
    duration: float,
    output_path: Path,
    words_per_caption: int = 10,
) -> Path:
    words = narration.split()

    if not words:
        raise ValueError("Cannot create subtitles from empty narration")

    if not 0 < duration < float("inf"):
        raise ValueError("Subtitle duration must be finite and positive")

    # The timing is approximate: no word timestamps are available.
    if words_per_caption < 1:
        raise ValueError("words_per_caption must be positive")

    # The timing is approximate: no word timestamps are available.
    captions = [
        words[index:index + words_per_caption]
        for index in range(0, len(words), words_per_caption)
    ]

    total_ms = round(duration * 1000)

    if total_ms < len(captions):
        raise ValueError("Scene is too short for subtitle intervals")

    def timestamp(milliseconds: int) -> str:
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (
            f"{hours:02d}:{minutes:02d}:{seconds:02d},"
            f"{milliseconds:03d}"
        )

    sections: list[str] = []
    consumed_words = 0
    start_ms = 0

    for number, caption_words in enumerate(captions, start=1):
        consumed_words += len(caption_words)
        end_ms = round(total_ms * consumed_words / len(words))

        # Keep every word; split each caption into at most two lines.
        midpoint = (len(caption_words) + 1) // 2
        lines = [" ".join(caption_words[:midpoint])]

        if midpoint < len(caption_words):
            lines.append(" ".join(caption_words[midpoint:]))

        caption_text = "\n".join(lines)

        sections.append(
            f"{number}\n"
            f"{timestamp(start_ms)} --> {timestamp(end_ms)}\n"
            f"{caption_text}"
        )

        start_ms = end_ms

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )

    return output_path


def render_scene(
    scene: dict[str, Any],
    scenes_dir: Path,
    *,
    profile: VideoProfile,
) -> Path:
    scene_number = int(scene["scene_number"])
    total_duration = float(scene["actual_duration_seconds"])
    audio_path = Path(scene["audio_path"])
    assets = [Path(str(item)) for item in scene["assets"]]

    scene_dir = scenes_dir / f"scene_{scene_number:03d}_parts"
    scene_dir.mkdir(parents=True, exist_ok=True)

    visual_type = str(scene.get("visual_type") or "photo")
    if profile.name == "news_vertical":
        total_frames = math.ceil(total_duration * profile.fps)

        if total_frames < len(assets):
            raise ValueError(
                f"Scene {scene_number} has more assets than frames"
            )

        base_frames, extra_frames = divmod(
            total_frames,
            len(assets),
        )
        segment_frame_counts = [
            base_frames + (1 if index < extra_frames else 0)
            for index in range(len(assets))
        ]
    else:
        segment_duration = total_duration / len(assets)
        segment_frame_counts = [
            duration_frames(segment_duration)
            for _ in assets
        ]
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
            frames=segment_frame_counts[index - 1],
            output_path=segment_path,
            profile=profile,
            visual_type=visual_type,
        )

        segment_paths.append(segment_path)

    silent_path = scenes_dir / f"scene_{scene_number:03d}_silent.mp4"

    concat_video_segments(
        segment_paths=segment_paths,
        output_path=silent_path,
        temp_dir=scene_dir,
    )

    scene_path = scenes_dir / f"scene_{scene_number:03d}.mp4"

    subtitles_path = write_scene_subtitles(
        narration=str(scene.get("narration") or ""),
        duration=total_duration,
        output_path=scene_dir / "subtitles.srt",
        words_per_caption=(
            6 if profile.name == "news_vertical" else 10
        ),
    )

    mux_audio(
        video_path=silent_path,
        audio_path=audio_path,
        output_path=scene_path,
        subtitles_path=subtitles_path,
        profile=profile,
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
        "-ac",
        "2",
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
    profile = get_video_profile(
        manifest.get("profile", DEFAULT_VIDEO_PROFILE)
    )

    run_id = int(manifest["run_id"])

    render_dir = args.output_dir
    scenes_dir = render_dir / "scenes"

    render_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_paths = [
        render_scene(
            scene,
            scenes_dir,
            profile=profile,
        )
        for scene in manifest["scenes"]
    ]

    final_path = render_dir / f"final_{run_id}.mp4"

    if args.no_branding:
        sequence_paths = scene_paths
        branding_report = {
            "enabled": False,
            "extra_duration_seconds": 0.0,
        }
    else:
        sequence_paths, branding_report = build_branded_sequence(
            scene_paths=scene_paths,
            render_dir=render_dir,
            profile=profile,
        )

    concat_final(
        scene_paths=sequence_paths,
        output_path=final_path,
        render_dir=render_dir,
    )

    news_duration = sum(
        float(scene["actual_duration_seconds"])
        for scene in manifest["scenes"]
    )
    branding_duration = float(
        branding_report["extra_duration_seconds"]
    )

    expected_duration = round(
        news_duration + branding_duration,
        3,
    )

    report = {
        "run_id": run_id,
        "profile": profile.name,
        "canvas": {
            "width": profile.width,
            "height": profile.height,
            "fps": profile.fps,
        },
        "expected_duration_seconds": expected_duration,
        "news_duration_seconds": round(news_duration, 6),
        "branding_duration_seconds": round(branding_duration, 6),
        "branding": branding_report,
        "final_video": str(final_path),
        "final_probe": probe(final_path),
        "scene_files": [str(path) for path in scene_paths],
        "sequence_files": [str(path) for path in sequence_paths],
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