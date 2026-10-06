from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


WIDTH = 2400
HEIGHT = 1350
FPS = 30
SAMPLE_RATE = 44100

MEDIA_DIR = Path("/app/media")
TRANSITION_SECONDS = 1.0

CRF = "20"
PRESET = "medium"


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def media_duration(path: Path) -> float:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Media missing or empty: {path}")

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0.0)

    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid media duration: {path}")

    return duration


def encoding_options() -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", CRF,
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-video_track_timescale", "90000",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", str(SAMPLE_RATE),
        "-ac", "2",
        "-movflags", "+faststart",
    ]


def canvas_filter() -> str:
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"fps={FPS},"
        "format=yuv420p,"
        "setsar=1"
    )


def render_brand_clip(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> float:
    video_seconds = media_duration(video_path)
    audio_seconds = media_duration(audio_path)

    frame_count = max(1, round(audio_seconds * FPS))
    duration = frame_count / FPS

    # An extra second provides padding beyond the requested final frame.
    padding = max(0.0, duration - video_seconds) + 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filters = (
        f"[0:v:0]setpts=PTS-STARTPTS,{canvas_filter()},"
        f"tpad=stop_mode=clone:stop_duration={padding:.6f},"
        f"trim=duration={duration:.6f},"
        f"setpts=N/({FPS}*TB)[v];"
        f"[1:a:0]aresample={SAMPLE_RATE},"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS,"
        f"apad,atrim=duration={duration:.6f}[a]"
    )

    run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", filters,
            "-map", "[v]",
            "-map", "[a]",
            "-t", f"{duration:.6f}",
            *encoding_options(),
            str(output_path),
        ]
    )

    return duration


def extract_boundary_frame(
    *,
    video_path: Path,
    output_path: Path,
    last: bool,
) -> None:
    duration = media_duration(video_path)

    command = ["ffmpeg", "-y"]

    if last:
        timestamp = max(0.0, duration - 1.0 / FPS)
        command.extend(["-ss", f"{timestamp:.6f}"])

    command.extend(
        [
            "-i", str(video_path),
            "-map", "0:v:0",
            "-frames:v", "1",
            "-q:v", "2",
            "-update", "1",
            str(output_path),
        ]
    )

    run(command)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Boundary frame was not created: {output_path}")


def tempo_filter(speed: float) -> str:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Audio tempo must be finite and positive")

    stages: list[float] = []

    while speed > 2.0:
        stages.append(2.0)
        speed /= 2.0

    while speed < 0.5:
        stages.append(0.5)
        speed /= 0.5

    stages.append(speed)
    return ",".join(f"atempo={stage:.9f}" for stage in stages)


def render_transition(
    *,
    previous_scene: Path,
    next_scene: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
) -> float:
    audio_seconds = media_duration(audio_path)
    duration = TRANSITION_SECONDS

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous_frame = work_dir / "previous_last.jpg"
    next_frame = work_dir / "next_first.jpg"

    extract_boundary_frame(
        video_path=previous_scene,
        output_path=previous_frame,
        last=True,
    )
    extract_boundary_frame(
        video_path=next_scene,
        output_path=next_frame,
        last=False,
    )

    # Fade the two held frames, without overlapping news narration.
    fade_span = (round(duration * FPS) - 1) / FPS
    audio_tempo = tempo_filter(audio_seconds / duration)

    filters = (
        f"[0:v:0]{canvas_filter()},"
        f"settb=AVTB,setpts=N/({FPS}*TB)[previous];"
        f"[1:v:0]{canvas_filter()},"
        f"settb=AVTB,setpts=N/({FPS}*TB)[next];"
        "[previous][next]"
        f"blend=all_expr='A*(1-min(T/{fade_span:.9f},1))"
        f"+B*min(T/{fade_span:.9f},1)',"
        f"trim=duration={duration:.6f},"
        f"setpts=N/({FPS}*TB)[v];"
        f"[2:a:0]{audio_tempo},"
        f"aresample={SAMPLE_RATE},"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS,"
        f"apad,atrim=duration={duration:.6f},"
        "afade=t=in:st=0:d=0.02,"
        "afade=t=out:st=0.95:d=0.05[a]"
    )

    run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", str(FPS),
            "-i", str(previous_frame),
            "-loop", "1",
            "-framerate", str(FPS),
            "-i", str(next_frame),
            "-i", str(audio_path),
            "-filter_complex", filters,
            "-map", "[v]",
            "-map", "[a]",
            "-t", f"{duration:.6f}",
            *encoding_options(),
            str(output_path),
        ]
    )

    return duration


def build_branded_sequence(
    *,
    scene_paths: list[Path],
    render_dir: Path,
    media_dir: Path = MEDIA_DIR,
) -> tuple[list[Path], dict[str, object]]:
    if not scene_paths:
        raise ValueError("Cannot brand an empty scene sequence")

    video_path = media_dir / "video.mp4"
    intro_audio = media_dir / "news_intro.wav"
    ending_audio = media_dir / "news_ending.wav"
    transition_audio = media_dir / "news_transition.wav"

    # Fail before rendering if any required input is missing or invalid.
    for path in (
        video_path,
        intro_audio,
        ending_audio,
        transition_audio,
    ):
        media_duration(path)

    branding_dir = render_dir / "branding"
    branding_dir.mkdir(parents=True, exist_ok=True)

    intro_path = branding_dir / "intro.mp4"
    ending_path = branding_dir / "ending.mp4"

    intro_seconds = render_brand_clip(
        video_path=video_path,
        audio_path=intro_audio,
        output_path=intro_path,
    )
    ending_seconds = render_brand_clip(
        video_path=video_path,
        audio_path=ending_audio,
        output_path=ending_path,
    )

    sequence = [intro_path]
    transition_paths: list[Path] = []

    for index, scene_path in enumerate(scene_paths):
        sequence.append(scene_path)

        if index + 1 < len(scene_paths):
            transition_number = index + 1
            transition_path = (
                branding_dir / f"transition_{transition_number:03d}.mp4"
            )

            render_transition(
                previous_scene=scene_path,
                next_scene=scene_paths[index + 1],
                audio_path=transition_audio,
                output_path=transition_path,
                work_dir=(
                    branding_dir
                    / f"transition_{transition_number:03d}_frames"
                ),
            )

            sequence.append(transition_path)
            transition_paths.append(transition_path)

    sequence.append(ending_path)

    extra_seconds = (
        intro_seconds
        + ending_seconds
        + len(transition_paths) * TRANSITION_SECONDS
    )

    report: dict[str, object] = {
        "enabled": True,
        "intro_duration_seconds": intro_seconds,
        "ending_duration_seconds": ending_seconds,
        "transition_duration_seconds": TRANSITION_SECONDS,
        "transition_count": len(transition_paths),
        "extra_duration_seconds": round(extra_seconds, 6),
        "intro_file": str(intro_path),
        "ending_file": str(ending_path),
        "transition_files": [str(path) for path in transition_paths],
        "sequence_files": [str(path) for path in sequence],
    }

    return sequence, report