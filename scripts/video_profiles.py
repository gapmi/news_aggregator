from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoProfile:
    name: str
    width: int
    height: int
    fps: int


DEFAULT_VIDEO_PROFILE = "news_horizontal"


VIDEO_PROFILES: dict[str, VideoProfile] = {
    "news_horizontal": VideoProfile(
        name="news_horizontal",
        width=2400,
        height=1350,
        fps=30,
    ),
    "news_vertical": VideoProfile(
        name="news_vertical",
        width=1080,
        height=1920,
        fps=30,
    ),
}


def get_video_profile(name: str) -> VideoProfile:
    try:
        return VIDEO_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(VIDEO_PROFILES))
        raise ValueError(
            f"Unknown video profile: {name!r}. "
            f"Available profiles: {available}"
        ) from exc