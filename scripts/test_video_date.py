from __future__ import annotations

import unittest

from datetime import datetime, timedelta, timezone

from scripts.render_news_video_v2 import build_subtitle_style
from scripts.video_date import (
    build_date_style,
    format_issue_date,
    gmt_date_from_timestamp,
    validate_issue_date,
)
from scripts.video_profiles import get_video_profile
import tempfile

from pathlib import Path
from unittest.mock import patch

from scripts.render_news_video_v2 import (
    concat_final,
    write_date_subtitles,
)

class VideoDateTests(unittest.TestCase):
    def test_positive_offset_can_move_date_to_previous_day(self):
        timestamp = datetime(
            2026, 11, 22, 1, 30,
            tzinfo=timezone(timedelta(hours=3)),
        )

        self.assertEqual(
            gmt_date_from_timestamp(timestamp),
            "2026-11-21",
        )

    def test_negative_offset_can_move_date_to_next_day(self):
        timestamp = datetime(
            2026, 11, 21, 23, 30,
            tzinfo=timezone(timedelta(hours=-5)),
        )

        self.assertEqual(
            gmt_date_from_timestamp(timestamp),
            "2026-11-22",
        )

    def test_naive_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            gmt_date_from_timestamp(datetime(2026, 11, 21, 12))

    def test_english_display_format(self):
        self.assertEqual(
            format_issue_date("2026-11-21"),
            "November 21, 2026 \u00b7 GMT",
        )
        self.assertEqual(
            format_issue_date("2026-01-02"),
            "January 2, 2026 \u00b7 GMT",
        )

    def test_invalid_or_noncanonical_dates_are_rejected(self):
        for value in (
            "",
            "20261121",
            "21/11/2026",
            "2026-02-30",
            "2026-11-21T12:00:00Z",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_issue_date(value)

    def test_date_style_changes_only_alignment(self):
        for name in ("news_horizontal", "news_vertical"):
            with self.subTest(profile=name):
                subtitle_style = build_subtitle_style(
                    get_video_profile(name)
                )
                date_style = build_date_style(subtitle_style)

                self.assertIn("Alignment=8", date_style)
                self.assertEqual(
                    date_style.replace(
                        "Alignment=8", "Alignment=2"
                    ),
                    subtitle_style,
                )

    def test_missing_or_duplicate_alignment_is_rejected(self):
        for style in (
            "FontName=DejaVu Sans,FontSize=14",
            "Alignment=2,Alignment=8",
        ):
            with self.subTest(style=style):
                with self.assertRaises(ValueError):
                    build_date_style(style)

    def test_date_subtitles_cover_full_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "issue_date.srt"

            write_date_subtitles(
                issue_date="2026-11-21",
                duration_seconds=125.4601,
                output_path=path,
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                (
                    "1\n"
                    "00:00:00,000 --> 00:02:05,461\n"
                    "November 21, 2026 \u00b7 GMT\n"
                ),
            )

    def test_date_subtitles_reject_invalid_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            for duration in (
                0, -1, float("inf"), float("nan"), True,
            ):
                with self.subTest(duration=duration):
                    with self.assertRaises(ValueError):
                        write_date_subtitles(
                            issue_date="2026-11-21",
                            duration_seconds=duration,
                            output_path=(
                                Path(temporary) / "issue_date.srt"
                            ),
                        )

    def test_final_concat_applies_profile_date_style(self):
        for name in ("news_horizontal", "news_vertical"):
            with self.subTest(profile=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    profile = get_video_profile(name)

                    with patch(
                        "scripts.render_news_video_v2.run"
                    ) as run:
                        concat_final(
                            scene_paths=[root / "scene.mp4"],
                            output_path=root / "final.mp4",
                            render_dir=root,
                            profile=profile,
                            issue_date="2026-11-21",
                            duration_seconds=12.5,
                        )

                    command = run.call_args.args[0]
                    self.assertEqual(
                        command[command.index("-fflags") + 1],
                        "+genpts",
                    )
                    video_filter = command[
                        command.index("-vf") + 1
                    ]

                    expected_style = build_date_style(
                        build_subtitle_style(profile)
                    )

                    self.assertIn(expected_style, video_filter)
                    self.assertTrue(
                        (root / "issue_date.srt").is_file()
                    )

    def test_legacy_concat_has_no_date_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            with patch(
                "scripts.render_news_video_v2.run"
            ) as run:
                concat_final(
                    scene_paths=[root / "scene.mp4"],
                    output_path=root / "final.mp4",
                    render_dir=root,
                )

            command = run.call_args.args[0]
            self.assertNotIn("-vf", command)
            self.assertFalse(
                (root / "issue_date.srt").exists()
            )

if __name__ == "__main__":
    unittest.main()