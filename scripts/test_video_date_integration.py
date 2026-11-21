from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import run_video_pipeline as pipeline

from scripts import build_render_manifest as manifest_builder
from scripts import video_fingerprint as fingerprint
from scripts.video_profiles import get_video_profile
from datetime import datetime, timezone

from unittest.mock import MagicMock

class VideoDateIntegrationTests(unittest.TestCase):
    def manifest_argv(self):
        return [
            "build_render_manifest",
            "--run-id", "580",
            "--audio-manifest", "audio_manifest.json",
            "--assets-manifest", "manifest.csv",
            "--output-dir", "render_assets",
        ]

    def test_clustering_run_query_explicitly_interprets_utc(self):
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value

        cursor.fetchone.return_value = {
            "id": 580,
            "status": "success",
            "finished_at": datetime(
                2026, 11, 21, 6, 0, 15,
                tzinfo=timezone.utc,
            ),
        }

        result = pipeline.validate_clustering_run(conn, 580)

        self.assertEqual(result, "2026-11-21")

        query, parameters = cursor.execute.call_args.args
        normalized_query = " ".join(query.split())

        self.assertIn(
            "finished_at AT TIME ZONE 'UTC' AS finished_at",
            normalized_query,
        )
        self.assertEqual(parameters, (580,))

    def test_fingerprint_rejects_invalid_date_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            for invalid_date in ("20261121", "2026-02-30", ""):
                with self.subTest(issue_date=invalid_date):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Issue date",
                    ):
                        fingerprint.build_news_fingerprint(
                            run_id=580,
                            profile=get_video_profile(
                                "news_vertical"
                            ),
                            project_root=root,
                            scene_manifest_path=root / "script.json",
                            audio_manifest_path=root / "audio.json",
                            assets_manifest_path=root / "assets.csv",
                            media_dir=root / "media",
                            max_assets_per_scene=3,
                            issue_date_gmt=invalid_date,
                        )

    def test_render_report_requires_matching_date_and_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "render_report_v2.json"

            cases = [
                ({}, False),
                (
                    {
                        "issue_date_gmt": "2026-11-20",
                        "issue_date_text":
                            "November 20, 2026 \u00b7 GMT",
                    },
                    False,
                ),
                (
                    {
                        "issue_date_gmt": "2026-11-21",
                        "issue_date_text": "wrong text",
                    },
                    False,
                ),
                (
                    {
                        "issue_date_gmt": "2026-11-21",
                        "issue_date_text":
                            "November 21, 2026 \u00b7 GMT",
                    },
                    True,
                ),
            ]

            for report, expected in cases:
                with self.subTest(report=report):
                    report_path.write_text(
                        json.dumps(report),
                        encoding="utf-8",
                    )

                    self.assertEqual(
                        pipeline.render_report_has_issue_date(
                            render_dir=root,
                            issue_date_gmt="2026-11-21",
                        ),
                        expected,
                    )
                    
    def test_manifest_accepts_issue_date(self):
        argv = self.manifest_argv() + [
            "--issue-date-gmt", "2026-11-21",
        ]

        with patch("sys.argv", argv):
            args = manifest_builder.parse_args()

        self.assertEqual(args.issue_date_gmt, "2026-11-21")

    def test_manifest_legacy_arguments_remain_supported(self):
        with patch("sys.argv", self.manifest_argv()):
            args = manifest_builder.parse_args()

        self.assertIsNone(args.issue_date_gmt)

    def test_fingerprint_tracks_issue_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio_manifest.json"
            script = root / "scene_manifest.json"
            assets = root / "manifest.csv"

            audio.write_text(
                json.dumps({"scenes": []}),
                encoding="utf-8",
            )
            script.write_text("{}", encoding="utf-8")
            assets.write_text(
                "scene_number,status,asset_type,asset_path\n",
                encoding="utf-8",
            )

            kwargs = {
                "run_id": 580,
                "profile": get_video_profile("news_vertical"),
                "project_root": root,
                "scene_manifest_path": script,
                "audio_manifest_path": audio,
                "assets_manifest_path": assets,
                "media_dir": root / "media",
                "max_assets_per_scene": 3,
            }

            with (
                patch.object(
                    fingerprint.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        stdout="ffmpeg test version",
                    ),
                ),
                patch.object(
                    fingerprint,
                    "version",
                    return_value="test",
                ),
            ):
                first = fingerprint.build_news_fingerprint(
                    **kwargs,
                    issue_date_gmt="2026-11-21",
                )
                same = fingerprint.build_news_fingerprint(
                    **kwargs,
                    issue_date_gmt="2026-11-21",
                )
                changed = fingerprint.build_news_fingerprint(
                    **kwargs,
                    issue_date_gmt="2026-11-22",
                )
                undated = fingerprint.build_news_fingerprint(
                    **kwargs,
                )

            self.assertEqual(first, same)
            self.assertNotEqual(first["sha256"], changed["sha256"])
            self.assertNotEqual(first["sha256"], undated["sha256"])
            self.assertIn(
                "settings.issue_date_gmt",
                fingerprint.changed_components(first, changed),
            )
            self.assertIn(
                "code.video_date.py",
                first["payload"]["files"],
            )


if __name__ == "__main__":
    unittest.main()