from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from scripts import upload_youtube_video as uploader
from scripts import video_worker as worker


class VideoReleaseSafetyTests(unittest.TestCase):
    def test_upload_query_selects_requested_profile(self) -> None:
        for profile in ("news_horizontal", "news_vertical"):
            with self.subTest(profile=profile):
                conn = MagicMock()
                cursor = (
                    conn.cursor.return_value
                    .__enter__.return_value
                )
                cursor.fetchone.return_value = {
                    "id": 123,
                    "meta": {
                        "profile": profile,
                        "stage": "rendered",
                    },
                }

                record = uploader.latest_video_pipeline_run(
                    conn,
                    run_id=578,
                    profile=profile,
                )

                sql, parameters = cursor.execute.call_args.args
                normalized_sql = " ".join(sql.split())

                self.assertEqual(parameters, (578, profile))
                self.assertIn(
                    "COALESCE( meta->>'profile', "
                    "'news_horizontal' ) = %s",
                    normalized_sql,
                )
                self.assertIn(
                    "meta->>'stage' IN "
                    "( 'rendered', 'youtube_uploaded' )",
                    normalized_sql,
                )
                self.assertEqual(record["id"], 123)

    def test_vertical_nonprivate_rejected_before_external_access(
        self,
    ) -> None:
        vertical_path = (
            uploader.PROJECT_ROOT
            / "artifacts"
            / "run_578"
            / "vertical"
            / "render"
            / "final_578.mp4"
        )

        for privacy in ("public", "unlisted"):
            with self.subTest(privacy=privacy):
                argv = [
                    "upload_youtube_video",
                    "--run-id", "578",
                    "--profile", "news_vertical",
                    "--video-path", str(vertical_path),
                    "--privacy-status", privacy,
                    "--dry-run",
                ]

                with (
                    patch.object(sys, "argv", argv),
                    patch.object(uploader, "get_conn") as db,
                    patch.object(
                        uploader, "load_credentials"
                    ) as oauth,
                    patch.object(
                        uploader, "build_youtube_client"
                    ) as client,
                    patch.object(
                        uploader, "resumable_upload"
                    ) as upload,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Vertical uploads require",
                    ):
                        uploader.main()

                db.assert_not_called()
                oauth.assert_not_called()
                client.assert_not_called()
                upload.assert_not_called()

    def test_vertical_private_target_is_allowed(self) -> None:
        path = (
            uploader.PROJECT_ROOT
            / "artifacts"
            / "run_578"
            / "vertical"
            / "render"
            / "final_578.mp4"
        )

        result = uploader.validate_upload_target(
            run_id=578,
            profile="news_vertical",
            video_path=path,
            privacy_status="private",
        )

        self.assertEqual(result, path)

    def test_profiles_cannot_use_each_others_paths(self) -> None:
        root = (
            uploader.PROJECT_ROOT
            / "artifacts"
            / "run_578"
        )
        horizontal = root / "render" / "final_578.mp4"
        vertical = (
            root / "vertical" / "render" / "final_578.mp4"
        )

        for profile, path in (
            ("news_horizontal", vertical),
            ("news_vertical", horizontal),
        ):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Video path does not match profile",
                ):
                    uploader.validate_upload_target(
                        run_id=578,
                        profile=profile,
                        video_path=path,
                        privacy_status="private",
                    )

    def test_worker_cleanup_is_dry_run(self) -> None:
        with patch.object(worker, "run_command") as command:
            worker.run_cleanup()

        command.assert_called_once_with([
            sys.executable,
            "-m",
            "scripts.cleanup_video_artifacts",
            "--retention-hours",
            str(worker.RETENTION_HOURS),
            "--dry-run",
        ])


if __name__ == "__main__":
    unittest.main()