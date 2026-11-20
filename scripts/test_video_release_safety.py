from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from scripts import upload_youtube_video as uploader
from scripts import video_worker as worker


class VideoReleaseSafetyTests(unittest.TestCase):
    def test_upload_query_selects_horizontal_render(self) -> None:
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value

        cursor.fetchone.return_value = {
            "id": 123,
            "meta": {
                "profile": "news_horizontal",
                "stage": "rendered",
            },
        }

        record = uploader.latest_video_pipeline_run(
            conn,
            run_id=570,
        )

        cursor.execute.assert_called_once()
        sql, parameters = cursor.execute.call_args.args
        normalized_sql = " ".join(sql.split())

        self.assertEqual(parameters, (570,))
        self.assertIn(
            "COALESCE( meta->>'profile', "
            "'news_horizontal' ) = 'news_horizontal'",
            normalized_sql,
        )
        self.assertIn(
            "meta->>'stage' IN ( 'rendered', 'youtube_uploaded' )",
            normalized_sql,
        )
        self.assertEqual(record["id"], 123)

    def test_vertical_upload_is_rejected_before_database_or_oauth(
        self,
    ) -> None:
        vertical_path = (
            uploader.PROJECT_ROOT
            / "artifacts"
            / "run_570"
            / "vertical"
            / "render"
            / "final_570.mp4"
        )

        argv = [
            "upload_youtube_video",
            "--run-id",
            "570",
            "--video-path",
            str(vertical_path),
            "--dry-run",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch.object(uploader, "get_conn") as get_conn,
            patch.object(
                uploader,
                "load_credentials",
            ) as load_credentials,
                        patch.object(
                uploader,
                "build_youtube_client",
            ) as build_youtube,
            patch.object(
                uploader,
                "resumable_upload",
            ) as upload,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "only supports the horizontal video",
            ):
                uploader.main()

        get_conn.assert_not_called()
        load_credentials.assert_not_called()
        build_youtube.assert_not_called()
        upload.assert_not_called()

    def test_worker_cleanup_is_dry_run(self) -> None:
        with patch.object(worker, "run_command") as run_command:
            worker.run_cleanup()

        run_command.assert_called_once_with([
            sys.executable,
            "-m",
            "scripts.cleanup_video_artifacts",
            "--retention-hours",
            str(worker.RETENTION_HOURS),
            "--dry-run",
        ])


if __name__ == "__main__":
    unittest.main()