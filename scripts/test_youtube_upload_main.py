from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts import upload_youtube_video as uploader


class YouTubeUploadMainTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.video_path = (
            self.root
            / "artifacts"
            / "run_578"
            / "vertical"
            / "render"
            / "final_578.mp4"
        )
        self.video_path.parent.mkdir(parents=True)
        with self.video_path.open("wb") as file:
            file.truncate(1_000_001)

        self.args = SimpleNamespace(
            run_id=578,
            profile="news_vertical",
            video_path=self.video_path,
            privacy_status="private",
            made_for_kids=False,
            dry_run=False,
        )

        self.conn = MagicMock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        self.stack.enter_context(
            patch.object(uploader, "PROJECT_ROOT", self.root)
        )
        self.stack.enter_context(
            patch.object(uploader, "parse_args", return_value=self.args)
        )
        self.stack.enter_context(
            patch.object(uploader, "get_conn", return_value=self.conn)
        )

        self.mocks = {}
        for name in (
            "acquire_upload_lock",
            "inspect_upload_history",
            "latest_video_pipeline_run",
            "load_video_metadata",
            "load_credentials",
            "build_youtube_client",
            "mark_upload_started",
            "resumable_upload",
            "save_upload_result",
        ):
            self.mocks[name] = self.stack.enter_context(
                patch.object(uploader, name)
            )

        self.mocks["inspect_upload_history"].return_value = None
        self.mocks["latest_video_pipeline_run"].return_value = {
            "id": 1001,
            "meta": {
                "profile": "news_vertical",
                "stage": "rendered",
                "final_video_path": str(self.video_path),
            },
        }
        self.mocks["load_video_metadata"].return_value = {
            "title": "Test title",
            "description": "Test description",
        }
        self.mocks["resumable_upload"].return_value = {
            "id": "vertical-test-id",
        }

        self.events = MagicMock()
        for name in (
            "mark_upload_started",
            "resumable_upload",
            "save_upload_result",
        ):
            self.events.attach_mock(self.mocks[name], name)

    def run_main(self):
        output = StringIO()
        with redirect_stdout(output):
            result = uploader.main()
        return result, json.loads(output.getvalue())

    def test_private_dry_run_does_not_start_or_send_upload(self):
        self.args.dry_run = True

        result, output = self.run_main()

        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "dry_run")
        self.assertEqual(output["profile"], "news_vertical")
        self.assertEqual(output["privacy_status"], "private")

        for name in (
            "load_credentials",
            "build_youtube_client",
            "mark_upload_started",
            "resumable_upload",
            "save_upload_result",
        ):
            self.mocks[name].assert_not_called()

        self.conn.close.assert_called_once()

    def test_existing_upload_skips_oauth_and_upload(self):
        self.mocks["inspect_upload_history"].return_value = (
            "existing-test-id"
        )

        result, output = self.run_main()

        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "already_uploaded")
        self.assertEqual(
            output["youtube_video_id"],
            "existing-test-id",
        )

        for name in (
            "latest_video_pipeline_run",
            "load_video_metadata",
            "load_credentials",
            "build_youtube_client",
            "mark_upload_started",
            "resumable_upload",
            "save_upload_result",
        ):
            self.mocks[name].assert_not_called()

        self.conn.close.assert_called_once()

    def test_success_marks_started_before_send_and_saves_result(self):
        result, output = self.run_main()

        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "uploaded")
        self.assertEqual(output["privacy_status"], "private")

        self.assertEqual(
            [event[0] for event in self.events.mock_calls],
            [
                "mark_upload_started",
                "resumable_upload",
                "save_upload_result",
            ],
        )

        self.mocks["mark_upload_started"].assert_called_once_with(
            self.conn,
            pipeline_run_id=1001,
            run_id=578,
            profile="news_vertical",
            privacy_status="private",
        )

        upload_kwargs = (
            self.mocks["resumable_upload"].call_args.kwargs
        )
        self.assertEqual(upload_kwargs["privacy_status"], "private")
        self.assertEqual(
            upload_kwargs["video_path"],
            self.video_path,
        )

        saved_kwargs = (
            self.mocks["save_upload_result"].call_args.kwargs
        )
        self.assertEqual(saved_kwargs["pipeline_run_id"], 1001)
        saved_patch = saved_kwargs["patch"]
        self.assertEqual(
            saved_patch["youtube_video_id"],
            "vertical-test-id",
        )
        self.assertEqual(
            saved_patch["youtube_upload_state"],
            "uploaded",
        )
        self.assertEqual(
            saved_patch["youtube_privacy_status"],
            "private",
        )
        self.conn.close.assert_called_once()

    def test_send_failure_does_not_save_success_or_retry(self):
        self.mocks["resumable_upload"].side_effect = RuntimeError(
            "simulated connection failure"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "simulated connection failure",
        ):
            self.run_main()

        self.mocks["mark_upload_started"].assert_called_once()
        self.mocks["resumable_upload"].assert_called_once()
        self.mocks["save_upload_result"].assert_not_called()
        self.conn.close.assert_called_once()

    def test_save_failure_does_not_send_video_again(self):
        self.mocks["save_upload_result"].side_effect = RuntimeError(
            "simulated database failure"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "simulated database failure",
        ):
            self.run_main()

        self.mocks["mark_upload_started"].assert_called_once()
        self.mocks["resumable_upload"].assert_called_once()
        self.mocks["save_upload_result"].assert_called_once()
        self.conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()