from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from scripts import video_worker as worker


class VideoWorkerUploadProfileTests(unittest.TestCase):
    def test_uploaded_horizontal_is_not_rebuilt_or_uploaded(self):
        conn = MagicMock()

        with (
            patch.object(
                worker,
                "inspect_upload_history",
                side_effect=["horizontal-test-id", None],
            ),
            patch.object(worker, "run_video_pipeline") as build,
            patch.object(worker, "run_upload") as upload,
        ):
            worker.process_run_profiles(conn, run_id=578)

        build.assert_called_once_with(
            578,
            profiles=("news_vertical",),
        )
        upload.assert_called_once_with(
            578,
            profile="news_vertical",
        )

    def test_new_run_builds_and_uploads_in_profile_order(self):
        events = MagicMock()

        with (
            patch.object(
                worker,
                "inspect_upload_history",
                return_value=None,
            ),
            patch.object(worker, "run_video_pipeline") as build,
            patch.object(worker, "run_upload") as upload,
        ):
            events.attach_mock(build, "build")
            events.attach_mock(upload, "upload")

            worker.process_run_profiles(MagicMock(), run_id=578)

        self.assertEqual(events.mock_calls, [
            call.build(578, profiles=("news_horizontal",)),
            call.upload(578, profile="news_horizontal"),
            call.build(578, profiles=("news_vertical",)),
            call.upload(578, profile="news_vertical"),
        ])

    def test_vertical_upload_is_always_private(self):
        with (
            patch.object(worker, "UPLOAD_ENABLED", True),
            patch.object(worker, "PRIVACY_STATUS", "public"),
            patch.object(worker, "run_command") as command,
        ):
            worker.run_upload(578, profile="news_vertical")

        args = command.call_args.args[0]
        self.assertEqual(
            args[args.index("--privacy-status") + 1],
            "private",
        )
        self.assertEqual(
            args[args.index("--profile") + 1],
            "news_vertical",
        )
        self.assertEqual(
            args[args.index("--video-path") + 1],
            str(
                worker.PROJECT_ROOT
                / "artifacts"
                / "run_578"
                / "vertical"
                / "render"
                / "final_578.mp4"
            ),
        )
        self.assertNotIn("--dry-run", args)

    def test_disabled_uploads_are_dry_run_for_both_profiles(self):
        with (
            patch.object(worker, "UPLOAD_ENABLED", False),
            patch.object(worker, "run_command") as command,
        ):
            for profile in ("news_horizontal", "news_vertical"):
                worker.run_upload(578, profile=profile)

        self.assertEqual(command.call_count, 2)
        for invocation in command.call_args_list:
            self.assertIn("--dry-run", invocation.args[0])

    def test_horizontal_upload_failure_stops_vertical_processing(self):
        error = subprocess.CalledProcessError(1, ["upload-test"])

        with (
            patch.object(
                worker,
                "inspect_upload_history",
                return_value=None,
            ),
            patch.object(worker, "run_video_pipeline") as build,
            patch.object(
                worker,
                "run_upload",
                side_effect=error,
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                worker.process_run_profiles(
                    MagicMock(),
                    run_id=578,
                )

        build.assert_called_once_with(
            578,
            profiles=("news_horizontal",),
        )

    def test_unresolved_upload_stops_before_build(self):
        with (
            patch.object(
                worker,
                "inspect_upload_history",
                side_effect=RuntimeError("manual reconciliation required"),
            ),
            patch.object(worker, "run_video_pipeline") as build,
            patch.object(worker, "run_upload") as upload,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "manual reconciliation",
            ):
                worker.process_run_profiles(
                    MagicMock(),
                    run_id=578,
                )

        build.assert_not_called()
        upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()