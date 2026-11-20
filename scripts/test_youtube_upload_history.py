from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from scripts import upload_youtube_video as uploader


class YouTubeUploadHistoryTests(unittest.TestCase):
    def connection(self, rows):
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = rows
        return conn, cursor

    def test_previous_upload_is_found_behind_new_render(self):
        conn, cursor = self.connection([
            {"id": 202, "meta": {"stage": "rendered"}},
            {
                "id": 201,
                "meta": {
                    "youtube_video_id": "existing-test-id",
                    "youtube_privacy_status": "private",
                },
            },
        ])

        result = uploader.inspect_upload_history(
            conn,
            run_id=578,
            profile="news_vertical",
        )

        self.assertEqual(result, "existing-test-id")
        self.assertEqual(
            cursor.execute.call_args.args[1],
            (578, "news_vertical"),
        )

    def test_upload_result_checks_rowcount_after_execute(self):
        conn, cursor = self.connection([])
        cursor.rowcount = -1

        def execute_update(*args, **kwargs):
            cursor.rowcount = 1

        cursor.execute.side_effect = execute_update

        uploader.save_upload_result(
            conn,
            pipeline_run_id=201,
            patch={
                "youtube_video_id": "saved-test-id",
                "youtube_upload_state": "uploaded",
                "youtube_privacy_status": "private",
            },
        )

        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()

        sql, parameters = cursor.execute.call_args.args
        normalized_sql = " ".join(sql.split())

        self.assertIn(
            "meta->>'youtube_upload_state' = 'started'",
            normalized_sql,
        )
        self.assertEqual(parameters[1], 201)

    def test_upload_result_without_updated_row_is_not_committed(self):
        conn, cursor = self.connection([])
        cursor.rowcount = -1

        def execute_update(*args, **kwargs):
            cursor.rowcount = 0

        cursor.execute.side_effect = execute_update

        with self.assertRaisesRegex(
            RuntimeError,
            "manual reconciliation required",
        ):
            uploader.save_upload_result(
                conn,
                pipeline_run_id=201,
                patch={
                    "youtube_video_id": "saved-test-id",
                    "youtube_upload_state": "uploaded",
                },
            )

        cursor.execute.assert_called_once()
        conn.commit.assert_not_called()

    def test_unresolved_attempt_blocks_retry(self):
        conn, _ = self.connection([
            {
                "id": 201,
                "meta": {"youtube_upload_state": "started"},
            },
        ])

        with self.assertRaisesRegex(
            RuntimeError,
            "manual reconciliation required",
        ):
            uploader.inspect_upload_history(
                conn,
                run_id=578,
                profile="news_vertical",
            )

    def test_conflicting_video_ids_are_rejected(self):
        conn, _ = self.connection([
            {
                "id": 202,
                "meta": {
                    "youtube_video_id": "test-a",
                    "youtube_privacy_status": "private",
                },
            },
            {
                "id": 201,
                "meta": {
                    "youtube_video_id": "test-b",
                    "youtube_privacy_status": "private",
                },
            },
        ])

        with self.assertRaisesRegex(
            RuntimeError,
            "Multiple YouTube IDs",
        ):
            uploader.inspect_upload_history(
                conn,
                run_id=578,
                profile="news_vertical",
            )

    def test_busy_lock_is_rejected(self):
        conn, cursor = self.connection([])
        cursor.fetchone.return_value = (False,)

        with self.assertRaisesRegex(RuntimeError, "lock is busy"):
            uploader.acquire_upload_lock(
                conn,
                run_id=578,
                profile="news_vertical",
            )

        conn.commit.assert_not_called()

    def test_started_marker_must_update_one_record(self):
        conn, cursor = self.connection([])
        cursor.rowcount = 0

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot claim rendered video",
        ):
            uploader.mark_upload_started(
                conn,
                pipeline_run_id=202,
                run_id=578,
                profile="news_vertical",
                privacy_status="private",
            )

        conn.commit.assert_not_called()

    def test_vertical_upload_requires_recorded_private_status(self):
        for privacy in (None, "public", "unlisted"):
            with self.subTest(privacy=privacy):
                meta = {
                    "youtube_video_id": "existing-test-id",
                }
                if privacy is not None:
                    meta["youtube_privacy_status"] = privacy

                conn, _ = self.connection([
                    {"id": 201, "meta": meta},
                ])

                with self.assertRaisesRegex(
                    RuntimeError,
                    "not recorded as private",
                ):
                    uploader.inspect_upload_history(
                        conn,
                        run_id=578,
                        profile="news_vertical",
                    )

if __name__ == "__main__":
    unittest.main()