from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import upload_youtube_video as uploader


class FakeHttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.resp = SimpleNamespace(status=status)


class YouTubeUploadChunkTests(unittest.TestCase):
    def test_more_than_five_successful_chunks(self) -> None:
        request = MagicMock()
        progress = MagicMock()
        progress.progress.return_value = 0.5

        request.next_chunk.side_effect = [
            *[(progress, None) for _ in range(7)],
            (None, {"id": "vertical-test"}),
        ]

        response = uploader.upload_chunks(
            request,
            http_error_type=FakeHttpError,
        )

        self.assertEqual(response, {"id": "vertical-test"})
        self.assertEqual(request.next_chunk.call_count, 8)

    def test_successful_chunk_resets_error_counter(self) -> None:
        request = MagicMock()
        progress = MagicMock()
        progress.progress.return_value = 0.5

        request.next_chunk.side_effect = [
            FakeHttpError(503),
            (progress, None),
            FakeHttpError(503),
            (None, {"id": "vertical-test"}),
        ]

        with patch.object(uploader.time, "sleep") as sleep:
            response = uploader.upload_chunks(
                request,
                http_error_type=FakeHttpError,
                max_consecutive_errors=2,
            )

        self.assertEqual(response["id"], "vertical-test")
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [2, 2],
        )

    def test_nonretryable_error_is_not_retried(self) -> None:
        request = MagicMock()
        request.next_chunk.side_effect = FakeHttpError(403)

        with patch.object(uploader.time, "sleep") as sleep:
            with self.assertRaises(FakeHttpError):
                uploader.upload_chunks(
                    request,
                    http_error_type=FakeHttpError,
                )

        request.next_chunk.assert_called_once()
        sleep.assert_not_called()

    def test_consecutive_errors_stop_at_limit(self) -> None:
        request = MagicMock()
        request.next_chunk.side_effect = FakeHttpError(503)

        with patch.object(uploader.time, "sleep") as sleep:
            with self.assertRaises(FakeHttpError):
                uploader.upload_chunks(
                    request,
                    http_error_type=FakeHttpError,
                    max_consecutive_errors=3,
                )

        self.assertEqual(request.next_chunk.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_missing_progress_and_response_is_rejected(self) -> None:
        request = MagicMock()
        request.next_chunk.return_value = (None, None)

        with self.assertRaisesRegex(
            RuntimeError,
            "neither progress nor response",
        ):
            uploader.upload_chunks(
                request,
                http_error_type=FakeHttpError,
            )

        request.next_chunk.assert_called_once()


if __name__ == "__main__":
    unittest.main()