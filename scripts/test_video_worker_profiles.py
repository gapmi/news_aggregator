from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from scripts import video_worker as worker


class VideoWorkerProfileTests(unittest.TestCase):
    def test_builds_horizontal_then_vertical(self) -> None:
        with patch.object(worker, "run_command") as run_command:
            worker.run_video_pipeline(570)

        self.assertEqual(run_command.call_count, 2)

        commands = [
            call.args[0]
            for call in run_command.call_args_list
        ]

        for command in commands:
            self.assertEqual(
                command[:3],
                [
                    worker.sys.executable,
                    "-m",
                    "scripts.run_video_pipeline",
                ],
            )
            self.assertEqual(
                command[command.index("--run-id") + 1],
                "570",
            )

        horizontal, vertical = commands

        self.assertEqual(
            horizontal[horizontal.index("--profile") + 1],
            "news_horizontal",
        )
        self.assertNotIn("--existing-inputs-only", horizontal)

        self.assertEqual(
            vertical[vertical.index("--profile") + 1],
            "news_vertical",
        )
        self.assertIn("--existing-inputs-only", vertical)

        for command in commands:
            self.assertNotIn("--force-script", command)
            self.assertNotIn("--force-tts", command)
            self.assertNotIn("--force-assets", command)
            self.assertNotIn("--force-render", command)

    def test_horizontal_failure_stops_sequence(self) -> None:
        error = subprocess.CalledProcessError(
            1,
            ["horizontal-test"],
        )

        with patch.object(
            worker,
            "run_command",
            side_effect=error,
        ) as run_command:
            with self.assertRaises(
                subprocess.CalledProcessError
            ):
                worker.run_video_pipeline(570)

        run_command.assert_called_once()

    def test_vertical_failure_is_not_hidden(self) -> None:
        error = subprocess.CalledProcessError(
            1,
            ["vertical-test"],
        )

        with patch.object(
            worker,
            "run_command",
            side_effect=[None, error],
        ) as run_command:
            with self.assertRaises(
                subprocess.CalledProcessError
            ):
                worker.run_video_pipeline(570)

        self.assertEqual(run_command.call_count, 2)


if __name__ == "__main__":
    unittest.main()