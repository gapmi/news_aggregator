from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import run_video_pipeline as pipeline


class VideoPipelineProfileTests(unittest.TestCase):
    def test_default_profile_is_horizontal(self) -> None:
        argv = [
            "run_video_pipeline",
            "--run-id",
            "568",
        ]

        with patch.object(sys, "argv", argv):
            args = pipeline.parse_args()

        self.assertEqual(args.profile, "news_horizontal")
        self.assertFalse(args.existing_inputs_only)

    def test_existing_inputs_rejects_generation_flags(self) -> None:
        for flag in (
            "--force-script",
            "--force-tts",
            "--force-assets",
        ):
            with self.subTest(flag=flag):
                argv = [
                    "run_video_pipeline",
                    "--run-id",
                    "568",
                    "--profile",
                    "news_vertical",
                    "--existing-inputs-only",
                    flag,
                ]

                with (
                    patch.object(sys, "argv", argv),
                    patch.object(pipeline, "get_conn") as get_conn,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cannot be combined",
                    ):
                        pipeline.main()

                get_conn.assert_not_called()

    def test_existing_inputs_skips_generation(self) -> None:
        argv = [
            "run_video_pipeline",
            "--run-id",
            "568",
            "--profile",
            "news_vertical",
            "--existing-inputs-only",
            "--skip-render",
        ]

        validated_inputs = {
            "scene_count": 8,
            "audio_duration_seconds": 100.88,
            "scene_manifest_sha256": "script-test-hash",
            "audio_manifest_sha256": "audio-test-hash",
            "source_assets_manifest_sha256": "assets-test-hash",
        }

        with tempfile.TemporaryDirectory() as temporary:
            artifacts_root = Path(temporary)
            run_root = artifacts_root / "run_568"
            render_assets = run_root / "vertical" / "render_assets"
            conn = MagicMock()

            def fake_run_command(
                command: list[str],
                *,
                env=None,
            ) -> None:
                self.assertEqual(
                    command[2],
                    "scripts.build_render_manifest",
                )

                profile_index = command.index("--profile")
                self.assertEqual(
                    command[profile_index + 1],
                    "news_vertical",
                )

                output_index = command.index("--output-dir")
                output_dir = Path(command[output_index + 1])
                self.assertEqual(output_dir, render_assets)

                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "render_manifest.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )

            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "ARTIFACTS_ROOT",
                        artifacts_root,
                    )
                )

                mocks = {}

                replacements = {
                    "get_conn": {"return_value": conn},
                    "try_acquire_lock": {"return_value": True},
                    "release_lock": {},
                    "validate_clustering_run": {},
                    "find_existing_success": {"return_value": None},
                    "start_pipeline_record": {"return_value": 123},
                    "update_pipeline_meta": {},
                    "complete_pipeline_record": {},
                    "validate_existing_inputs": {
                        "return_value": validated_inputs,
                    },
                    "run_command": {
                        "side_effect": fake_run_command,
                    },
                    "has_valid_mistral_script": {},
                    "has_valid_tts_output": {},
                }

                for name, options in replacements.items():
                    mocks[name] = stack.enter_context(
                        patch.object(pipeline, name, **options)
                    )

                result = pipeline.main()

            self.assertEqual(result, 0)

            mocks["validate_existing_inputs"].assert_called_once()
            mocks["run_command"].assert_called_once()
            mocks["has_valid_mistral_script"].assert_not_called()
            mocks["has_valid_tts_output"].assert_not_called()
            mocks["release_lock"].assert_called_once_with(conn)
            conn.close.assert_called_once()

            mocks["complete_pipeline_record"].assert_called_once()
            completion = mocks[
                "complete_pipeline_record"
            ].call_args.kwargs

            self.assertEqual(completion["status"], "success")
            self.assertEqual(
                completion["patch"]["stage"],
                "render_manifest_ready",
            )

            state_path = (
                run_root / "vertical" / ".pipeline_state.json"
            )
            state = json.loads(
                state_path.read_text(encoding="utf-8")
            )

            self.assertEqual(state["profile"], "news_vertical")
            self.assertTrue(state["existing_inputs_only"])
            self.assertEqual(
                state["stage"],
                "render_manifest_ready",
            )

            self.assertFalse(
                (run_root / ".pipeline_state.json").exists()
            )
            self.assertFalse((run_root / "render").exists())

    def test_existing_output_is_protected(self) -> None:
        argv = [
            "run_video_pipeline",
            "--run-id",
            "568",
            "--profile",
            "news_vertical",
            "--existing-inputs-only",
        ]

        with tempfile.TemporaryDirectory() as temporary:
            artifacts_root = Path(temporary)
            profile_root = artifacts_root / "run_568" / "vertical"
            render_dir = profile_root / "render"
            render_dir.mkdir(parents=True)

            final_path = render_dir / "final_568.mp4"
            final_path.write_bytes(b"existing-video")

            state_path = profile_root / ".pipeline_state.json"
            original_state = {
                "run_id": 568,
                "profile": "news_vertical",
                "pipeline_run_id": 999,
                "status": "success",
                "stage": "rendered",
            }
            state_path.write_text(
                json.dumps(original_state),
                encoding="utf-8",
            )

            original_video = final_path.read_bytes()
            original_state_bytes = state_path.read_bytes()
            conn = MagicMock()

            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "ARTIFACTS_ROOT",
                        artifacts_root,
                    )
                )
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "get_conn",
                        return_value=conn,
                    )
                )
                stack.enter_context(
                    patch.object(
                        pipeline,
                        "try_acquire_lock",
                        return_value=True,
                    )
                )
                release_lock = stack.enter_context(
                    patch.object(pipeline, "release_lock")
                )
                stack.enter_context(
                    patch.object(pipeline, "validate_clustering_run")
                )
                start_record = stack.enter_context(
                    patch.object(pipeline, "start_pipeline_record")
                )
                validate_inputs = stack.enter_context(
                    patch.object(pipeline, "validate_existing_inputs")
                )
                run_command = stack.enter_context(
                    patch.object(pipeline, "run_command")
                )

                with self.assertLogs(
                    pipeline.log,
                    level="ERROR",
                ) as captured:
                    result = pipeline.main()

            self.assertEqual(result, 1)
            self.assertTrue(
                any(
                    "Render directory is not empty" in entry
                    for entry in captured.output
                )
            )

            start_record.assert_not_called()
            validate_inputs.assert_not_called()
            run_command.assert_not_called()
            release_lock.assert_called_once_with(conn)
            conn.close.assert_called_once()

            self.assertEqual(
                final_path.read_bytes(),
                original_video,
            )
            self.assertEqual(
                state_path.read_bytes(),
                original_state_bytes,
            )
    def test_existing_inputs_rejects_narration_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()

            script_path = root / "scene_manifest.json"
            audio_path = audio_dir / "audio_manifest.json"
            assets_path = root / "assets.csv"

            script_scene = {
                "scene_number": 1,
                "target_duration_seconds": 10,
                "narration": "Original narration.",
            }
            audio_scene = {
                "run_id": 568,
                "scene_number": 1,
                "target_duration_seconds": 10,
                "actual_duration_seconds": 5.0,
                "audio_file": "scene_001.wav",
                "narration": "Different narration.",
            }

            script_path.write_text(
                json.dumps({
                    "run_id": 568,
                    "scene_count": 1,
                    "scenes": [script_scene],
                }),
                encoding="utf-8",
            )
            audio_path.write_text(
                json.dumps({
                    "run_id": 568,
                    "scene_count": 1,
                    "total_actual_duration_seconds": 5.0,
                    "scenes": [audio_scene],
                }),
                encoding="utf-8",
            )
            (audio_dir / "scene_001_tts.json").write_text(
                json.dumps(audio_scene),
                encoding="utf-8",
            )

            with patch.object(pipeline, "probe_mp4") as probe:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "narration mismatch",
                ):
                    pipeline.validate_existing_inputs(
                        run_id=568,
                        scene_manifest_path=script_path,
                        audio_manifest_path=audio_path,
                        assets_manifest_path=assets_path,
                    )

            probe.assert_not_called()

    def test_existing_inputs_rejects_missing_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()

            script_path = root / "scene_manifest.json"
            audio_path = audio_dir / "audio_manifest.json"
            assets_path = root / "assets.csv"

            narration = "Matching narration."

            script_scene = {
                "scene_number": 1,
                "target_duration_seconds": 10,
                "narration": narration,
            }
            audio_scene = {
                "run_id": 568,
                "scene_number": 1,
                "target_duration_seconds": 10,
                "actual_duration_seconds": 5.0,
                "audio_file": "scene_001.wav",
                "narration": narration,
            }

            script_path.write_text(
                json.dumps({
                    "run_id": 568,
                    "scene_count": 1,
                    "scenes": [script_scene],
                }),
                encoding="utf-8",
            )
            audio_path.write_text(
                json.dumps({
                    "run_id": 568,
                    "scene_count": 1,
                    "total_actual_duration_seconds": 5.0,
                    "scenes": [audio_scene],
                }),
                encoding="utf-8",
            )
            (audio_dir / "scene_001_tts.json").write_text(
                json.dumps(audio_scene),
                encoding="utf-8",
            )

            with patch.object(pipeline, "probe_mp4") as probe:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "WAV missing or empty",
                ):
                    pipeline.validate_existing_inputs(
                        run_id=568,
                        scene_manifest_path=script_path,
                        audio_manifest_path=audio_path,
                        assets_manifest_path=assets_path,
                    )

            probe.assert_not_called()
if __name__ == "__main__":
    unittest.main()