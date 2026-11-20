from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import render_news_video_v2 as renderer
from scripts.render_news_video_v2 import plan_vertical_photos


class VerticalPhotoSequenceTests(unittest.TestCase):
    def test_repeats_photos_and_trims_last_segment(self) -> None:
        plan = plan_vertical_photos(
            asset_count=3,
            total_frames=1051,
            fps=30,
        )

        self.assertEqual(
            plan,
            [
                (0, 300),
                (1, 300),
                (2, 300),
                (0, 151),
            ],
        )
        self.assertEqual(
            sum(frames for _, frames in plan),
            1051,
        )

    def test_short_audio_uses_only_first_photo(self) -> None:
        self.assertEqual(
            plan_vertical_photos(
                asset_count=3,
                total_frames=45,
                fps=30,
            ),
            [(0, 45)],
        )

    def test_exact_cycle_has_no_extra_segment(self) -> None:
        self.assertEqual(
            plan_vertical_photos(
                asset_count=2,
                total_frames=600,
                fps=30,
            ),
            [(0, 300), (1, 300)],
        )

    def test_rejects_invalid_arguments(self) -> None:
        for arguments in (
            {"asset_count": 0, "total_frames": 30, "fps": 30},
            {"asset_count": 1, "total_frames": 0, "fps": 30},
            {"asset_count": 1, "total_frames": 30, "fps": 0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    plan_vertical_photos(**arguments)

    def test_render_scene_uses_sequence_and_muxes_audio_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = [
                root / "photo_01.png",
                root / "photo_02.png",
            ]
            audio_path = root / "scene_001.wav"
            scenes_dir = root / "scenes"
            scenes_dir.mkdir()

            scene = {
                "scene_number": 1,
                "actual_duration_seconds": 25.0,
                "audio_path": str(audio_path),
                "assets": [str(path) for path in assets],
                "visual_type": "photo",
                "narration": "Test narration continues across all photos.",
            }
            profile = renderer.get_video_profile("news_vertical")

            with (
                patch.object(
                    renderer,
                    "render_asset_segment",
                ) as render_segment,
                patch.object(
                    renderer,
                    "concat_video_segments",
                ) as concat_segments,
                patch.object(
                    renderer,
                    "write_scene_subtitles",
                    return_value=root / "subtitles.srt",
                ) as write_subtitles,
                patch.object(
                    renderer,
                    "mux_audio",
                ) as mux_audio,
            ):
                result = renderer.render_scene(
                    scene,
                    scenes_dir,
                    profile=profile,
                )

            self.assertEqual(
                [
                    (
                        call.kwargs["asset_path"],
                        call.kwargs["frames"],
                    )
                    for call in render_segment.call_args_list
                ],
                [
                    (assets[0], 300),
                    (assets[1], 300),
                    (assets[0], 150),
                ],
            )

            expected_segments = [
                scenes_dir / "scene_001_parts" / "part_01.mp4",
                scenes_dir / "scene_001_parts" / "part_02.mp4",
                scenes_dir / "scene_001_parts" / "part_03.mp4",
            ]
            self.assertEqual(
                [
                    call.kwargs["output_path"]
                    for call in render_segment.call_args_list
                ],
                expected_segments,
            )
            concat_segments.assert_called_once_with(
                segment_paths=expected_segments,
                output_path=(
                    scenes_dir / "scene_001_silent.mp4"
                ),
                temp_dir=scenes_dir / "scene_001_parts",
            )

            write_subtitles.assert_called_once_with(
                narration=scene["narration"],
                duration=25.0,
                output_path=(
                    scenes_dir
                    / "scene_001_parts"
                    / "subtitles.srt"
                ),
                words_per_caption=6,
            )
            mux_audio.assert_called_once_with(
                video_path=(
                    scenes_dir / "scene_001_silent.mp4"
                ),
                audio_path=audio_path,
                output_path=scenes_dir / "scene_001.mp4",
                subtitles_path=root / "subtitles.srt",
                profile=profile,
            )
            self.assertEqual(
                result,
                scenes_dir / "scene_001.mp4",
            )
if __name__ == "__main__":
    unittest.main()