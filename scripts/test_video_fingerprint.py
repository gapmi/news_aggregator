from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

from unittest.mock import Mock, patch
from scripts import video_fingerprint as fingerprint
from scripts.video_profiles import get_video_profile

from scripts.video_fingerprint import (
    build_fingerprint,
    changed_components,
)


class VideoFingerprintTests(unittest.TestCase):
    def test_same_content_has_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            second = root / "second.wav"

            first.write_bytes(b"same-audio")
            second.write_bytes(b"same-audio")

            previous = build_fingerprint(
                settings={"profile": "news_vertical"},
                files={"audio.scene_001": first},
            )
            current = build_fingerprint(
                settings={"profile": "news_vertical"},
                files={"audio.scene_001": second},
            )

            self.assertEqual(previous["sha256"], current["sha256"])
            self.assertEqual(
                changed_components(previous, current),
                [],
            )

    def test_changed_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.wav"
            path.write_bytes(b"original")

            previous = build_fingerprint(
                settings={},
                files={"audio.scene_001": path},
            )

            path.write_bytes(b"modified")

            current = build_fingerprint(
                settings={},
                files={"audio.scene_001": path},
            )

            self.assertNotEqual(previous["sha256"], current["sha256"])
            self.assertEqual(
                changed_components(previous, current),
                ["files.audio.scene_001"],
            )

    def test_changed_profile_is_detected(self) -> None:
        previous = build_fingerprint(
            settings={"profile": "news_horizontal"},
            files={},
        )
        current = build_fingerprint(
            settings={"profile": "news_vertical"},
            files={},
        )

        self.assertNotEqual(previous["sha256"], current["sha256"])
        self.assertEqual(
            changed_components(previous, current),
            ["settings.profile"],
        )

    def test_appearing_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.png"

            previous = build_fingerprint(
                settings={},
                files={"source.photo": path},
            )

            path.write_bytes(b"new-photo")

            current = build_fingerprint(
                settings={},
                files={"source.photo": path},
            )

            self.assertEqual(
                previous["payload"]["files"]["source.photo"]["status"],
                "missing",
            )
            self.assertNotEqual(previous["sha256"], current["sha256"])
            self.assertEqual(
                changed_components(previous, current),
                ["files.source.photo"],
            )

    def test_news_fingerprint_tracks_source_and_audio_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()

            script_path = root / "scene_manifest.json"
            audio_manifest_path = audio_dir / "audio_manifest.json"
            assets_path = root / "assets.csv"
            photo_path = root / "photo.png"
            wav_path = audio_dir / "scene_001.wav"

            script_path.write_text(
                json.dumps({"run_id": 568, "scenes": []}),
                encoding="utf-8",
            )
            audio_manifest_path.write_text(
                json.dumps({
                    "run_id": 568,
                    "scenes": [
                        {
                            "scene_number": 1,
                            "audio_file": "scene_001.wav",
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (audio_dir / "scene_001_tts.json").write_text(
                "{}",
                encoding="utf-8",
            )
            assets_path.write_text(
                "scene_number,status,asset_type,asset_path\n"
                "1,ok,image,photo.png\n",
                encoding="utf-8",
            )

            photo_path.write_bytes(b"original-photo")
            wav_path.write_bytes(b"original-audio")
            maps_dir = root / "media" / "maps"
            maps_dir.mkdir(parents=True)

            geojson_path = (
                maps_dir / "ne_110m_admin_0_countries.geojson"
            )
            geojson_path.write_bytes(b"original-map")

            profile = get_video_profile("news_vertical")

            def snapshot():
                return fingerprint.build_news_fingerprint(
                    run_id=568,
                    profile=profile,
                    project_root=root,
                    scene_manifest_path=script_path,
                    audio_manifest_path=audio_manifest_path,
                    assets_manifest_path=assets_path,
                    media_dir=root / "branding_media",
                    max_assets_per_scene=3,
                )

            with (
                patch.object(
                    fingerprint.subprocess,
                    "run",
                    return_value=Mock(
                        stdout="ffmpeg test build",
                    ),
                ) as ffmpeg_run,
                patch.object(
                    fingerprint,
                    "version",
                    return_value="test-pillow",
                ),
                patch.object(
                    Path,
                    "rglob",
                    return_value=[],
                ),
            ):
                original = snapshot()
                unchanged = snapshot()

                photo_path.write_bytes(b"modified-photo")
                changed_photo = snapshot()

                wav_path.write_bytes(b"modified-audio")
                changed_audio = snapshot()
                geojson_path.write_bytes(b"modified-map")
                changed_map = snapshot()

            self.assertEqual(
                original["sha256"],
                unchanged["sha256"],
            )
            self.assertEqual(
                fingerprint.changed_components(
                    original,
                    changed_photo,
                ),
                ["files.source.row_000001"],
            )
            self.assertEqual(
                fingerprint.changed_components(
                    changed_photo,
                    changed_audio,
                ),
                ["files.audio.scene_001"],
            )

            self.assertEqual(
                original["payload"]["files"]["maps.geojson"]["status"],
                "present",
            )
            self.assertEqual(
                fingerprint.changed_components(
                    changed_audio,
                    changed_map,
                ),
                ["files.maps.geojson"],
            )

            self.assertEqual(
                original["payload"]["settings"]["profile"]["name"],
                "news_vertical",
            )
            self.assertEqual(ffmpeg_run.call_count, 5)

            ffmpeg_run.assert_called_with(
                ["ffmpeg", "-version"],
                check=True,
                capture_output=True,
                text=True,
            )
    def test_saved_render_checks_record_inputs_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_path = root / "final_568.mp4"
            record_path = root / "render_fingerprint.json"

            original_video = b"video-A"
            final_path.write_bytes(original_video)

            inputs = fingerprint.build_fingerprint(
                settings={"profile": "news_vertical"},
                files={},
            )

            record = {
                "record_version": 1,
                "run_id": 568,
                "profile": "news_vertical",
                "inputs": inputs,
                "output": {
                    "filename": final_path.name,
                    "size_bytes": len(original_video),
                    "sha256": fingerprint.sha256_file(final_path),
                    "duration_seconds": 7.0,
                },
            }

            def write_record():
                record_path.write_text(
                    json.dumps(record),
                    encoding="utf-8",
                )

            def check(current_inputs=inputs):
                return fingerprint.check_saved_render(
                    record_path=record_path,
                    final_path=final_path,
                    run_id=568,
                    profile_name="news_vertical",
                    current_inputs=current_inputs,
                )

            matched, reason = check()
            self.assertFalse(matched)
            self.assertIn("Cannot verify saved render", reason)

            write_record()
            self.assertEqual(
                check(),
                (True, "Saved inputs and final video match"),
            )

            changed_inputs = fingerprint.build_fingerprint(
                settings={"profile": "news_horizontal"},
                files={},
            )
            self.assertEqual(
                check(changed_inputs),
                (
                    False,
                    "Render inputs differ from saved fingerprint",
                ),
            )

            final_path.write_bytes(b"video-B")
            self.assertEqual(
                check(),
                (False, "Final video SHA256 mismatch"),
            )

            final_path.write_bytes(original_video)
            record["profile"] = "news_horizontal"
            write_record()
            self.assertEqual(
                check(),
                (False, "Fingerprint run or profile mismatch"),
            )

            record_path.write_text("{broken", encoding="utf-8")
            matched, reason = check()
            self.assertFalse(matched)
            self.assertIn("Cannot verify saved render", reason)
if __name__ == "__main__":
    unittest.main()