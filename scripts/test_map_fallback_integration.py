import csv
import json
import random
import subprocess
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    project = Path(__file__).resolve().parents[1]

    with TemporaryDirectory(prefix="map_integration_") as temporary:
        root = Path(temporary)
        output = root / "output"
        geojson = root / "countries.geojson"
        audio_manifest = root / "audio_manifest.json"
        assets_manifest = root / "assets.csv"

        features = []
        for code, name, longitude in [
            ("GTM", "Guatemala", -90),
            ("CHN", "China", 105),
            ("USA", "United States of America", -105),
        ]:
            features.append({
                "type": "Feature",
                "properties": {
                    "ADM0_A3": code,
                    "NAME_EN": name,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [longitude - 2, 10],
                        [longitude + 2, 10],
                        [longitude + 2, 14],
                        [longitude - 2, 14],
                        [longitude - 2, 10],
                    ]],
                },
            })

        write_json(geojson, {
            "type": "FeatureCollection",
            "features": features,
        })

        audio = root / "audio.wav"
        with wave.open(str(audio), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(8000)
            file.writeframes(b"\x00\x00" * 8000)

        narrations = [
            "Guatemala has intensified security operations.",
            "China announced restrictions involving the United States.",
            "Officials announced new measures.",
            "Guatemala has announced new measures.",
        ]

        write_json(audio_manifest, {
            "run_id": 1,
            "scenes": [
                {
                    "scene_number": number,
                    "audio_file": audio.name,
                    "actual_duration_seconds": 1.0,
                    "narration": narration,
                }
                for number, narration in enumerate(narrations, start=1)
            ],
        })

        # Detailed synthetic image to exercise the existing image branch.
        source_image = root / "source.png"
        rng = random.Random(42)
        image = Image.frombytes(
            "RGB",
            (320, 180),
            rng.randbytes(320 * 180 * 3),
        )
        image.save(source_image)

        columns = [
            "scene_number",
            "candidate_rank",
            "is_representative",
            "asset_type",
            "asset_source",
            "asset_path",
            "status",
        ]

        with assets_manifest.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerow({
                "scene_number": 4,
                "candidate_rank": 1,
                "is_representative": "true",
                "asset_type": "image",
                "asset_source": "rss_image",
                "asset_path": str(source_image),
                "status": "ok",
            })

        command = [
            sys.executable,
            "-m",
            "scripts.build_render_manifest",
            "--run-id", "1",
            "--audio-manifest", str(audio_manifest),
            "--assets-manifest", str(assets_manifest),
            "--output-dir", str(output),
            "--countries-geojson", str(geojson),
        ]

        result = subprocess.run(
            command,
            cwd=project,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(
                f"Builder failed:\n{result.stdout}\n{result.stderr}"
            )

        manifest = json.loads(
            (output / "render_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        selection = json.loads(
            (output / "selected_assets.json").read_text(
                encoding="utf-8"
            )
        )

        assert manifest["scene_count"] == 4
        selected = {
            row["scene_number"]: row
            for row in selection["assets"]
        }
        assert len(selected) == 4

        expected = [
            (1, "country", "GTM", "single_country_mention"),
            (2, "world", None, "multiple_countries"),
            (3, "world", None, "no_country"),
        ]

        for number, mode, code, status in expected:
            row = selected[number]
            assert row["asset_type"] == "generated_map", row
            assert row["map_mode"] == mode, row
            assert row["highlighted_country_code"] == code, row
            assert row["geography_resolution"]["status"] == status, row

            with Image.open(row["render_asset_path"]) as rendered:
                assert rendered.size == (2400, 1350)

            print(f"PASS: scene {number}: {mode}, country={code}")

        photo = selected[4]
        assert photo["asset_type"] == "image", photo
        assert photo["asset_source"] == "rss_image", photo
        assert photo["fallback_reason"] is None, photo

        with Image.open(photo["render_asset_path"]) as copied:
            assert copied.convert("RGB").tobytes() == image.tobytes()

        print("PASS: scene 4: source image preserved")

        for number, scene in enumerate(manifest["scenes"], start=1):
            assert scene["narration"] == narrations[number - 1]
            assert Path(scene["audio_path"]).is_file()
            assert len(scene["assets"]) == 1
            assert Path(scene["assets"][0]).is_file()

        print("PASS: narration, audio paths and asset paths preserved")

    print("PASS: all integration checks completed")


if __name__ == "__main__":
    main()