import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.map_fallback import BACKGROUND, render_world_map


def main():
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-30, -20],
                            [30, -20],
                            [30, 20],
                            [-30, 20],
                            [-30, -20],
                        ]
                    ],
                },
            }
        ],
    }

    with TemporaryDirectory(prefix="world_map_test_") as directory:
        directory = Path(directory)
        geojson = directory / "test.geojson"
        output = directory / "world.png"

        geojson.write_text(json.dumps(data), encoding="utf-8")

        result = render_world_map(
            geojson_path=geojson,
            output_path=output,
            width=2400,
            height=1350,
        )

        assert output.is_file() and output.stat().st_size > 0

        with Image.open(output) as image:
            assert image.size == (2400, 1350), image.size

            map_height = result["map_size"][1]
            caption = image.crop((0, map_height, 2400, 1350))

            expected = tuple(
                (channel, channel) for channel in BACKGROUND
            )
            assert caption.getextrema() == expected, (
                "Caption band is not empty"
            )

            assert image.getpixel(
                (1200, map_height // 2)
            ) != BACKGROUND, "Test polygon was not drawn"

        assert result["country_code"] is None
        assert result["mode"] == "world"
        assert result["caption_height"] == 297

    print(
        "PASS: PNG created; size 2400x1350; "
        "polygon drawn; caption band empty."
    )


if __name__ == "__main__":
    main()