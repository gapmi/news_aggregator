from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts import render_news_video_v2 as renderer


class VideoSubtitleStyleTests(unittest.TestCase):
    def test_horizontal_style_is_unchanged(self):
        profile = SimpleNamespace(name="news_horizontal")

        self.assertEqual(
            renderer.build_subtitle_style(profile),
            (
                "FontName=DejaVu Sans,"
                "FontSize=20,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=1,"
                "Outline=2,"
                "Shadow=0,"
                "Alignment=2,"
                "MarginL=30,MarginR=30,MarginV=24"
            ),
        )

    def test_vertical_style_is_smaller_and_lower(self):
        profile = SimpleNamespace(name="news_vertical")

        self.assertEqual(
            renderer.build_subtitle_style(profile),
            (
                "FontName=DejaVu Sans,"
                "FontSize=14,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=1,"
                "Outline=1,"
                "Shadow=0,"
                "Alignment=2,"
                "MarginL=24,MarginR=24,MarginV=24"
            ),
        )


if __name__ == "__main__":
    unittest.main()