import unittest

from clustering.mistral_validation import (
    FORBIDDEN_VISUAL_PROMPT_PATTERNS,
    REQUIRED_VISUAL_PROMPT_SUFFIX,
    _find_pattern_matches,
)


class MistralVisualNegationTests(unittest.TestCase):
    def matches(self, value):
        return _find_pattern_matches(
            value,
            FORBIDDEN_VISUAL_PROMPT_PATTERNS,
        )

    def test_negative_label_forms_are_allowed(self):
        for value in (
            "no labels",
            "no borders or labels",
            "no borders and labels",
            "NO BORDERS OR LABELS",
            "no borders   or\nlabels",
        ):
            with self.subTest(value=value):
                self.assertNotIn("label", self.matches(value))

    def test_run_574_negative_phrase_is_allowed(self):
        value = (
            "Abstract global map with subtle glowing lines representing "
            "population growth, soft blue and green tones, "
            "no borders or labels, "
            + REQUIRED_VISUAL_PROMPT_SUFFIX
        )
        self.assertEqual([], self.matches(value))

    def test_positive_labels_remain_forbidden(self):
        for value in (
            "visible labels on a map",
            "no borders, visible labels on a map",
            "no borders or labels, then add labels",
        ):
            with self.subTest(value=value):
                self.assertIn("label", self.matches(value))

    def test_positive_graph_remains_forbidden(self):
        value = (
            "Abstract digital particles forming a rising graph, "
            "soft neon glow, dark background, no text or numbers, "
            + REQUIRED_VISUAL_PROMPT_SUFFIX
        )
        self.assertIn("graph", self.matches(value))

    def test_required_suffix_remains_allowed(self):
        self.assertEqual(
            [],
            self.matches(REQUIRED_VISUAL_PROMPT_SUFFIX),
        )


if __name__ == "__main__":
    unittest.main()