import unittest
from types import SimpleNamespace

from clustering.mistral_validation import (
    FORBIDDEN_FAKE_DOCUMENTARY_PATTERNS,
    FORBIDDEN_VISUAL_PROMPT_PATTERNS,
    REQUIRED_VISUAL_PROMPT_SUFFIX,
    _find_pattern_matches,
    _normalize_visual_prompt_suffix,
)

RUN_600_TAIL = (
    "cinematic editorial news documentary, restrained colors, "
    "natural lighting, realistic camera movement, 16:9, "
    "no text, no logos, no watermark"
)


def make_script(*prompts):
    return SimpleNamespace(
        scenes=[SimpleNamespace(visual_prompt=value) for value in prompts]
    )


class VisualPromptSuffixTests(unittest.TestCase):
    def normalized(self, prompt):
        script = make_script(prompt)
        _normalize_visual_prompt_suffix(script)
        return script.scenes[0].visual_prompt

    def test_run_600_tail_without_clean_composition_is_repaired(self):
        prompt = (
            "Empty modern courthouse exterior with American flag, "
            "overcast sky, clean composition, " + RUN_600_TAIL
        )
        result = self.normalized(prompt)
        self.assertTrue(result.endswith(REQUIRED_VISUAL_PROMPT_SUFFIX))
        self.assertTrue(result.startswith("Empty modern courthouse"))

    def test_prompt_without_style_tail_gets_suffix(self):
        result = self.normalized("Snow-covered rural road with bare trees")
        self.assertEqual(
            "Snow-covered rural road with bare trees, "
            + REQUIRED_VISUAL_PROMPT_SUFFIX,
            result,
        )

    def test_valid_prompt_is_unchanged(self):
        prompt = "Tropical coastline, " + REQUIRED_VISUAL_PROMPT_SUFFIX
        self.assertEqual(prompt, self.normalized(prompt))

    def test_prompt_with_only_style_tail_is_not_rewritten(self):
        self.assertEqual(RUN_600_TAIL, self.normalized(RUN_600_TAIL))

    def test_forbidden_words_still_detected_after_normalization(self):
        result = self.normalized("Large protest in a city street, " + RUN_600_TAIL)
        matches = _find_pattern_matches(
            result, FORBIDDEN_FAKE_DOCUMENTARY_PATTERNS
        )
        self.assertIn("protest", matches)
        self.assertEqual(
            [],
            _find_pattern_matches(
                REQUIRED_VISUAL_PROMPT_SUFFIX,
                FORBIDDEN_VISUAL_PROMPT_PATTERNS,
            ),
        )


if __name__ == "__main__":
    unittest.main()