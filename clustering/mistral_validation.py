from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MIN_SCENE_DURATION_SECONDS = 8
MAX_SCENE_DURATION_SECONDS = 20

MIN_WORDS_PER_SECOND = 2.1
MAX_WORDS_PER_SECOND = 2.8

ALLOWED_VISUAL_TYPES = {
    "editorial_b_roll",
    "abstract_data_visual",
    "map_animation",
    "generic_newsroom",
    "thematic_ai_video",
}

FORBIDDEN_VISUAL_PROMPT_PATTERNS = {
    "readable text": r"(?<!no\s)\breadable\s+text\b",
    "caption": r"(?<!no\s)\bcaptions?\b",
    "subtitle": r"(?<!no\s)\bsubtitles?\b",
    "headline": r"(?<!no\s)\bheadlines?\b",
    "ticker": r"(?<!no\s)\bticker\b",
    "visible logo": r"(?<!no\s)\blogos?\b",
    "visible watermark": r"(?<!no\s)\bwatermarks?\b",
    "label": r"(?<!un)\blabel(?:ed|led|ing|s)?\b",
    "labeled chart": r"\blabel(?:ed|led)\s+(?:bar\s+)?chart\b",
    "bar chart": r"\bbar\s+chart\b",
    "line chart": r"\bline\s+chart\b",
    "pie chart": r"\bpie\s+chart\b",
    "infographic": r"\binfographic\b",
    "graph": r"\bgraph\b",
    "table": r"\btable\b",
    "percentage": r"\bpercentage\b",
    "screen displaying": r"\bscreens?\s+displaying\b",
    "signage": r"(?<!no\s)\bsignage\b",
    "document": r"(?<!no\s)\bdocuments?\b",
    "social post": r"(?<!no\s)\bsocial\s+(?:media\s+)?posts?\b",
    "news ticker": r"\bnews\s+ticker\b",
}

FORBIDDEN_FAKE_DOCUMENTARY_PATTERNS = {
    "rescue team": r"\brescue\s+teams?\b",
    "search and rescue": r"\bsearch\s+and\s+rescue\b",
    "emergency worker": r"\bemergency\s+workers?\b",
    "soldier": r"\bsoldiers?\b",
    "military vehicle": r"\bmilitary\s+vehicles?\b",
    "weapon": r"\bweapons?\b",
    "battlefield": r"\bbattlefields?\b",
    "active conflict": r"\bactive\s+conflict\b",
    "active attack": r"\bactive\s+attacks?\b",
    "explosion": r"\bexplosions?\b",
    "blast": r"\bblasts?\b",
    "flooded area": r"\bflood(?:ed)?\s+areas?\b",
    "flood affected": r"\bflood[-\s]?affected\b",
    "wildfire scene": r"\bwildfire\s+scene\b",
    "firefighters": r"\bfirefighters?\b",
    "destroyed building": r"\bdestroyed\s+buildings?\b",
    "injured person": r"\binjured\s+(?:people|person|civilians?)\b",
    "dead body": r"\bdead\s+bodies\b",
    "casualty": r"\bcasualties\b",
    "protest": r"\bprotests?\b",
    "election": r"\belections?\b",
    "government meeting": r"\bgovernment\s+meetings?\b",
    "diplomatic meeting": r"\bdiplomatic\s+meetings?\b",
    "officials in discussion": r"\bofficials?\s+in\s+discussion\b",
    "identifiable official": r"\bidentifiable\s+officials?\b",
    "intelligence official": r"\bintelligence\s+officials?\b",
}

FORBIDDEN_NARRATION_TERMS = {
    "run": r"\bruns?\b",
    "cluster": r"\bclusters?\b",
    "lineage": r"\blineage\b",
    "anchor": r"\banchor\b",
    "threshold": r"\bthreshold\b",
    "payload": r"\bpayload\b",
    "similarity": r"\bsimilarity\b",
    "overlap": r"\boverlap\b",
    "score": r"\bscore\b",
    "core": r"\bcore\b",
    "edge": r"\bedge\b",
    "outlier": r"\boutliers?\b",
    "subcluster": r"\bsubclusters?\b",
    "database": r"\bdatabase\b",
    "algorithm": r"\balgorithm\b",
    "parent size": r"\bparent[_\s-]?size\b",
    "child size": r"\bchild[_\s-]?size\b",
}

FORBIDDEN_FIRST_PERSON_PATTERNS = {
    "I": r"\bI\b",
    "we": r"\bwe\b",
    "our": r"\bour\b",
    "you": r"\byou\b",
    "your": r"\byour\b",
}


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Literal["en"]
    title: str = Field(min_length=5, max_length=160)
    description: str = Field(min_length=20, max_length=2000)
    estimated_duration_seconds: int = Field(
        ge=30,
        le=3600,
    )
    editorial_angle: str = Field(min_length=10, max_length=500)


class NarrationScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_voiceover: str = Field(min_length=1, max_length=20000)


class VideoScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_number: int = Field(ge=1)
    duration_seconds: int = Field(
        ge=MIN_SCENE_DURATION_SECONDS,
        le=MAX_SCENE_DURATION_SECONDS,
    )
    narration: str = Field(min_length=1, max_length=3000)
    visual_type: Literal[
        "editorial_b_roll",
        "abstract_data_visual",
        "map_animation",
        "generic_newsroom",
        "thematic_ai_video",
    ]
    visual_prompt: str = Field(min_length=1, max_length=5000)
    topic_references: list[str] = Field(min_length=1, max_length=3)


class CoverageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continuing_topics: list[str]
    newly_visible_topics: list[str]
    reframed_topics: list[str]
    declining_topics: list[str]


class MistralVideoScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_metadata: VideoMetadata
    narration_script: NarrationScript
    scenes: list[VideoScene] = Field(min_length=2, max_length=100)
    coverage_summary: CoverageSummary
    fact_check_notes: list[str] = Field(min_length=1, max_length=30)


class MistralVideoValidationError(ValueError):
    """
    Raised when an LLM response is syntactically valid JSON but fails
    required production, factual-safety, or media-generation invariants.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            "Mistral video script validation failed:\n- "
            + "\n- ".join(errors)
        )


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _word_count(value: str) -> int:
    """
    Counts English-like words and also tolerates Cyrillic tokens in labels.
    """
    return len(
        re.findall(
            r"[A-Za-zА-Яа-яЁё0-9]+(?:['’-][A-Za-zА-Яа-яЁё0-9]+)?",
            value,
        )
    )


def _find_pattern_matches(
    value: str,
    patterns: dict[str, str],
) -> list[str]:
    """
    Finds forbidden prompt terms while allowing the mandatory negative suffix:
    'no text, no logos, no watermark'.

    The function does not treat explicit negative safety instructions as a
    request to generate those elements.
    """
    normalized = _normalize_whitespace(value).lower()

    normalized = re.sub(
        r"\bno\s+(?:readable\s+)?text\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+logos?\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+watermarks?\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+captions?\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+subtitles?\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+headlines?\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bno\s+(?:news\s+)?ticker\b",
        "",
        normalized,
    )

    matches: list[str] = []

    for label, pattern in patterns.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            matches.append(label)

    return matches


def _scene_word_bounds(duration_seconds: int) -> tuple[int, int]:
    min_words = round(duration_seconds * MIN_WORDS_PER_SECOND)
    max_words = round(duration_seconds * MAX_WORDS_PER_SECOND)
    return min_words, max_words


def _allowed_topic_names(input_payload: dict[str, Any]) -> set[str]:
    editorial_topics = input_payload.get("editorial_topics")

    if not isinstance(editorial_topics, list):
        raise ValueError(
            "input_payload.editorial_topics must be a list for validation"
        )

    names: set[str] = set()

    for topic in editorial_topics:
        if not isinstance(topic, dict):
            raise ValueError(
                "input_payload.editorial_topics must contain objects"
            )

        topic_name = topic.get("topic_name")

        if not isinstance(topic_name, str) or not topic_name.strip():
            raise ValueError(
                "Each input_payload.editorial_topics item must have topic_name"
            )

        names.add(topic_name)

    if not names:
        raise ValueError(
            "input_payload.editorial_topics must contain at least one topic"
        )

    return names


def _expected_target_duration(input_payload: dict[str, Any]) -> int:
    requirements = input_payload.get("video_requirements")

    if not isinstance(requirements, dict):
        raise ValueError(
            "input_payload.video_requirements must be an object"
        )

    duration = requirements.get("target_duration_seconds")

    if not isinstance(duration, int):
        raise ValueError(
            "input_payload.video_requirements.target_duration_seconds "
            "must be an integer"
        )

    if duration < 30:
        raise ValueError(
            "input_payload.video_requirements.target_duration_seconds "
            "must be at least 30"
        )

    return duration


def parse_mistral_video_script(
    raw_content: str,
) -> MistralVideoScript:
    """
    Parses raw Mistral message content and validates its JSON/Pydantic shape.

    Runtime constraints such as word count, duration sum and visual safety
    are validated separately by validate_mistral_video_script().
    """
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise MistralVideoValidationError(
            [f"Response is not valid JSON: {exc.msg}"]
        ) from exc

    if not isinstance(parsed, dict):
        raise MistralVideoValidationError(
            ["Response root must be a JSON object"]
        )

    try:
        return MistralVideoScript.model_validate(parsed)
    except ValidationError as exc:
        errors = []

        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            message = error["msg"]
            errors.append(f"Schema error at '{location}': {message}")

        raise MistralVideoValidationError(errors) from exc


def validate_mistral_video_script(
    script: MistralVideoScript,
    input_payload: dict[str, Any],
) -> None:
    """
    Validates cross-field, duration, narration and visual-safety invariants.

    Raises MistralVideoValidationError if any production requirement fails.
    """
    errors: list[str] = []

    target_duration_seconds = _expected_target_duration(input_payload)
    allowed_topics = _allowed_topic_names(input_payload)

    scenes = script.scenes
    scene_numbers = [scene.scene_number for scene in scenes]
    expected_scene_numbers = list(range(1, len(scenes) + 1))

    if scene_numbers != expected_scene_numbers:
        errors.append(
            "Scene numbers must start at 1 and increase without gaps: "
            f"expected={expected_scene_numbers}, actual={scene_numbers}"
        )

    duration_sum = sum(scene.duration_seconds for scene in scenes)

    if duration_sum != target_duration_seconds:
        errors.append(
            "Scene duration sum must equal target duration: "
            f"expected={target_duration_seconds}, actual={duration_sum}"
        )

    if (
        script.video_metadata.estimated_duration_seconds
        != target_duration_seconds
    ):
        errors.append(
            "video_metadata.estimated_duration_seconds must equal target "
            f"duration: expected={target_duration_seconds}, "
            f"actual={script.video_metadata.estimated_duration_seconds}"
        )

    expected_full_voiceover = _normalize_whitespace(
        " ".join(scene.narration for scene in scenes)
    )
    actual_full_voiceover = _normalize_whitespace(
        script.narration_script.full_voiceover
    )

    if actual_full_voiceover != expected_full_voiceover:
        errors.append(
            "narration_script.full_voiceover must exactly equal the "
            "concatenation of scenes[].narration in scene-number order"
        )

    voiceover_word_count = _word_count(
        script.narration_script.full_voiceover
    )
    min_voiceover_words = round(
        target_duration_seconds * MIN_WORDS_PER_SECOND
    )
    max_voiceover_words = round(
        target_duration_seconds * MAX_WORDS_PER_SECOND
    )

    if voiceover_word_count < min_voiceover_words:
        errors.append(
            "full_voiceover is too short for the target duration: "
            f"expected_at_least={min_voiceover_words} words, "
            f"actual={voiceover_word_count}"
        )

    if voiceover_word_count > max_voiceover_words:
        errors.append(
            "full_voiceover is too long for the target duration: "
            f"expected_at_most={max_voiceover_words} words, "
            f"actual={voiceover_word_count}"
        )

    narration_texts = [
        script.narration_script.full_voiceover,
        script.video_metadata.title,
        script.video_metadata.description,
        script.video_metadata.editorial_angle,
    ]
    narration_texts.extend(scene.narration for scene in scenes)

    for index, narration_text in enumerate(narration_texts):
        internal_terms = _find_pattern_matches(
            narration_text,
            FORBIDDEN_NARRATION_TERMS,
        )

        if internal_terms:
            errors.append(
                "Forbidden internal narration terms found in "
                f"narration_text[{index}]: {', '.join(internal_terms)}"
            )

        first_person_terms = _find_pattern_matches(
            narration_text,
            FORBIDDEN_FIRST_PERSON_PATTERNS,
        )

        if first_person_terms:
            errors.append(
                "First-person or direct-address terms found in "
                f"narration_text[{index}]: "
                f"{', '.join(first_person_terms)}"
            )

    all_scene_topics: set[str] = set()

    for scene in scenes:
        scene_words = _word_count(scene.narration)
        min_scene_words, max_scene_words = _scene_word_bounds(
            scene.duration_seconds
        )

        if scene_words < min_scene_words:
            errors.append(
                f"Scene {scene.scene_number} narration is too short: "
                f"expected_at_least={min_scene_words} words for "
                f"{scene.duration_seconds} seconds, actual={scene_words}"
            )

        if scene_words > max_scene_words:
            errors.append(
                f"Scene {scene.scene_number} narration is too long: "
                f"expected_at_most={max_scene_words} words for "
                f"{scene.duration_seconds} seconds, actual={scene_words}"
            )

        invalid_topics = [
            topic_name
            for topic_name in scene.topic_references
            if topic_name not in allowed_topics
        ]

        if invalid_topics:
            errors.append(
                f"Scene {scene.scene_number} has topic_references not found "
                f"in editorial_topics: {invalid_topics}"
            )

        all_scene_topics.update(scene.topic_references)

        text_matches = _find_pattern_matches(
            scene.visual_prompt,
            FORBIDDEN_VISUAL_PROMPT_PATTERNS,
        )

        if text_matches:
            errors.append(
                f"Scene {scene.scene_number} visual_prompt requests forbidden "
                f"textual visual elements: {', '.join(text_matches)}"
            )

        fake_documentary_matches = _find_pattern_matches(
            scene.visual_prompt,
            FORBIDDEN_FAKE_DOCUMENTARY_PATTERNS,
        )

        if fake_documentary_matches:
            errors.append(
                f"Scene {scene.scene_number} visual_prompt requests unsafe "
                f"fake-documentary content: "
                f"{', '.join(fake_documentary_matches)}"
            )

        expected_visual_suffix = (
            "cinematic editorial news documentary, restrained colors, "
            "natural lighting, realistic camera movement, clean composition, "
            "16:9, no text, no logos, no watermark"
        )

        normalized_prompt = _normalize_whitespace(
            scene.visual_prompt
        ).lower()
        normalized_suffix = _normalize_whitespace(
            expected_visual_suffix
        ).lower()

        if not normalized_prompt.endswith(normalized_suffix):
            errors.append(
                f"Scene {scene.scene_number} visual_prompt must end with the "
                "required safe editorial style suffix"
            )

    summary_lists = {
        "continuing_topics": script.coverage_summary.continuing_topics,
        "newly_visible_topics": script.coverage_summary.newly_visible_topics,
        "reframed_topics": script.coverage_summary.reframed_topics,
        "declining_topics": script.coverage_summary.declining_topics,
    }

    for field_name, topic_names in summary_lists.items():
        invalid_topics = [
            topic_name
            for topic_name in topic_names
            if topic_name not in allowed_topics
        ]

        if invalid_topics:
            errors.append(
                f"coverage_summary.{field_name} contains topics not present "
                f"in editorial_topics: {invalid_topics}"
            )

    if not all_scene_topics:
        errors.append(
            "At least one editorial topic must be referenced by the scenes"
        )

    if not script.fact_check_notes:
        errors.append("fact_check_notes must not be empty")

    if errors:
        raise MistralVideoValidationError(errors)


def parse_and_validate_mistral_video_script(
    raw_content: str,
    input_payload: dict[str, Any],
) -> MistralVideoScript:
    """
    One-call helper for normal pipeline use.

    1. Parse raw Mistral content as JSON.
    2. Validate the Pydantic schema.
    3. Validate production rules against the original input payload.
    4. Return a typed script only when every rule passes.
    """
    script = parse_mistral_video_script(raw_content)
    validate_mistral_video_script(script, input_payload)
    return script