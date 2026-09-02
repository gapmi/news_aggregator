from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MIN_SCENE_DURATION_SECONDS = 8
MAX_SCENE_DURATION_SECONDS = 20
MIN_VIDEO_DURATION_SECONDS = 120
MAX_VIDEO_DURATION_SECONDS = 600

MIN_SCENE_WORDS = 20

MIN_WORDS_PER_SECOND = 2.1
MAX_WORDS_PER_SECOND = 2.8

REQUIRED_VISUAL_PROMPT_SUFFIX = (
    "cinematic editorial news documentary, restrained colors, "
    "natural lighting, realistic camera movement, clean composition, "
    "16:9, no text, no logos, no watermark"
)

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
    "chart": r"\bcharts?\b",
    "graph": r"\bgraphs?\b",
    "table": r"\btables?\b",
    "infographic": r"\binfographics?\b",
    "dashboard": r"\bdashboards?\b",
    "data visualization": r"\bdata\s+visuali[sz]ation\b",
    "trend line": r"\btrend\s+lines?\b",
    "connecting lines": r"\bconnecting\s+lines?\b",
    "diagram": r"\bdiagrams?\b",
    "screen displaying": r"\bscreens?\s+displaying\b",
    "monitor": r"\bmonitors?\b",
    "television": r"\btelevisions?\b",
    "phone": r"\bphones?\b",
    "user interface": r"\buser\s+interfaces?\b",
    "social post": r"\bsocial\s+(?:media\s+)?posts?\b",
    "signage": r"(?<!no\s)\bsignage\b",
    "document": r"(?<!no\s)\bdocuments?\b",
    "newspaper": r"\bnewspapers?\b",
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
    "clustering": r"\bclustering\b",
    "window": r"\bwindows?\b",
    "lineage": r"\blineage\b",
    "centroid": r"\bcentroids?\b",
    "anchor": r"\banchor\b",
    "threshold": r"\bthreshold\b",
    "payload": r"\bpayload\b",
    "similarity": r"\bsimilarity\b",
    "overlap": r"\boverlap\b",
    "score": r"\bscore\b",
    "core": r"\bcore\b",
    "edge": r"\bedges?\b",
    "outlier": r"\boutliers?\b",
    "subcluster": r"\bsubclusters?\b",
    "database": r"\bdatabase\b",
    "algorithm": r"\balgorithm\b",
    "article count": r"\barticle\s+counts?\b",
    "topic reference": r"\btopic\s+references?\b",
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
    estimated_duration_seconds: int = Field(ge=30, le=3600)
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
    """Raised when a Mistral video script fails production validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            "Mistral video script validation failed:\n- "
            + "\n- ".join(errors)
        )


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _word_count(value: str) -> int:
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
    Find forbidden terms while allowing the required negative safety suffix.

    The suffix includes "no text", "no logos", and "no watermark"; these are
    protective instructions, not a request to render the named elements.
    """
    normalized = _normalize_whitespace(value).lower()

    negative_forms = [
        r"\bno\s+(?:readable\s+)?text\b",
        r"\bno\s+logos?\b",
        r"\bno\s+watermarks?\b",
        r"\bno\s+captions?\b",
        r"\bno\s+subtitles?\b",
        r"\bno\s+headlines?\b",
        r"\bno\s+(?:news\s+)?ticker\b",
        r"\bno\s+labels?\b",
        r"\bno\s+documents?\b",
        r"\bno\s+signage\b",
    ]

    for pattern in negative_forms:
        normalized = re.sub(pattern, "", normalized)

    matches: list[str] = []

    for label, pattern in patterns.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            matches.append(label)

    return matches


def _scene_word_bounds(duration_seconds: int) -> tuple[int, int]:
    return (
        MIN_SCENE_WORDS,
        round(duration_seconds * MAX_WORDS_PER_SECOND),
    )


def _allowed_topic_references(
    input_payload: dict[str, Any],
) -> set[str]:
    editorial_topics = input_payload.get("editorial_topics")

    if not isinstance(editorial_topics, list):
        raise ValueError(
            "input_payload.editorial_topics must be a list for validation"
        )

    references: set[str] = set()

    for topic in editorial_topics:
        if not isinstance(topic, dict):
            raise ValueError(
                "input_payload.editorial_topics must contain objects"
            )

        reference = topic.get("topic_reference")

        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                "Each editorial topic must contain a non-empty "
                "topic_reference"
            )

        references.add(reference)

    if not references:
        raise ValueError(
            "input_payload.editorial_topics must contain at least one topic"
        )

    return references


def _normalize_source_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9а-яё]+", " ", value)
    return " ".join(value.split())


def _source_aliases(source_name: str) -> set[str]:
    normalized = _normalize_source_text(source_name)

    aliases = {normalized}

    if normalized.startswith("google news "):
        aliases.add("google news")

    if " " in normalized:
        aliases.add(normalized.split()[0])

    if "." in source_name:
        domain = source_name.casefold().split("/")[0]
        aliases.add(_normalize_source_text(domain))

        hostname = domain.removeprefix("www.")
        aliases.add(_normalize_source_text(hostname))

        labels = hostname.split(".")
        if labels:
            aliases.add(_normalize_source_text(labels[0]))

    return {
        alias
        for alias in aliases
        if len(alias) >= 3
    }


def _allowed_source_names(
    input_payload: dict[str, Any],
) -> set[str]:
    sources: set[str] = set()

    editorial_topics = input_payload.get("editorial_topics")

    if not isinstance(editorial_topics, list):
        return sources

    for topic in editorial_topics:
        if not isinstance(topic, dict):
            continue

    for article in topic.get("evidence_articles") or []:
        if not isinstance(article, dict):
            continue

        candidates = [
            article.get("source_name"),
            *(article.get("source_aliases") or []),
        ]

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                sources.update(_source_aliases(candidate))

    return sources

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

    if not MIN_VIDEO_DURATION_SECONDS <= duration <= MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            "input_payload.video_requirements.target_duration_seconds "
            f"must be between {MIN_VIDEO_DURATION_SECONDS} and "
            f"{MAX_VIDEO_DURATION_SECONDS}"
        )

    return duration


def _source_names_mentioned(
    narration: str,
    allowed_sources: set[str],
) -> set[str]:
    normalized_narration = _normalize_source_text(narration)

    return {
        source
        for source in allowed_sources
        if source in normalized_narration
    }


def parse_mistral_video_script(raw_content: str) -> MistralVideoScript:
    """Parse JSON and validate the Pydantic object shape."""
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
            errors.append(
                f"Schema error at '{location}': {error['msg']}"
            )

        raise MistralVideoValidationError(errors) from exc


def validate_mistral_video_script(
    script: MistralVideoScript,
    input_payload: dict[str, Any],
) -> None:
    """
    Validate timing, source grounding, internal-language and visual invariants.

    It does not attempt to fact-check natural-language claims against article
    bodies because the current database stores article titles, not full text.
    It ensures that named sources were supplied in evidence articles.
    """
    errors: list[str] = []

    target_duration_seconds = _expected_target_duration(input_payload)
    allowed_references = _allowed_topic_references(input_payload)
    allowed_sources = _allowed_source_names(input_payload)

    scenes = script.scenes
    scene_numbers = [scene.scene_number for scene in scenes]
    expected_scene_numbers = list(range(1, len(scenes) + 1))

    if scene_numbers != expected_scene_numbers:
        errors.append(
            "Scene numbers must start at 1 and increase without gaps: "
            f"expected={expected_scene_numbers}, actual={scene_numbers}"
        )

    duration_sum = sum(scene.duration_seconds for scene in scenes)

    if not MIN_VIDEO_DURATION_SECONDS <= duration_sum <= MAX_VIDEO_DURATION_SECONDS:
        errors.append(
            "Scene duration sum must be within the allowed video duration range: "
            f"expected_between={MIN_VIDEO_DURATION_SECONDS}.."
            f"{MAX_VIDEO_DURATION_SECONDS}, actual={duration_sum}"
        )

    if not (
        MIN_VIDEO_DURATION_SECONDS
        <= script.video_metadata.estimated_duration_seconds
        <= MAX_VIDEO_DURATION_SECONDS
    ):
        errors.append(
            "video_metadata.estimated_duration_seconds must be within the "
            f"allowed video duration range: expected_between="
            f"{MIN_VIDEO_DURATION_SECONDS}..{MAX_VIDEO_DURATION_SECONDS}, "
            f"actual={script.video_metadata.estimated_duration_seconds}"
        )

    if script.video_metadata.estimated_duration_seconds != duration_sum:
        errors.append(
            "video_metadata.estimated_duration_seconds must equal the sum of "
            f"scene durations: expected={duration_sum}, "
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
    min_voiceover_words = max(
        len(scenes) * MIN_SCENE_WORDS,
        round(duration_sum * MIN_WORDS_PER_SECOND),
    )
    max_voiceover_words = round(
        duration_sum * MAX_WORDS_PER_SECOND
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

    public_texts = [
        script.narration_script.full_voiceover,
        script.video_metadata.title,
        script.video_metadata.description,
        script.video_metadata.editorial_angle,
    ]
    public_texts.extend(scene.narration for scene in scenes)

    for index, text in enumerate(public_texts):
        internal_terms = _find_pattern_matches(
            text,
            FORBIDDEN_NARRATION_TERMS,
        )

        if internal_terms:
            errors.append(
                "Forbidden internal narration terms found in "
                f"public_text[{index}]: {', '.join(internal_terms)}"
            )

        first_person_terms = _find_pattern_matches(
            text,
            FORBIDDEN_FIRST_PERSON_PATTERNS,
        )

        if first_person_terms:
            errors.append(
                "First-person or direct-address terms found in "
                f"public_text[{index}]: "
                f"{', '.join(first_person_terms)}"
            )

    all_scene_references: set[str] = set()

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

        invalid_references = [
            reference
            for reference in scene.topic_references
            if reference not in allowed_references
        ]

        if invalid_references:
            errors.append(
                f"Scene {scene.scene_number} has topic_references not found "
                f"in editorial_topics: {invalid_references}"
            )

        all_scene_references.update(scene.topic_references)

        mentioned_sources = _source_names_mentioned(
            scene.narration,
            allowed_sources,
        )

        if (
            re.search(r"\baccording to\b|\breported by\b", scene.narration,
                      flags=re.IGNORECASE)
            and not mentioned_sources
        ):
            errors.append(
                f"Scene {scene.scene_number} uses source attribution but "
                "does not mention a source supplied in evidence articles"
            )

        visual_matches = _find_pattern_matches(
            scene.visual_prompt,
            FORBIDDEN_VISUAL_PROMPT_PATTERNS,
        )

        if visual_matches:
            errors.append(
                f"Scene {scene.scene_number} visual_prompt requests forbidden "
                f"visual elements: {', '.join(visual_matches)}"
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

        normalized_prompt = _normalize_whitespace(
            scene.visual_prompt
        ).casefold()
        normalized_suffix = _normalize_whitespace(
            REQUIRED_VISUAL_PROMPT_SUFFIX
        ).casefold()

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

    for field_name, references in summary_lists.items():
        invalid_references = [
            reference
            for reference in references
            if reference not in allowed_references
        ]

        if invalid_references:
            errors.append(
                f"coverage_summary.{field_name} contains references not "
                f"present in editorial_topics: {invalid_references}"
            )

    if not all_scene_references:
        errors.append(
            "At least one editorial topic must be referenced by scenes"
        )

    if not script.fact_check_notes:
        errors.append("fact_check_notes must not be empty")

    if errors:
        raise MistralVideoValidationError(errors)


def parse_and_validate_mistral_video_script(
    raw_content: str,
    input_payload: dict[str, Any],
) -> MistralVideoScript:
    """Parse Mistral JSON and apply production validation."""
    script = parse_mistral_video_script(raw_content)

    script.narration_script.full_voiceover = _normalize_whitespace(
        " ".join(scene.narration for scene in script.scenes)
    )

    validate_mistral_video_script(script, input_payload)
    return script