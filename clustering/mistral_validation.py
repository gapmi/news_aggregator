from __future__ import annotations

from collections import Counter
import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MIN_SCENE_DURATION_SECONDS = 12
MAX_SCENE_DURATION_SECONDS = 50

MIN_VIDEO_DURATION_SECONDS = 60
MAX_VIDEO_DURATION_SECONDS = 600

MIN_SCENE_WORDS = 20
MAX_SCENE_WORDS = 110
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
    "data table": r"\bdata\s+tables?\b",
    "tabular layout": r"\btabular\s+(?:layout|display|format)\b",
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
    "firefighter": r"\bfirefighters?\b",
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

# Ordinary words such as run, score, core, edge and window remain allowed.
FORBIDDEN_NARRATION_TERMS = {
    "clustering run": r"\bclustering\s+runs?\b",
    "cluster id": r"\bcluster[_\s-]?ids?\b",
    "topic reference": r"\btopic[_\s-]?references?\b",
    "editorial topic identifier": r"\beditorial[_\s-]?topics?\b",
    "lineage": r"\blineage\b",
    "centroid": r"\bcentroids?\b",
    "payload": r"\bpayload\b",
    "subcluster": r"\bsubclusters?\b",
    "article count": r"\barticle[_\s-]?counts?\b",
    "parent size": r"\bparent[_\s-]?size\b",
    "child size": r"\bchild[_\s-]?size\b",
    "similarity score": r"\bsimilarity\s+scores?\b",
    "clustering threshold": r"\bclustering\s+thresholds?\b",
}

FORBIDDEN_FIRST_PERSON_PATTERNS = {
    "I": r"\bI\b",
    "we": r"\bwe\b",
    "our": r"\bour\b",
    "you": r"\byou\b",
    "your": r"\byour\b",
}

FORBIDDEN_MONITORING_PATTERNS = {
    "monitored coverage": r"\bmonitored\s+(?:coverage|publications?)\b",
    "monitoring visibility": r"\bnewly\s+visible\b",
    "publication volume": r"\bpublication[-\s]+volume\b",
    "coverage change": (
        r"\bcoverage\s+(?:rose|increased|declined|decreased|"
        r"remained\s+stable|was\s+stable|grew|fell)\b"
    ),
}

FORBIDDEN_NARRATION_FORMAT_PATTERNS = {
    "URL": r"https?://|www\.",
    "code fence": r"```",
    "citation marker": r"\[(?:web|file|cite):\d+\]",
    "Markdown heading": r"(?m)^\s*#{1,6}\s",
    "Markdown bold": r"\*\*",
    "template placeholder": r"\{\{|\}\}",
}


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Literal["en"]
    title: str = Field(min_length=5, max_length=160)
    description: str = Field(min_length=20, max_length=2000)
    estimated_duration_seconds: int = Field(
        ge=MIN_VIDEO_DURATION_SECONDS,
        le=MAX_VIDEO_DURATION_SECONDS,
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
    topic_references: list[str] = Field(min_length=1, max_length=1)


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
    normalized = _normalize_whitespace(value).casefold()

    negative_forms = (
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
    )

    for pattern in negative_forms:
        normalized = re.sub(pattern, "", normalized)

    return [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]


def _scene_word_bounds(duration_seconds: int) -> tuple[int, int]:
    maximum = min(
        MAX_SCENE_WORDS,
        round(duration_seconds * MAX_WORDS_PER_SECOND),
    )
    return MIN_SCENE_WORDS, maximum


def _topics_by_reference(
    input_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    topics = input_payload.get("editorial_topics")

    if not isinstance(topics, list) or not topics:
        raise ValueError("input_payload.editorial_topics must be non-empty")

    result: dict[str, dict[str, Any]] = {}

    for topic in topics:
        if not isinstance(topic, dict):
            raise ValueError("Each editorial topic must be an object")

        reference = topic.get("topic_reference")

        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Each topic needs a non-empty topic_reference")

        if reference in result:
            raise ValueError(f"Duplicate topic_reference: {reference}")

        result[reference] = topic

    return result


def _allowed_topic_references(
    input_payload: dict[str, Any],
) -> set[str]:
    return set(_topics_by_reference(input_payload))


def _normalize_source_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        char for char in decomposed
        if not unicodedata.combining(char)
    )
    return " ".join(
        "".join(
            char if char.isalnum() else " "
            for char in without_marks
        ).split()
    )


def _source_aliases(source_name: str) -> set[str]:
    if not isinstance(source_name, str):
        return set()

    normalized = _normalize_source_text(source_name)

    if not normalized:
        return set()

    aliases = {normalized}
    raw_source = source_name.strip().casefold()

    if " — " in source_name:
        publisher, _, _ = source_name.partition(" — ")
        publisher_alias = _normalize_source_text(publisher)
        if len(publisher_alias) >= 3:
            aliases.add(publisher_alias)

    if "." in raw_source and " " not in raw_source:
        hostname, _, _ = raw_source.partition("/")

        if hostname.startswith("www."):
            hostname = hostname[4:]

        normalized_hostname = _normalize_source_text(hostname)

        if normalized_hostname:
            aliases.add(normalized_hostname)

        first_label, _, _ = hostname.partition(".")
        normalized_first_label = _normalize_source_text(first_label)

        if len(normalized_first_label) >= 3:
            aliases.add(normalized_first_label)

    return {
        alias
        for alias in aliases
        if len(alias) >= 3
    }


def _allowed_source_names(
    input_payload: dict[str, Any],
) -> set[str]:
    sources: set[str] = set()

    for topic in input_payload.get("editorial_topics") or []:
        if not isinstance(topic, dict):
            continue

        for article in topic.get("evidence_articles") or []:
            if not isinstance(article, dict):
                continue

            aliases = article.get("source_aliases") or []
            if not isinstance(aliases, list):
                aliases = []

            for candidate in [article.get("source_name"), *aliases]:
                if isinstance(candidate, str) and candidate.strip():
                    sources.update(_source_aliases(candidate))

    return sources


def _source_names_mentioned(
    narration: str,
    allowed_sources: set[str],
) -> set[str]:
    normalized = f" {_normalize_source_text(narration)} "

    return {
        source
        for source in allowed_sources
        if f" {source} " in normalized
    }


def _expected_target_duration(input_payload: dict[str, Any]) -> int:
    requirements = input_payload.get("video_requirements")

    if not isinstance(requirements, dict):
        raise ValueError("video_requirements must be an object")

    duration = requirements.get("target_duration_seconds")

    if (
        not isinstance(duration, int)
        or isinstance(duration, bool)
        or not MIN_VIDEO_DURATION_SECONDS
        <= duration
        <= MAX_VIDEO_DURATION_SECONDS
    ):
        raise ValueError(
            "target_duration_seconds must be an integer between "
            f"{MIN_VIDEO_DURATION_SECONDS} and {MAX_VIDEO_DURATION_SECONDS}"
        )

    return duration


def _normalize_derived_fields(script: MistralVideoScript) -> None:
    ordered = sorted(
        script.scenes,
        key=lambda scene: scene.scene_number,
    )
    script.narration_script.full_voiceover = _normalize_whitespace(
        " ".join(scene.narration for scene in ordered)
    )
    script.video_metadata.estimated_duration_seconds = sum(
        scene.duration_seconds for scene in ordered
    )


def parse_mistral_video_script(
    raw_content: str,
) -> MistralVideoScript:
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

    # Do not silently drop invalid primitive entries from scenes.
    try:
        return MistralVideoScript.model_validate(parsed)
    except ValidationError as exc:
        errors = [
            "Schema error at "
            f"'{'.'.join(str(part) for part in error['loc'])}': "
            f"{error['msg']}"
            for error in exc.errors()
        ]
        raise MistralVideoValidationError(errors) from exc


def validate_mistral_video_script(
    script: MistralVideoScript,
    input_payload: dict[str, Any],
) -> None:
    errors: list[str] = []

    minimum_duration = _expected_target_duration(input_payload)
    topics = _topics_by_reference(input_payload)
    allowed_references = set(topics)
    scenes = script.scenes

    if len(scenes) != len(topics):
        errors.append(
            "Scene count must exactly equal the number of editorial topics: "
            f"expected={len(topics)}, actual={len(scenes)}"
        )

    numbers = [scene.scene_number for scene in scenes]
    expected_numbers = list(range(1, len(scenes) + 1))

    if numbers != expected_numbers:
        errors.append(
            "Scene numbers must start at 1 and increase without gaps: "
            f"expected={expected_numbers}, actual={numbers}"
        )

    duration_sum = sum(scene.duration_seconds for scene in scenes)

    if duration_sum < minimum_duration:
        errors.append(
            "Scene duration sum must not be shorter than the requested "
            "minimum duration: "
            f"expected_at_least={minimum_duration}, actual={duration_sum}"
        )

    if not MIN_VIDEO_DURATION_SECONDS <= duration_sum <= MAX_VIDEO_DURATION_SECONDS:
        errors.append(
            "Scene duration sum must be within the allowed video duration "
            f"range: expected_between={MIN_VIDEO_DURATION_SECONDS}.."
            f"{MAX_VIDEO_DURATION_SECONDS}, actual={duration_sum}"
        )

    if script.video_metadata.estimated_duration_seconds != duration_sum:
        errors.append(
            "video_metadata.estimated_duration_seconds must equal the sum "
            f"of scene durations: expected={duration_sum}, "
            f"actual={script.video_metadata.estimated_duration_seconds}"
        )

    expected_voiceover = _normalize_whitespace(
        " ".join(scene.narration for scene in scenes)
    )

    if _normalize_whitespace(
        script.narration_script.full_voiceover
    ) != expected_voiceover:
        errors.append(
            "narration_script.full_voiceover must exactly equal the "
            "concatenation of scenes[].narration in scene-number order"
        )

    total_words = _word_count(script.narration_script.full_voiceover)
    minimum_words = len(scenes) * MIN_SCENE_WORDS
    maximum_words = round(duration_sum * MAX_WORDS_PER_SECOND)

    if total_words < minimum_words:
        errors.append(
            "full_voiceover is too short for the number of scenes: "
            f"expected_at_least={minimum_words} words, actual={total_words}"
        )

    if total_words > maximum_words:
        errors.append(
            "full_voiceover is too long for the total scene duration: "
            f"expected_at_most={maximum_words} words, actual={total_words}"
        )

    public_texts = [
        script.narration_script.full_voiceover,
        script.video_metadata.title,
        script.video_metadata.description,
        script.video_metadata.editorial_angle,
        *(scene.narration for scene in scenes),
    ]

    for index, text in enumerate(public_texts):
        for label, patterns in (
            ("Forbidden internal narration terms", FORBIDDEN_NARRATION_TERMS),
            ("First-person or direct-address terms", FORBIDDEN_FIRST_PERSON_PATTERNS),
        ):
            matches = _find_pattern_matches(text, patterns)
            if matches:
                errors.append(
                    f"{label} found in public_text[{index}]: "
                    + ", ".join(matches)
                )

    reference_counts: Counter[str] = Counter()
    forbid_monitoring = bool(
        (input_payload.get("constraints") or {}).get(
            "do_not_narrate_monitored_publication_volume"
        )
    )

    for scene in scenes:
        prefix = f"Scene {scene.scene_number}"
        count = _word_count(scene.narration)
        minimum, maximum = _scene_word_bounds(scene.duration_seconds)

        if count < minimum:
            errors.append(
                f"{prefix} narration is too short: "
                f"expected_at_least={minimum} words for "
                f"{scene.duration_seconds} seconds, actual={count}"
            )

        if count > maximum:
            errors.append(
                f"{prefix} narration is too long: "
                f"expected_at_most={maximum} words for "
                f"{scene.duration_seconds} seconds, actual={count}"
            )

        reference_counts.update(scene.topic_references)

        invalid = [
            reference for reference in scene.topic_references
            if reference not in allowed_references
        ]

        if invalid:
            errors.append(
                f"{prefix} has topic_references not found "
                f"in editorial_topics: {invalid}"
            )
        else:
            topic = topics[scene.topic_references[0]]
            sources = _allowed_source_names({
                "editorial_topics": [topic]
            })

            if not _source_names_mentioned(scene.narration, sources):
                errors.append(
                    f"{prefix} must mention a source supplied "
                    "in evidence articles for its own topic"
                )

        format_matches = _find_pattern_matches(
            scene.narration,
            FORBIDDEN_NARRATION_FORMAT_PATTERNS,
        )
        if format_matches:
            errors.append(
                f"{prefix} narration contains forbidden formatting: "
                + ", ".join(format_matches)
            )

        if forbid_monitoring:
            matches = _find_pattern_matches(
                scene.narration,
                FORBIDDEN_MONITORING_PATTERNS,
            )
            if matches:
                errors.append(
                    f"{prefix} narration contains monitoring statements: "
                    + ", ".join(matches)
                )

        for label, patterns in (
            ("forbidden visual elements", FORBIDDEN_VISUAL_PROMPT_PATTERNS),
            ("unsafe fake-documentary content", FORBIDDEN_FAKE_DOCUMENTARY_PATTERNS),
        ):
            matches = _find_pattern_matches(
                scene.visual_prompt, patterns
            )
            if matches:
                errors.append(
                    f"{prefix} visual_prompt requests {label}: "
                    + ", ".join(matches)
                )

        if not _normalize_whitespace(
            scene.visual_prompt
        ).casefold().endswith(
            _normalize_whitespace(
                REQUIRED_VISUAL_PROMPT_SUFFIX
            ).casefold()
        ):
            errors.append(
                f"{prefix} visual_prompt must end with the "
                "required safe editorial style suffix"
            )

    for reference in sorted(allowed_references):
        count = reference_counts[reference]
        if count != 1:
            errors.append(
                "Every editorial topic must be referenced exactly once: "
                f"topic_reference={reference!r}, actual={count}"
            )

    summary_lists = {
        "continuing_topics": script.coverage_summary.continuing_topics,
        "newly_visible_topics": script.coverage_summary.newly_visible_topics,
        "reframed_topics": script.coverage_summary.reframed_topics,
        "declining_topics": script.coverage_summary.declining_topics,
    }

    for name, references in summary_lists.items():
        invalid = [
            reference for reference in references
            if reference not in allowed_references
        ]
        if invalid:
            errors.append(
                f"coverage_summary.{name} contains references not "
                f"present in editorial_topics: {invalid}"
            )

    if errors:
        raise MistralVideoValidationError(errors)


def parse_and_validate_mistral_video_script(
    raw_content: str,
    input_payload: dict[str, Any],
) -> MistralVideoScript:
    script = parse_mistral_video_script(raw_content)
    _normalize_derived_fields(script)
    validate_mistral_video_script(script, input_payload)
    return script