from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from clustering.mistral_validation import (
    MistralVideoScript,
    MistralVideoValidationError,
    parse_and_validate_mistral_video_script,
)

DEFAULT_REPAIR_MAX_TOKENS = 4_096
DEFAULT_REPAIR_TEMPERATURE = 0.1

MINIMUM_WORDS_PER_SECOND = 2.1
MAXIMUM_WORDS_PER_SECOND = 2.8
TARGET_WORDS_PER_SECOND = 2.35

REQUIRED_VISUAL_PROMPT_SUFFIX = (
    " No text, no logos, no watermark, no labels, no captions, no subtitles."
)

FORBIDDEN_INTERNAL_NARRATION_TERMS = (
    "cluster",
    "clusters",
    "topic_reference",
    "topic_references",
    "editorial_topic",
    "editorial_topics",
    "lineage",
)


class MistralRepairError(RuntimeError):
    """Raised when a Mistral repair request or response cannot be processed."""


def _json_for_prompt(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )


def _ensure_json_object(raw_content: str, *, field_name: str) -> None:
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise MistralRepairError(
            f"{field_name} must be a non-empty JSON object string."
        )

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise MistralRepairError(
            f"{field_name} is not valid JSON: {exc.msg}."
        ) from exc

    if not isinstance(parsed, dict):
        raise MistralRepairError(
            f"{field_name} must contain a JSON object, not "
            f"{type(parsed).__name__}."
        )


def _get_output_json_schema() -> dict[str, Any]:
    """
    Obtain the Pydantic schema used by the production validator.

    Supports Pydantic v2 first and Pydantic v1 as fallback. The schema is
    prompt context only; parse_and_validate_mistral_video_script remains the
    authoritative runtime check.
    """
    schema_method = getattr(MistralVideoScript, "model_json_schema", None)

    if callable(schema_method):
        schema = schema_method()

        if isinstance(schema, dict):
            return schema

    schema_method = getattr(MistralVideoScript, "schema", None)

    if callable(schema_method):
        schema = schema_method()

        if isinstance(schema, dict):
            return schema

    raise MistralRepairError(
        "Could not obtain JSON Schema from MistralVideoScript."
    )


def _get_scene_narration_requirements(
    invalid_raw_content: str,
) -> list[dict[str, int | None]]:
    """
    Read scene_number and duration_seconds from the prior script and calculate
    per-scene validator-style word bounds.

    The repair prompt uses this to prevent a valid total word count from being
    distributed incorrectly across short and long scenes.
    """
    previous_script = json.loads(invalid_raw_content)
    scenes = previous_script.get("scenes")

    if not isinstance(scenes, list):
        return []

    requirements: list[dict[str, int | None]] = []

    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue

        scene_number = scene.get("scene_number", index)
        duration_seconds = scene.get("duration_seconds")

        if (
            not isinstance(scene_number, int)
            or isinstance(scene_number, bool)
            or not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or duration_seconds <= 0
        ):
            requirements.append(
                {
                    "scene_number": (
                        scene_number
                        if isinstance(scene_number, int)
                        else index
                    ),
                    "duration_seconds": (
                        duration_seconds
                        if isinstance(duration_seconds, int)
                        else None
                    ),
                    "minimum_words": None,
                    "target_words": None,
                    "maximum_words": None,
                }
            )
            continue

        minimum_words = math.ceil(
            duration_seconds * MINIMUM_WORDS_PER_SECOND
        )
        target_words = math.ceil(
            duration_seconds * TARGET_WORDS_PER_SECOND
        )
        maximum_words = math.floor(
            duration_seconds * MAXIMUM_WORDS_PER_SECOND
        )

        requirements.append(
            {
                "scene_number": scene_number,
                "duration_seconds": duration_seconds,
                "minimum_words": minimum_words,
                "target_words": target_words,
                "maximum_words": maximum_words,
            }
        )

    return requirements


def _get_narration_length_requirements(
    input_payload: Mapping[str, Any],
    invalid_raw_content: str,
) -> dict[str, Any]:
    """
    Calculate total and scene-level narration limits visible to the model.

    The total target deliberately sits above the validator minimum. A model
    asked to generate exactly the minimum often under-runs after tokenization
    or phrasing differences.
    """
    video_requirements = input_payload.get("video_requirements")

    target_duration_seconds: int | None = None

    if isinstance(video_requirements, Mapping):
        candidate = video_requirements.get("target_duration_seconds")

        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            target_duration_seconds = candidate

    scene_requirements = _get_scene_narration_requirements(
        invalid_raw_content
    )

    minimum_full_voiceover_words: int | None = None
    target_full_voiceover_words: int | None = None
    maximum_full_voiceover_words: int | None = None

    if target_duration_seconds is not None:
        minimum_full_voiceover_words = math.ceil(
            target_duration_seconds * MINIMUM_WORDS_PER_SECOND
        )
        target_full_voiceover_words = math.ceil(
            target_duration_seconds * TARGET_WORDS_PER_SECOND
        )
        maximum_full_voiceover_words = math.floor(
            target_duration_seconds * MAXIMUM_WORDS_PER_SECOND
        )

    return {
        "target_duration_seconds": target_duration_seconds,
        "minimum_full_voiceover_words": minimum_full_voiceover_words,
        "target_full_voiceover_words": target_full_voiceover_words,
        "maximum_full_voiceover_words": maximum_full_voiceover_words,
        "scene_requirements": scene_requirements,
    }


def extract_mistral_response_content(response: Any) -> str:
    """
    Extract content from the project client result or raw API JSON.

    Primary project-client form:
        MistralCompletionResult.content

    Raw API JSON fallback:
        choices[0].message.content
    """
    result_content = getattr(response, "content", None)

    if isinstance(result_content, str) and result_content.strip():
        return result_content.strip()

    if not isinstance(response, Mapping):
        raise MistralRepairError(
            "Mistral response must be a Mapping or an object with "
            "a non-empty string .content attribute."
        )

    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise MistralRepairError(
            "Mistral response has no non-empty 'choices' array."
        )

    first_choice = choices[0]

    if not isinstance(first_choice, Mapping):
        raise MistralRepairError(
            "Mistral response choices[0] must be an object."
        )

    message = first_choice.get("message")

    if not isinstance(message, Mapping):
        raise MistralRepairError(
            "Mistral response choices[0].message must be an object."
        )

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for chunk in content:
            if not isinstance(chunk, Mapping):
                continue

            if chunk.get("type") != "text":
                continue

            text = chunk.get("text")

            if isinstance(text, str):
                text_parts.append(text)

        joined = "".join(text_parts).strip()

        if joined:
            return joined

    raise MistralRepairError(
        "Mistral response has neither a valid .content value nor a valid "
        "choices[0].message.content value."
    )


def build_mistral_repair_request(
    *,
    input_payload: Mapping[str, Any],
    invalid_raw_content: str,
    validation_errors: list[str],
    model: str,
    excluded_topic_references: list[str] | None = None,
    max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
    temperature: float = DEFAULT_REPAIR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Build one Mistral JSON-mode repair request without calling the API.

    The request carries the source payload, validation errors, Pydantic schema,
    explicit narration budgets, and the exact required visual prompt suffix.
    """
    if not isinstance(input_payload, Mapping):
        raise MistralRepairError("input_payload must be an object.")

    _ensure_json_object(
        invalid_raw_content,
        field_name="invalid_raw_content",
    )

    if (
        not isinstance(validation_errors, list)
        or not validation_errors
        or not all(
            isinstance(error, str) and error.strip()
            for error in validation_errors
        )
    ):
        raise MistralRepairError(
            "validation_errors must be a non-empty list of non-empty strings."
        )

    if not isinstance(model, str) or not model.strip():
        raise MistralRepairError("model must be a non-empty string.")

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise MistralRepairError("max_tokens must be an integer.")

    if max_tokens < 512:
        raise MistralRepairError(
            "max_tokens must be greater than or equal to 512."
        )

    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
        or temperature > 2
    ):
        raise MistralRepairError(
            "temperature must be a number between 0 and 2."
        )

    normalized_excluded_references: list[str] = []

    if excluded_topic_references is not None:
        if not isinstance(excluded_topic_references, list):
            raise MistralRepairError(
                "excluded_topic_references must be a list when provided."
            )

        for reference in excluded_topic_references:
            if not isinstance(reference, str) or not reference.strip():
                raise MistralRepairError(
                    "excluded_topic_references must contain only "
                    "non-empty strings."
                )

            normalized_excluded_references.append(reference.strip())

    output_json_schema = _get_output_json_schema()

    narration_length_requirements = _get_narration_length_requirements(
        input_payload=input_payload,
        invalid_raw_content=invalid_raw_content,
    )

    system_prompt = f"""
You repair JSON video scripts for a production pipeline.

Return exactly one complete JSON object and nothing else:
- no Markdown code fences;
- no explanations;
- no preamble;
- no trailing text;
- no patch, diff, or partial JSON.

OUTPUT_JSON_SCHEMA is mandatory. Return every required field in the schema.
Do not omit fields even if the previous script omitted them.
Every scene must include all required properties, including scene_number,
duration_seconds, narration, visual_prompt, visual_type, and topic_references.

The only authoritative facts are in INPUT_PAYLOAD.
Do not add, infer, fabricate, strengthen, or alter facts, dates, numbers,
locations, entities, quotations, claims, conclusions, or causal statements
beyond INPUT_PAYLOAD.

Repair every listed validation error.

Narration requirements:
- Follow NARRATION_LENGTH_REQUIREMENTS exactly.
- Each scene narration must be between its listed minimum_words and
  maximum_words inclusive. Aim near target_words.
- The complete full_voiceover must be between minimum_full_voiceover_words and
  maximum_full_voiceover_words inclusive. Aim near target_full_voiceover_words,
  not near the minimum.
- Do not use internal system terms in spoken narration: {", ".join(FORBIDDEN_INTERNAL_NARRATION_TERMS)}.
- Write natural factual spoken voiceover, not notes, instructions, or outlines.
- Use only facts explicitly supported by INPUT_PAYLOAD.
- Before producing JSON, write all scene narrations mentally in numeric order.
  Then set narration_script.full_voiceover to an exact character-for-character
  concatenation of those final narration strings, in ascending scene_number
  order, joined by exactly one ASCII space.
- Do not summarize, shorten, reorder, or independently rewrite full_voiceover.

Topic-reference requirements:
- Every scene, including opening and closing scenes, must contain from one to
  three topic_references.
- Never return topic_references: [].
- Every topic_references item must exactly match an allowed reference from
  INPUT_PAYLOAD.
- For a general opening or closing scene, reuse one or more allowed editorial
  topic references.

Visual prompt requirements:
- Describe cinematic, non-textual footage only.
- Write camera-visible subjects, settings, lighting, motion, and composition.
- Never request text, readable words, labels, captions, subtitles, headlines,
  logos, watermarks, charts, graphs, tables, infographics, documents,
  social-media posts, user interfaces, dashboards, data visualization, trend
  lines, nodes, connecting lines, diagrams, screen displays, signage, or
  tickers.
- Never request screens, monitors, televisions, phones, devices, posters,
  newspapers, documents, boards, maps, or displays showing information.
- Use real-world places, people, objects, weather, water, transit,
  architecture, empty editorial workspaces without monitors, abstract light,
  or unlabeled geographic terrain.
- Every visual_prompt must end with this exact suffix, including capitalization
  and punctuation:
{REQUIRED_VISUAL_PROMPT_SUFFIX}

Return the complete repaired JSON object only.
""".strip()

    user_prompt_parts = [
        "INPUT_PAYLOAD:",
        _json_for_prompt(dict(input_payload)),
        "OUTPUT_JSON_SCHEMA:",
        _json_for_prompt(output_json_schema),
        "PREVIOUS_INVALID_SCRIPT_JSON:",
        invalid_raw_content.strip(),
        "VALIDATION_ERRORS:",
        _json_for_prompt(validation_errors),
        "NARRATION_LENGTH_REQUIREMENTS:",
        _json_for_prompt(narration_length_requirements),
        "REQUIRED_VISUAL_PROMPT_SUFFIX:",
        REQUIRED_VISUAL_PROMPT_SUFFIX,
    ]

    if normalized_excluded_references:
        user_prompt_parts.extend(
            [
                "EXCLUDED_TOPIC_REFERENCES:",
                _json_for_prompt(normalized_excluded_references),
                (
                    "Do not mention, summarize, prioritize, or create a "
                    "dedicated spoken scene for excluded topic references. "
                    "Do not replace them with unsupported facts."
                ),
            ]
        )

    user_prompt_parts.append(
        "Return the complete repaired JSON object only."
    )

    return {
        "model": model.strip(),
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_object",
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(user_prompt_parts),
            },
        ],
    }


def validate_mistral_repair_response(
    *,
    response: Any,
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract repair content and run the authoritative validator.

    A MistralVideoValidationError means the repair remains invalid. The caller
    must stop instead of scheduling another automatic repair attempt.
    """
    if not isinstance(input_payload, Mapping):
        raise MistralRepairError("input_payload must be an object.")

    raw_content = extract_mistral_response_content(response)

    return parse_and_validate_mistral_video_script(
        raw_content=raw_content,
        input_payload=dict(input_payload),
    )


def repair_mistral_video_script(
    *,
    input_payload: Mapping[str, Any],
    invalid_raw_content: str,
    validation_errors: list[str],
    model: str,
    call_mistral: Callable[[dict[str, Any]], Any],
    excluded_topic_references: list[str] | None = None,
    max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
    temperature: float = DEFAULT_REPAIR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Perform one repair request and validate the returned script.

    This helper intentionally performs no second repair attempt. Production
    orchestration should save the generated request, raw result, and validation
    errors as audit artifacts before proceeding to TTS or visual generation.
    """
    if not callable(call_mistral):
        raise MistralRepairError("call_mistral must be callable.")

    request_body = build_mistral_repair_request(
        input_payload=input_payload,
        invalid_raw_content=invalid_raw_content,
        validation_errors=validation_errors,
        model=model,
        excluded_topic_references=excluded_topic_references,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response = call_mistral(request_body)

    return validate_mistral_repair_response(
        response=response,
        input_payload=input_payload,
    )