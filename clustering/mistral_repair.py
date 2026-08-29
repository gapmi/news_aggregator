from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from clustering.mistral_validation import (
    MistralVideoValidationError,
    MistralVideoScript,
    parse_and_validate_mistral_video_script,
)

DEFAULT_REPAIR_MAX_TOKENS = 4_096
DEFAULT_REPAIR_TEMPERATURE = 0.1
MINIMUM_WORDS_PER_SECOND = 2.1


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
    """
    Ensure that raw_content contains one valid JSON object.

    Repair embeds a previous model response into the next prompt. Reject non-JSON
    strings so HTML errors, Markdown, and unrelated text cannot become repair
    instructions.
    """
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
    Return the same JSON Schema enforced by Pydantic validation.

    Supports both Pydantic v2 and v1. The project uses this only to make the
    contract explicit in the repair prompt; authoritative enforcement remains
    parse_and_validate_mistral_video_script().
    """
    model_json_schema = getattr(MistralVideoScript, "model_json_schema", None)

    if callable(model_json_schema):
        schema = model_json_schema()

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


def _get_narration_length_requirements(
    input_payload: Mapping[str, Any],
) -> dict[str, int | None]:
    """
    Derive the narration target from video_requirements.target_duration_seconds.

    The validator requires approximately 2.1 spoken words per second. Passing
    the computed value in the repair prompt prevents vague instructions such as
    "make narration long enough".
    """
    video_requirements = input_payload.get("video_requirements")

    if not isinstance(video_requirements, Mapping):
        return {
            "target_duration_seconds": None,
            "minimum_full_voiceover_words": None,
        }

    target_duration_seconds = video_requirements.get(
        "target_duration_seconds"
    )

    if (
        not isinstance(target_duration_seconds, int)
        or isinstance(target_duration_seconds, bool)
        or target_duration_seconds <= 0
    ):
        return {
            "target_duration_seconds": None,
            "minimum_full_voiceover_words": None,
        }

    return {
        "target_duration_seconds": target_duration_seconds,
        "minimum_full_voiceover_words": math.ceil(
            target_duration_seconds * MINIMUM_WORDS_PER_SECOND
        ),
    }


def extract_mistral_response_content(response: Any) -> str:
    """
    Extract assistant content from the project's client result or raw API JSON.

    Primary project-client result:
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

            chunk_text = chunk.get("text")

            if isinstance(chunk_text, str):
                text_parts.append(chunk_text)

        extracted_content = "".join(text_parts).strip()

        if extracted_content:
            return extracted_content

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
    Build one Mistral JSON-mode repair request.

    The request includes the full JSON Schema used by the validator. The schema
    makes fields such as scenes[].visual_type visibly required, rather than
    relying on the model to infer every field from an earlier response.
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
        input_payload
    )

    system_prompt = """
You repair JSON video scripts for a production pipeline.

Return exactly one complete JSON object and nothing else:
- no Markdown code fences;
- no explanations;
- no preamble;
- no trailing text;
- no patch, diff, or partial JSON.

OUTPUT_JSON_SCHEMA is mandatory. Return every required field in the schema.
Do not omit fields even if the previous invalid script omitted them.
In particular, every scene must include all required properties, including
scene_number, duration_seconds, narration, visual_prompt, visual_type, and
topic_references. Do not invent field names. Do not remove required fields.

The authoritative facts are only those in INPUT_PAYLOAD.
Do not add, infer, fabricate, strengthen, or alter facts, dates, numbers,
locations, entities, quotations, claims, conclusions, or causal statements
beyond INPUT_PAYLOAD.

The previous script failed deterministic validation.
Repair every listed validation error while preserving the JSON contract,
the required scene count, scene numbering and order, valid topic references,
input language, and target duration.

Narration requirements:
- Meet or exceed minimum_full_voiceover_words from
  NARRATION_LENGTH_REQUIREMENTS.
- Make every scenes[].narration long enough for that scene's stated duration.
- Write natural factual spoken voiceover, not instructions, notes, or outlines.
- Use only facts explicitly supported by INPUT_PAYLOAD.
- Maintain one coherent sequence across all scenes.
- narration_script.full_voiceover must EXACTLY equal all
  scenes[].narration strings in ascending scene-number order, with exactly one
  ASCII space between adjacent narration strings.
- Do not summarize, shorten, reorder, paraphrase, or independently rewrite
  narration_script.full_voiceover.

Topic-reference requirements:
- Every scene, including opening and closing scenes, must have at least one
  topic_references item.
- Never return topic_references: [].
- Every topic_references item must exactly match an allowed reference in
  INPUT_PAYLOAD.
- For a general opening or closing scene, reuse an allowed editorial topic
  reference rather than leaving topic_references empty.

Visual prompt requirements:
- Describe cinematic, non-textual footage only.
- Write camera-visible subjects, settings, lighting, motion, and composition.
- Never describe information diagrams or data representations.
- Never request readable text, labels, captions, subtitles, headlines, logos,
  watermarks, charts, graphs, tables, infographics, documents, social-media
  posts, user interfaces, dashboards, data visualization, trend lines, nodes,
  connecting lines, diagrams, screen displays, signage, or tickers.
- Never request screens, monitors, televisions, phones, devices, posters,
  newspapers, documents, boards, maps, or displays that show information.
- Do not include safety suffixes such as "no text", "no logo", "no logos",
  "no watermark", or "no watermarks"; downstream rendering handles them.
- Prefer real-world places, people, objects, weather, water, transit,
  architecture, empty editorial workspaces without monitors, abstract light,
  and unlabeled geographic terrain.

Repair every listed validation error in one pass.
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
    Extract repaired content and validate it with the authoritative validator.

    MistralVideoValidationError means the repair is still invalid. The caller
    should stop instead of sending a second automatic repair request.
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
    Execute exactly one repair call and validate the resulting script.

    For production audit logging, prefer calling build_mistral_repair_request(),
    call_mistral(), and validate_mistral_repair_response() separately, allowing
    storage of both request bodies, raw responses, and validation errors.

    This helper deliberately does not issue a second repair call.
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