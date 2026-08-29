from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from clustering.mistral_validation import (
    MistralVideoValidationError,
    parse_and_validate_mistral_video_script,
)

DEFAULT_REPAIR_MAX_TOKENS = 4_096
DEFAULT_REPAIR_TEMPERATURE = 0.1


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

    The original invalid model output is embedded in the repair prompt.
    Rejecting arbitrary plain text, HTML, or Markdown prevents prompt
    corruption and makes repair input deterministic.
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


def extract_mistral_response_content(response: Any) -> str:
    """
    Extract raw assistant content from either supported response shape.

    Primary project-client shape:
        MistralCompletionResult.content

    Raw Mistral HTTP JSON fallback:
        choices[0].message.content

    The returned content is not parsed or validated here. The caller must pass
    it to parse_and_validate_mistral_video_script().
    """
    result_content = getattr(response, "content", None)

    if isinstance(result_content, str) and result_content.strip():
        return result_content.strip()

    if not isinstance(response, Mapping):
        raise MistralRepairError(
            "Mistral response must be either a Mapping or an object with "
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
    Build a single Mistral JSON-mode request for repairing a video script.

    The request contains:
    - authoritative INPUT_PAYLOAD;
    - previous invalid JSON script;
    - deterministic validator error strings;
    - optional topic references prohibited from spoken narration.

    This function does not make an HTTP call.
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

    system_prompt = """
You repair JSON video scripts for a production pipeline.

Return exactly one valid JSON object and nothing else:
- no Markdown code fences;
- no explanations;
- no preamble;
- no trailing text.

The authoritative facts are only those in INPUT_PAYLOAD.
Do not add, infer, fabricate, strengthen, or alter facts, dates, numbers,
locations, entities, quotations, claims, conclusions, or causal statements
beyond INPUT_PAYLOAD.

The previous script failed deterministic validation.
Repair every listed validation error while preserving the established JSON
contract:
- retain all required top-level and nested field names;
- retain the same object nesting;
- retain the required scene count;
- retain scene numbering and required scene order;
- preserve valid topic references;
- preserve the input language;
- preserve the target duration.

Narration requirements:
- Make every scenes[].narration long enough for that scene's duration.
- Write natural factual spoken voiceover, not instructions, notes, or outlines.
- Use only facts explicitly supported by INPUT_PAYLOAD.
- Maintain one coherent sequence across all scenes.
- narration_script.full_voiceover must EXACTLY equal all
  scenes[].narration strings in ascending scene-number order, with exactly one
  ASCII space between adjacent narration strings.
- Do not summarize, shorten, reorder, paraphrase, or independently rewrite
  narration_script.full_voiceover.

Visual prompt requirements:
- Describe cinematic non-textual visuals only.
- Never request readable text, labels, captions, subtitles, headlines, logos,
  watermarks, charts, graphs, tables, infographics, documents, social-media
  posts, screens displaying information, signage, or tickers.
- Never request a screen, display, device, poster, newspaper, document, or
  board with readable information.
- Do not include safety suffixes such as "no text", "no logo", "no logos",
  "no watermark", or "no watermarks"; downstream rendering handles them.
- Prefer people, places, objects, natural movement, abstract movement,
  unlabeled maps, atmospheric newsroom imagery, and non-textual metaphors.

Fix every listed validation error in one pass.
Return the complete repaired JSON object, never a patch, diff, or partial JSON.
""".strip()

    user_prompt_parts = [
        "INPUT_PAYLOAD:",
        _json_for_prompt(dict(input_payload)),
        "PREVIOUS_INVALID_SCRIPT_JSON:",
        invalid_raw_content.strip(),
        "VALIDATION_ERRORS:",
        _json_for_prompt(validation_errors),
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

    user_prompt_parts.append("Return the repaired JSON object only.")

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
    Extract raw repair content and validate it against the video-script contract.

    Raises:
        MistralRepairError:
            If the Mistral response cannot be interpreted.

        MistralVideoValidationError:
            If the repair completion remains invalid. The caller must stop
            rather than submit a second repair request.
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
    Perform exactly one Mistral repair call and validate the result.

    `call_mistral` may return the project's MistralCompletionResult object
    or a raw Mistral response mapping. No retries are made here beyond any
    network retries already built into the provided call_mistral function.

    For production audit logging, use build_mistral_repair_request(),
    call_mistral(), and validate_mistral_repair_response() separately, so the
    initial and repair raw request/response artifacts can be saved.
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