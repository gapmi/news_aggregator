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
    Ensure a string contains a JSON object.

    The repaired script is generated in JSON mode, but checking the previous
    content before embedding it into the repair prompt prevents accidental
    prompt corruption caused by non-JSON API output.
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


def extract_mistral_response_content(response: Mapping[str, Any]) -> str:
    """
    Extract assistant content from a non-streaming Mistral chat completion.

    Expected standard response shape:
        choices[0].message.content

    Some adapters return content as a list of typed chunks; that format is
    supported too. The returned value is raw model output and is not parsed
    or validated here.
    """
    if not isinstance(response, Mapping):
        raise MistralRepairError("Mistral response must be an object.")

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

            chunk_type = chunk.get("type")
            chunk_text = chunk.get("text")

            if chunk_type == "text" and isinstance(chunk_text, str):
                text_parts.append(chunk_text)

        joined = "".join(text_parts).strip()

        if joined:
            return joined

    raise MistralRepairError(
        "Mistral response choices[0].message.content is empty "
        "or has an unsupported format."
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

    The repair request includes only:
    - the authoritative input payload;
    - the original invalid JSON script;
    - deterministic validator errors.

    It does not make an API call. The caller can save this request as an
    audit artifact before passing it to call_mistral().
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

    if not isinstance(max_tokens, int) or max_tokens < 512:
        raise MistralRepairError("max_tokens must be an integer of at least 512.")

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

    excluded_topics_instruction = ""

    if normalized_excluded_references:
        excluded_topics_instruction = (
            "\n\nTopics that must not appear in spoken narration or receive "
            "a dedicated scene:\n"
            f"{_json_for_prompt(normalized_excluded_references)}\n"
            "Do not mention, summarize, prioritize, or create spoken scenes "
            "for these excluded topics. Do not replace them with unsupported "
            "facts."
        )

    system_prompt = """
You repair JSON video scripts for a production pipeline.

Return exactly one valid JSON object and nothing else:
- no Markdown fences;
- no explanations;
- no preamble;
- no trailing text.

The authoritative facts are only those in INPUT_PAYLOAD.
Do not add, infer, fabricate, strengthen, or alter facts, dates, numbers,
locations, entities, quotations, claims, conclusions, or causal statements
beyond INPUT_PAYLOAD.

The previous script failed deterministic validation.
Repair every listed error while preserving the established JSON contract:
- keep the same top-level and nested field names;
- keep the same object nesting;
- keep the same scene count;
- keep scenes numbered in the required order;
- preserve valid topic references;
- preserve the input language;
- preserve the target duration.

Narration requirements:
- Make every scenes[].narration long enough for its own duration.
- Use natural, factual spoken voiceover rather than instructions or notes.
- Use only facts present in INPUT_PAYLOAD.
- Keep a coherent sequence across all scenes.
- narration_script.full_voiceover must EXACTLY equal all
  scenes[].narration values in ascending scene-number order, joined using
  exactly one ASCII space between adjacent scene narrations.
- Do not summarize, shorten, reorder, paraphrase, or independently rewrite
  scene narration in narration_script.full_voiceover.

Visual prompt requirements:
- Describe only cinematic, non-textual visuals.
- Never request readable text, labels, captions, subtitles, headlines, logos,
  watermarks, charts, graphs, tables, infographics, documents, social-media
  posts, screens displaying information, signage, or tickers.
- Do not request a screen, display, device, poster, newspaper, document, or
  board with readable information.
- Do not add phrases such as "no text", "no logo", "no logos", "no watermark",
  or "no watermarks"; the downstream renderer applies that safety suffix.
- Prefer people, places, objects, natural motion, abstract motion, unlabeled
  maps, atmospheric newsroom imagery, and non-textual visual metaphors.

Repair every listed validation error in one pass.
""".strip()

    user_prompt = "\n\n".join(
        [
            "INPUT_PAYLOAD:",
            _json_for_prompt(dict(input_payload)),
            "PREVIOUS_INVALID_SCRIPT_JSON:",
            invalid_raw_content.strip(),
            "VALIDATION_ERRORS:",
            _json_for_prompt(validation_errors),
            (
                "Return the complete repaired JSON object only. "
                "Do not return a patch, diff, or partial JSON."
            ),
        ]
    )

    if excluded_topics_instruction:
        user_prompt += excluded_topics_instruction

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
                "content": user_prompt,
            },
        ],
    }


def validate_mistral_repair_response(
    *,
    response: Mapping[str, Any],
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract repaired model output and apply the authoritative script validator.

    Returns the validated parsed script.

    Raises:
        MistralRepairError:
            If the Mistral API response cannot be read.

        MistralVideoValidationError:
            If the repaired script is still invalid. The caller should stop
            after this error instead of attempting a second repair.
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
    call_mistral: Callable[[dict[str, Any]], Mapping[str, Any]],
    excluded_topic_references: list[str] | None = None,
    max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
    temperature: float = DEFAULT_REPAIR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Execute one repair request and validate its result.

    This function intentionally makes exactly one Mistral repair call.
    It does not retry after a failed repair validation.

    For production audit logging, prefer calling:
    - build_mistral_repair_request();
    - call_mistral();
    - validate_mistral_repair_response();

    separately, so that both raw requests and raw responses can be persisted.
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

    if not isinstance(response, Mapping):
        raise MistralRepairError(
            "call_mistral returned a non-object repair response."
        )

    return validate_mistral_repair_response(
        response=response,
        input_payload=input_payload,
    )