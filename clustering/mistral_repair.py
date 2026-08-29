from __future__ import annotations

import copy
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
    """Raised when a repair request or repair response cannot be processed."""


def _json_for_prompt(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )


def _extract_response_content(response: Mapping[str, Any]) -> str:
    """
    Extract assistant text from a non-streaming Mistral chat completion.

    Supports the usual response shape:
        choices[0].message.content

    It also accepts choices[0].messages[0].content for integrations that
    return the multi-message variant.
    """
    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise MistralRepairError(
            "Mistral repair response has no non-empty 'choices' array."
        )

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise MistralRepairError(
            "Mistral repair response choice is not an object."
        )

    message = first_choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    messages = first_choice.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, Mapping):
                continue

            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, list):
                text_parts: list[str] = []

                for chunk in content:
                    if not isinstance(chunk, Mapping):
                        continue

                    if chunk.get("type") == "text":
                        text = chunk.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)

                joined = "".join(text_parts).strip()
                if joined:
                    return joined

    raise MistralRepairError(
        "Could not extract assistant content from Mistral repair response."
    )


def build_mistral_repair_request(
    *,
    input_payload: Mapping[str, Any],
    invalid_raw_content: str,
    validation_errors: list[str],
    model: str,
    max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
    temperature: float = DEFAULT_REPAIR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Build a single JSON-mode Mistral request that repairs a previously
    generated but invalid video-script JSON document.

    The request intentionally repeats the authoritative source payload and
    instructs the model not to introduce facts not present in that payload.
    """
    if not isinstance(input_payload, Mapping):
        raise MistralRepairError("input_payload must be an object.")

    if not isinstance(invalid_raw_content, str) or not invalid_raw_content.strip():
        raise MistralRepairError(
            "invalid_raw_content must be a non-empty JSON string."
        )

    if not validation_errors:
        raise MistralRepairError(
            "validation_errors must be non-empty before repair is requested."
        )

    if not isinstance(model, str) or not model.strip():
        raise MistralRepairError("model must be a non-empty string.")

    if max_tokens < 512:
        raise MistralRepairError("max_tokens must be at least 512.")

    if temperature < 0:
        raise MistralRepairError("temperature must be non-negative.")

    system_prompt = """
You repair JSON video scripts for a production pipeline.

Return exactly one valid JSON object and nothing else:
- no Markdown fences;
- no explanations;
- no preamble;
- no trailing text.

The authoritative facts are only those in INPUT_PAYLOAD.
Do not add, infer, fabricate, or strengthen facts, dates, numbers, locations,
entities, quotations, claims, or causal statements beyond INPUT_PAYLOAD.

The previous script is structurally close but failed deterministic validation.
Repair every listed validation error while preserving the existing JSON contract,
field names, object nesting, scene count, scene numbering, topic references,
language, and target duration.

For narration:
- Make every scene narration long enough for its own duration.
- Make narration natural spoken voiceover, not notes or instructions.
- Use only facts supported by INPUT_PAYLOAD.
- Keep a coherent story across all scenes.
- narration_script.full_voiceover must be EXACTLY the scene narrations
  concatenated in ascending scene-number order, separated by exactly one space.
- Do not summarize or paraphrase the scene narrations in full_voiceover.

For visual_prompt:
- Describe only non-textual cinematic visuals.
- Never request readable text, labels, captions, subtitles, headlines, logos,
  watermarks, charts, graphs, tables, infographics, documents, social posts,
  screens displaying information, signage, or tickers.
- Do not include a negative safety suffix such as “no text” or “no logos”.
  The downstream system handles that separately.
- Prefer people, places, abstract motion, maps without labels, atmospheric
  newsroom imagery, objects, or symbolic non-textual visual metaphors.

Do not change confirmed topic references unless a validation error explicitly
requires it. Repair all listed errors in one pass.
""".strip()

    user_prompt = "\n\n".join(
        [
            "INPUT_PAYLOAD:",
            _json_for_prompt(dict(input_payload)),
            "PREVIOUS_INVALID_SCRIPT_JSON:",
            invalid_raw_content.strip(),
            "VALIDATION_ERRORS:",
            _json_for_prompt(validation_errors),
            "Return the repaired JSON object only.",
        ]
    )

    return {
        "model": model.strip(),
        "temperature": temperature,
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
    Extract the repaired JSON text from an API response and pass it through
    the existing authoritative validator.

    Returns the parsed, validated script dictionary. Raises
    MistralVideoValidationError if repaired content is still invalid.
    """
    raw_content = _extract_response_content(response)

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
    max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
    temperature: float = DEFAULT_REPAIR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Execute exactly one repair request and validate its result.

    The caller decides whether repair is allowed and is responsible for
    persisting request/response audit data. This function does not retry.
    """
    request_body = build_mistral_repair_request(
        input_payload=input_payload,
        invalid_raw_content=invalid_raw_content,
        validation_errors=validation_errors,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response = call_mistral(copy.deepcopy(request_body))

    if not isinstance(response, Mapping):
        raise MistralRepairError(
            "call_mistral returned a non-object repair response."
        )

    return validate_mistral_repair_response(
        response=response,
        input_payload=input_payload,
    )

#Repair prompt будет требовать:

##увеличить narration до 252–336 слов;

##переписать scene narrations под их duration;

##сделать full_voiceover точной конкатенацией всех сцен;

##убрать screen displaying, label, charts, text overlays и infographic;

##не включать слабую тему Que, Debate в spoken scenes;

##оставить только безопасные abstract/map/newsroom visuals;

##не добавлять новых фактов;

##сохранить ровно тот же JSON contract.