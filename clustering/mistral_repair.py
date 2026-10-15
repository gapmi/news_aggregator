from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from typing import Any

from clustering.mistral_request import build_mistral_request
from clustering.mistral_validation import (
    MAX_SCENE_DURATION_SECONDS,
    MAX_SCENE_WORDS,
    MAX_WORDS_PER_SECOND,
    MIN_SCENE_DURATION_SECONDS,
    MIN_SCENE_WORDS,
    MistralVideoScript,
    parse_and_validate_mistral_video_script,
)


DEFAULT_REPAIR_MAX_TOKENS = 4096
DEFAULT_REPAIR_TEMPERATURE = 0.1


class MistralRepairError(RuntimeError):
    """Raised when repair input or response cannot be processed."""


def _json_for_prompt(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _get_output_json_schema() -> dict[str, Any]:
    for method_name in ("model_json_schema", "schema"):
        method = getattr(MistralVideoScript, method_name, None)

        if callable(method):
            schema = method()

            if isinstance(schema, dict):
                return schema

    raise MistralRepairError(
        "Could not obtain JSON Schema from MistralVideoScript."
    )


def _prepare_payload(
    input_payload: Mapping[str, Any],
    excluded_topic_references: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(input_payload, Mapping):
        raise MistralRepairError("input_payload must be an object.")

    payload = deepcopy(dict(input_payload))

    if excluded_topic_references is None:
        return payload

    if not isinstance(excluded_topic_references, list):
        raise MistralRepairError(
            "excluded_topic_references must be a list."
        )

    excluded: set[str] = set()

    for reference in excluded_topic_references:
        if not isinstance(reference, str) or not reference.strip():
            raise MistralRepairError(
                "Excluded references must be non-empty strings."
            )
        excluded.add(reference.strip())

    topics = payload.get("editorial_topics")

    if not isinstance(topics, list):
        raise MistralRepairError("editorial_topics must be a list.")

    allowed = {
        topic.get("topic_reference")
        for topic in topics
        if isinstance(topic, dict)
        and isinstance(topic.get("topic_reference"), str)
    }

    unknown = excluded - allowed

    if unknown:
        raise MistralRepairError(
            f"Excluded references not present in payload: {sorted(unknown)}"
        )

    payload["editorial_topics"] = [
        topic
        for topic in topics
        if isinstance(topic, dict)
        and topic.get("topic_reference") not in excluded
    ]

    return payload


def _get_scene_narration_requirements(
    invalid_raw_content: str,
) -> list[dict[str, int]]:
    """
    Describe valid prior scene durations as guidance, not immutable limits.

    Malformed JSON is passed to the model for repair without guessed budgets.
    """
    try:
        previous = json.loads(invalid_raw_content)
    except json.JSONDecodeError:
        return []

    if not isinstance(previous, dict):
        return []

    scenes = previous.get("scenes")

    if not isinstance(scenes, list):
        return []

    requirements: list[dict[str, int]] = []

    for scene in scenes:
        if not isinstance(scene, dict):
            continue

        number = scene.get("scene_number")
        duration = scene.get("duration_seconds")

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
            or not isinstance(duration, int)
            or isinstance(duration, bool)
            or not MIN_SCENE_DURATION_SECONDS
            <= duration
            <= MAX_SCENE_DURATION_SECONDS
        ):
            continue

        requirements.append({
            "scene_number": number,
            "prior_duration_seconds": duration,
            "minimum_words": MIN_SCENE_WORDS,
            "maximum_words_at_prior_duration": min(
                MAX_SCENE_WORDS,
                round(duration * MAX_WORDS_PER_SECOND),
            ),
        })

    return requirements


def extract_mistral_response_content(response: Any) -> str:
    content = getattr(response, "content", None)

    if isinstance(content, str) and content.strip():
        return content.strip()

    if not isinstance(response, Mapping):
        raise MistralRepairError(
            "Response must provide content or a raw API response object."
        )

    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise MistralRepairError("Response has no choices.")

    first = choices[0]

    if not isinstance(first, Mapping):
        raise MistralRepairError("Response choices[0] must be an object.")

    message = first.get("message")

    if not isinstance(message, Mapping):
        raise MistralRepairError("Response message must be an object.")

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = [
            chunk["text"]
            for chunk in content
            if isinstance(chunk, Mapping)
            and chunk.get("type") == "text"
            and isinstance(chunk.get("text"), str)
        ]

        joined = "".join(parts).strip()

        if joined:
            return joined

    raise MistralRepairError("Response has no usable text content.")


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
    Build a repair request using the main request's evidence and policy.

    No API calls or database writes are performed here.
    """
    if (
        not isinstance(invalid_raw_content, str)
        or not invalid_raw_content.strip()
    ):
        raise MistralRepairError(
            "invalid_raw_content must be a non-empty string."
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
            "validation_errors must contain non-empty strings."
        )

    payload = _prepare_payload(
        input_payload,
        excluded_topic_references,
    )

    try:
        request = build_mistral_request(
            input_payload=payload,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (TypeError, ValueError) as exc:
        raise MistralRepairError(
            f"Cannot build repair request: {exc}"
        ) from exc

    repair_instructions = f"""
REPAIR MODE:

Repair the previous invalid script using the same evidence and narration
policy as the main generation request.

1. Return one complete JSON object, not a patch or partial response.
2. Correct all listed validation errors.
3. Do not treat the previous script as factual evidence. Unsupported
   statements in it must be removed or corrected using supplied evidence.
4. Preserve valid, relevant source-grounded content where possible.
5. Do not add facts, filler or monitoring statements to satisfy word limits.
6. Every allowed topic must occur in exactly one scene; every scene must
   contain exactly one allowed topic_reference.
7. Do not add standalone introductions, conclusions or transition scenes.
8. The narration minimum is {MIN_SCENE_WORDS} words per scene.
   There is no minimum words-per-second requirement.
9. The maximum is min({MAX_SCENE_WORDS},
   round(duration_seconds * {MAX_WORDS_PER_SECOND})) words per scene,
   using the final duration. Both limits apply.
10. Prior scene budgets are guidance only. When a duration changes,
    recompute its budget using the final duration. Keep exactly one
    scene per allowed editorial topic; do not merge separate topics.
11. Do not impose a total word budget based on the requested minimum duration.
    The total maximum depends on the final sum of scene durations.
12. Keep durations within the main contract. Do not change timing merely
    to conceal repetitive or unsupported narration.
13. full_voiceover must be constructed from the final scene narrations.
    estimated_duration_seconds must be their summed duration.
    Both fields may change when their underlying scene data changes.
14. If the previous response is malformed JSON, reconstruct the required
    structure without inventing missing facts.
15. Apply all main factual and visual restrictions during repair.
16. Do not claim external verification or guaranteed factual correctness.
17. Make the smallest changes needed to correct the listed errors
    and any unsupported claims identified from the supplied evidence.
    Do not rewrite or shorten otherwise valid scenes merely because
    another scene failed validation.
18. For a word-budget error, first increase the affected scene's duration
    within the allowed range and total video-duration limit.
    Shorten narration only when necessary, preserving material facts,
    attribution and connected spoken delivery.
19. For an attribution error, use an exact allowed publisher name
    supplied for that topic, retaining its original spelling.
    Do not translate it or replace it with an inferred publisher.
    Correct the attribution without unnecessarily shortening the story.
    A publisher name does not make unsupported claims valid.
""".strip()

    request["messages"][0]["content"] += (
        "\n\n" + repair_instructions
    )

    topic_sources = [
        {
            "topic_reference": topic["topic_reference"],
            "allowed_source_names": sorted({
                article["source_name"]
                for article in topic.get("evidence_articles") or []
                if isinstance(article, dict)
                and isinstance(article.get("source_name"), str)
                and article["source_name"].strip()
            }),
        }
        for topic in payload["editorial_topics"]
    ]

    #test str

    request["messages"].append({
        "role": "user",
        "content": "\n\n".join([
            "OUTPUT_JSON_SCHEMA:",
            _json_for_prompt(_get_output_json_schema()),
            "PREVIOUS_INVALID_SCRIPT_JSON:",
            invalid_raw_content.strip(),
            "VALIDATION_ERRORS:",
            _json_for_prompt(validation_errors),
            "REQUIRED_PUBLISHER_ATTRIBUTION_BY_TOPIC:",
            _json_for_prompt(topic_sources),
            (
                "Every scene must name a publisher allowed for its own topic. "
                "Use an exact name from "
                "REQUIRED_PUBLISHER_ATTRIBUTION_BY_TOPIC, retaining its "
                "original spelling. Keep each factual detail attributed "
                "to the source that supplied it. "
                "Write complete factual narration first; "
                f"choose a compatible duration from "
                f"{MIN_SCENE_DURATION_SECONDS} to "
                f"{MAX_SCENE_DURATION_SECONDS} seconds. "
                "Use about 2.3 words per second as a planning guide. "
                f"Each scene must contain at most {MAX_SCENE_WORDS} words "
                "and must also satisfy its duration-based word budget. "
                "The requested minimum is not an exact total. "
                "Increase the affected scene's duration before removing "
                "material supported details when the limits allow it. "
                f"Do not add facts or filler to reach "
                f"{MIN_SCENE_WORDS} words."
            ),
            "PRIOR_SCENE_WORD_BUDGETS:",
            _json_for_prompt(
                _get_scene_narration_requirements(invalid_raw_content)
            ),
            (
                "Return the complete repaired JSON object only. "
                "Use the source evidence and narration policy "
                "in the preceding INPUT_JSON."
            ),
        ]),
    })

    return request


def validate_mistral_repair_response(
    *,
    response: Any,
    input_payload: Mapping[str, Any],
) -> MistralVideoScript:
    if not isinstance(input_payload, Mapping):
        raise MistralRepairError("input_payload must be an object.")

    return parse_and_validate_mistral_video_script(
        raw_content=extract_mistral_response_content(response),
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
) -> MistralVideoScript:
    """Perform one repair call and validate against its effective payload."""
    if not callable(call_mistral):
        raise MistralRepairError("call_mistral must be callable.")

    payload = _prepare_payload(
        input_payload,
        excluded_topic_references,
    )

    request = build_mistral_repair_request(
        input_payload=payload,
        invalid_raw_content=invalid_raw_content,
        validation_errors=validation_errors,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response = call_mistral(request)

    return validate_mistral_repair_response(
        response=response,
        input_payload=payload,
    )