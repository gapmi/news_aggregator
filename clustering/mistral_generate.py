from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from clustering.mistral_client import (
    MistralClientError,
    MistralCompletionResult,
    call_mistral,
)
from clustering.mistral_payload import build_mistral_video_payload
from clustering.mistral_repair import build_mistral_repair_request
from clustering.mistral_request import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_mistral_request,
)
from clustering.mistral_storage import (
    save_mistral_failure,
    save_mistral_video_script,
)
from clustering.mistral_validation import (
    MistralVideoValidationError,
    parse_and_validate_mistral_video_script,
)
from clustering.offline import get_conn


_SCENE_TOO_LONG_ERROR_RE = re.compile(
    r"^Scene (?P<scene_number>\d+) narration is too long: "
    r"expected_at_most=(?P<maximum_words>\d+) words for "
    r"(?P<duration_seconds>\d+) seconds, actual=(?P<actual_words>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Mistral video payload, call Mistral, validate the JSON "
            "response, and save the result to PostgreSQL."
        )
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument(
        "--target-duration-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--headlines-per-topic",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MISTRAL_MODEL,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payload and request, but do not call Mistral or write DB.",
    )
    return parser.parse_args()


def _print_request(request_body: dict[str, Any]) -> None:
    encoded = json.dumps(
        request_body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    print("=" * 88)
    print("MISTRAL REQUEST")
    print("=" * 88)
    print(f"model={request_body['model']}")
    print(f"request_json_bytes={len(encoded)}")


def _print_response(result: MistralCompletionResult) -> None:
    print()
    print("=" * 88)
    print("MISTRAL RESPONSE")
    print("=" * 88)
    print(f"request_id={result.request_id}")
    print(f"model={result.model}")
    print(f"finish_reason={result.finish_reason}")
    print(f"latency_ms={result.latency_ms}")
    print(
        "usage="
        + json.dumps(result.usage, ensure_ascii=False, separators=(",", ":"))
    )
    print()
    print(result.content)


def _word_count_overflow_details(errors: list[str]) -> list[dict[str, int]] | None:
    """
    Return parsed scene limits only when every validation error is a
    scene-narration word-count overflow.

    Returning None prevents a targeted retry from running when a response has
    schema, factual-grounding, source-attribution, visual, duration, or any
    other production-validation failure.
    """
    if not errors:
        return None

    details: list[dict[str, int]] = []

    for error in errors:
        match = _SCENE_TOO_LONG_ERROR_RE.fullmatch(error)

        if match is None:
            return None

        details.append(
            {
                "scene_number": int(match.group("scene_number")),
                "maximum_words": int(match.group("maximum_words")),
                "actual_words": int(match.group("actual_words")),
                "duration_seconds": int(match.group("duration_seconds")),
            }
        )

    return details


def _build_word_limit_retry_request(
    *,
    input_payload: dict[str, Any],
    invalid_raw_content: str,
    validation_errors: list[str],
    overflow_details: list[dict[str, int]],
    model: str,
) -> dict[str, Any]:
    """
    Build a second and final repair request for word-count-only overflows.

    The existing repair builder supplies the payload, schema, prior JSON,
    validator errors, topic constraints, and normal narration limits. This
    wrapper adds a non-negotiable instruction to modify only narration in the
    named scenes and keep every other JSON field unchanged.
    """
    request_body = build_mistral_repair_request(
        input_payload=input_payload,
        invalid_raw_content=invalid_raw_content,
        validation_errors=validation_errors,
        model=model,
    )

    exact_limits = json.dumps(
        overflow_details,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_message = request_body["messages"][0]["content"]
    request_body["messages"][0]["content"] = (
        system_message
        + "\n\n"
        + "FINAL WORD-LIMIT RETRY:\n"
        + "This is the last automatic repair attempt. The only validation "
        + "errors are word-count overflows in the named scenes. Modify only "
        + "the narration field of those named scenes. Keep every other "
        + "field, including scene_number, duration_seconds, visual_type, "
        + "visual_prompt, topic_references, video_metadata, coverage_summary, "
        + "and fact_check_notes, exactly unchanged from "
        + "PREVIOUS_INVALID_SCRIPT_JSON.\n"
        + "For each named scene, shorten by removing only redundant wording "
        + "or repeated attribution while preserving source-grounded facts and "
        + "any source names already present. Do not add facts. Do not remove "
        + "or alter a topic_reference. Count words using ordinary word tokens "
        + "before returning. Never exceed the exact maximum_words below.\n"
        + f"EXACT_OVERFLOW_LIMITS={exact_limits}"
    )

    request_body["messages"].append(
        {
            "role": "user",
            "content": (
                "Apply only the FINAL WORD-LIMIT RETRY instructions. "
                "Return the complete repaired JSON object only."
            ),
        }
    )

    return request_body


def _save_validation_failure(
    *,
    conn: Any,
    run_id: int,
    request_body: dict[str, Any],
    input_payload: dict[str, Any],
    result: MistralCompletionResult,
    errors: list[str],
) -> None:
    conn.rollback()
    save_mistral_failure(
        conn,
        run_id=run_id,
        request_body=request_body,
        input_payload=input_payload,
        result=result,
        status="validation_failed",
        validation_errors=errors,
        raw_response_text=result.content,
    )
    conn.commit()


def _save_api_failure(
    *,
    conn: Any,
    run_id: int,
    request_body: dict[str, Any],
    input_payload: dict[str, Any],
    exc: MistralClientError,
) -> None:
    conn.rollback()
    save_mistral_failure(
        conn,
        run_id=run_id,
        request_body=request_body,
        input_payload=input_payload,
        result=None,
        status="api_failed",
        validation_errors=[
            "Repair API request failed: "
            f"{type(exc).__name__}: {exc}"
        ],
        raw_response_text=None,
    )
    conn.commit()


def _print_validation_errors(label: str, errors: list[str]) -> None:
    print()
    print(label)

    for error in errors:
        print(f"- {error}")


def parse_and_save_script(
    *,
    conn: Any,
    run_id: int,
    request_body: dict[str, Any],
    input_payload: dict[str, Any],
    result: MistralCompletionResult,
):
    script = parse_and_validate_mistral_video_script(
        raw_content=result.content,
        input_payload=input_payload,
    )
    save_mistral_video_script(
        conn,
        run_id=run_id,
        request_body=request_body,
        input_payload=input_payload,
        result=result,
        script=script,
    )
    conn.commit()
    return script


def main() -> int:
    args = parse_args()
    conn = get_conn()

    try:
        input_payload = build_mistral_video_payload(
            conn=conn,
            child_run_id=args.run_id,
            target_duration_seconds=args.target_duration_seconds,
            max_topics=args.max_topics,
            headlines_per_topic=args.headlines_per_topic,
        )

        editorial_topics = input_payload.get("editorial_topics") or []
        if not editorial_topics:
            raise ValueError(
                f"No editorial_topics were produced for run_id={args.run_id}"
            )

        request_body = build_mistral_request(
            input_payload=input_payload,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        _print_request(request_body)

        if args.dry_run:
            print("dry_run=true")
            print(f"editorial_topics={len(editorial_topics)}")
            print(
                json.dumps(
                    input_payload,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        print("Sending one request to Mistral...")
        result = call_mistral(request_body)
        _print_response(result)

        try:
            script = parse_and_save_script(
                conn=conn,
                run_id=args.run_id,
                request_body=request_body,
                input_payload=input_payload,
                result=result,
            )
        except MistralVideoValidationError as initial_exc:
            _save_validation_failure(
                conn=conn,
                run_id=args.run_id,
                request_body=request_body,
                input_payload=input_payload,
                result=result,
                errors=initial_exc.errors,
            )

            _print_validation_errors(
                "validation=failed\nrepair_attempt=starting",
                initial_exc.errors,
            )

            repair_request = build_mistral_repair_request(
                input_payload=input_payload,
                invalid_raw_content=result.content,
                validation_errors=initial_exc.errors,
                model=args.model,
            )
            _print_request(repair_request)

            try:
                print("Sending one repair request to Mistral...")
                repair_result = call_mistral(repair_request)
                _print_response(repair_result)

                script = parse_and_save_script(
                    conn=conn,
                    run_id=args.run_id,
                    request_body=repair_request,
                    input_payload=input_payload,
                    result=repair_result,
                )
            except MistralClientError as repair_api_exc:
                _save_api_failure(
                    conn=conn,
                    run_id=args.run_id,
                    request_body=repair_request,
                    input_payload=input_payload,
                    exc=repair_api_exc,
                )
                print(
                    "ERROR: Mistral repair request failed: "
                    f"{type(repair_api_exc).__name__}: {repair_api_exc}",
                    file=sys.stderr,
                )
                return 2
            except MistralVideoValidationError as repair_validation_exc:
                overflow_details = _word_count_overflow_details(
                    repair_validation_exc.errors
                )

                if overflow_details is None:
                    _save_validation_failure(
                        conn=conn,
                        run_id=args.run_id,
                        request_body=repair_request,
                        input_payload=input_payload,
                        result=repair_result,
                        errors=repair_validation_exc.errors,
                    )
                    _print_validation_errors(
                        "repair_validation=failed",
                        repair_validation_exc.errors,
                    )
                    return 2

                _print_validation_errors(
                    "repair_validation=word_limit_only\n"
                    "word_limit_retry=starting",
                    repair_validation_exc.errors,
                )

                word_limit_retry_request = _build_word_limit_retry_request(
                    input_payload=input_payload,
                    invalid_raw_content=repair_result.content,
                    validation_errors=repair_validation_exc.errors,
                    overflow_details=overflow_details,
                    model=args.model,
                )
                _print_request(word_limit_retry_request)

                try:
                    print("Sending one final word-limit repair request to Mistral...")
                    word_limit_retry_result = call_mistral(
                        word_limit_retry_request
                    )
                    _print_response(word_limit_retry_result)

                    script = parse_and_save_script(
                        conn=conn,
                        run_id=args.run_id,
                        request_body=word_limit_retry_request,
                        input_payload=input_payload,
                        result=word_limit_retry_result,
                    )
                except MistralClientError as word_limit_api_exc:
                    _save_api_failure(
                        conn=conn,
                        run_id=args.run_id,
                        request_body=word_limit_retry_request,
                        input_payload=input_payload,
                        exc=word_limit_api_exc,
                    )
                    print(
                        "ERROR: Final word-limit repair request failed: "
                        f"{type(word_limit_api_exc).__name__}: "
                        f"{word_limit_api_exc}",
                        file=sys.stderr,
                    )
                    return 2
                except MistralVideoValidationError as word_limit_validation_exc:
                    _save_validation_failure(
                        conn=conn,
                        run_id=args.run_id,
                        request_body=word_limit_retry_request,
                        input_payload=input_payload,
                        result=word_limit_retry_result,
                        errors=word_limit_validation_exc.errors,
                    )
                    _print_validation_errors(
                        "word_limit_retry=failed",
                        word_limit_validation_exc.errors,
                    )
                    return 2

                print()
                print("word_limit_retry=passed")
                print(f"saved_run_id={args.run_id}")
                print(f"saved_scene_count={len(script.scenes)}")
                print(f"title={script.video_metadata.title}")
                return 0

            print()
            print("repair_validation=passed")
            print(f"saved_run_id={args.run_id}")
            print(f"saved_scene_count={len(script.scenes)}")
            print(f"title={script.video_metadata.title}")
            return 0

        print()
        print("validation=passed")
        print(f"saved_run_id={args.run_id}")
        print(f"saved_scene_count={len(script.scenes)}")
        print(f"title={script.video_metadata.title}")
        return 0

    except MistralClientError as exc:
        conn.rollback()
        save_mistral_failure(
            conn,
            run_id=args.run_id,
            request_body=None,
            input_payload=None,
            result=None,
            status="api_failed",
            validation_errors=[str(exc)],
            raw_response_text=None,
        )
        conn.commit()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())