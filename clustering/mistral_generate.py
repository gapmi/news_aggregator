from __future__ import annotations

import argparse
import json
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
        default=6,
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
            script = parse_and_validate_mistral_video_script(
                raw_content=result.content,
                input_payload=input_payload,
            )

        except MistralVideoValidationError as initial_exc:
            conn.rollback()

            save_mistral_failure(
                conn,
                run_id=args.run_id,
                request_body=request_body,
                input_payload=input_payload,
                result=result,
                status="validation_failed",
                validation_errors=initial_exc.errors,
                raw_response_text=result.content,
            )
            conn.commit()

            print()
            print("validation=failed")
            print("repair_attempt=starting")

            for error in initial_exc.errors:
                print(f"- {error}")

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

                script = parse_and_validate_mistral_video_script(
                    raw_content=repair_result.content,
                    input_payload=input_payload,
                )

            except MistralClientError as repair_api_exc:
                conn.rollback()

                save_mistral_failure(
                    conn,
                    run_id=args.run_id,
                    request_body=repair_request,
                    input_payload=input_payload,
                    result=None,
                    status="api_failed",
                    validation_errors=[
                        "Repair API request failed: "
                        f"{type(repair_api_exc).__name__}: {repair_api_exc}"
                    ],
                    raw_response_text=None,
                )
                conn.commit()

                print(
                    "ERROR: Mistral repair request failed: "
                    f"{type(repair_api_exc).__name__}: {repair_api_exc}",
                    file=sys.stderr,
                )
                return 2

            except MistralVideoValidationError as repair_validation_exc:
                conn.rollback()

                save_mistral_failure(
                    conn,
                    run_id=args.run_id,
                    request_body=repair_request,
                    input_payload=input_payload,
                    result=repair_result,
                    status="validation_failed",
                    validation_errors=repair_validation_exc.errors,
                    raw_response_text=repair_result.content,
                )
                conn.commit()

                print()
                print("repair_validation=failed")

                for error in repair_validation_exc.errors:
                    print(f"- {error}")

                return 2

            save_mistral_video_script(
                conn,
                run_id=args.run_id,
                request_body=repair_request,
                input_payload=input_payload,
                result=repair_result,
                script=script,
            )
            conn.commit()

            print()
            print("repair_validation=passed")
            print(f"saved_run_id={args.run_id}")
            print(f"saved_scene_count={len(script.scenes)}")
            print(f"title={script.video_metadata.title}")

            return 0

        save_mistral_video_script(
            conn,
            run_id=args.run_id,
            request_body=request_body,
            input_payload=input_payload,
            result=result,
            script=script,
        )
        conn.commit()

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