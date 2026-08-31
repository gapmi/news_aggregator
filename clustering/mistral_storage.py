from __future__ import annotations

import json
from typing import Any

from psycopg2.extras import Json, execute_values

from clustering.mistral_client import MistralCompletionResult
from clustering.mistral_validation import MistralVideoScript


def _script_to_dict(script: MistralVideoScript) -> dict[str, Any]:
    return script.model_dump(mode="json")


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_text(usage: dict[str, Any], key: str) -> str | None:
    value = usage.get(key)

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def save_mistral_video_script(
    conn,
    *,
    run_id: int,
    request_body: dict[str, Any],
    input_payload: dict[str, Any],
    result: MistralCompletionResult,
    script: MistralVideoScript,
) -> None:
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("run_id must be a positive integer")

    script_dict = _script_to_dict(script)
    video_metadata = script_dict["video_metadata"]
    narration_script = script_dict["narration_script"]
    coverage_summary = script_dict["coverage_summary"]
    fact_check_notes = script_dict["fact_check_notes"]
    scenes = script_dict["scenes"]

    raw_response_json = (
        result.raw_response if isinstance(result.raw_response, dict) else None
    )

    validation_errors: list[str] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.mistral_video_scripts (
                run_id,
                status,
                model,
                request_id,
                finish_reason,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                service_tier,
                title,
                description,
                language,
                estimated_duration_seconds,
                editorial_angle,
                full_voiceover,
                coverage_summary,
                fact_check_notes,
                script_json,
                raw_response_text,
                raw_response_json,
                validation_status,
                validation_errors,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (run_id) DO UPDATE
            SET
                status = EXCLUDED.status,
                model = EXCLUDED.model,
                request_id = EXCLUDED.request_id,
                finish_reason = EXCLUDED.finish_reason,
                latency_ms = EXCLUDED.latency_ms,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens,
                total_tokens = EXCLUDED.total_tokens,
                service_tier = EXCLUDED.service_tier,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                language = EXCLUDED.language,
                estimated_duration_seconds = EXCLUDED.estimated_duration_seconds,
                editorial_angle = EXCLUDED.editorial_angle,
                full_voiceover = EXCLUDED.full_voiceover,
                coverage_summary = EXCLUDED.coverage_summary,
                fact_check_notes = EXCLUDED.fact_check_notes,
                script_json = EXCLUDED.script_json,
                raw_response_text = EXCLUDED.raw_response_text,
                raw_response_json = EXCLUDED.raw_response_json,
                validation_status = EXCLUDED.validation_status,
                validation_errors = EXCLUDED.validation_errors,
                updated_at = NOW()
            """,
            (
                run_id,
                "success",
                result.model,
                result.request_id,
                result.finish_reason,
                result.latency_ms,
                _usage_int(result.usage, "prompt_tokens"),
                _usage_int(result.usage, "completion_tokens"),
                _usage_int(result.usage, "total_tokens"),
                _usage_text(result.usage, "service_tier"),
                video_metadata["title"],
                video_metadata["description"],
                video_metadata["language"],
                video_metadata["estimated_duration_seconds"],
                video_metadata["editorial_angle"],
                narration_script["full_voiceover"],
                Json(coverage_summary),
                Json(fact_check_notes),
                Json(script_dict),
                result.content,
                Json(raw_response_json) if raw_response_json is not None else None,
                "passed",
                Json(validation_errors),
            ),
        )

        cur.execute(
            """
            DELETE FROM public.mistral_video_script_scenes
            WHERE run_id = %s
            """,
            (run_id,),
        )

        scene_rows = [
            (
                run_id,
                int(scene["scene_number"]),
                int(scene["duration_seconds"]),
                scene["narration"],
                scene["visual_type"],
                scene["visual_prompt"],
                Json(scene["topic_references"]),
            )
            for scene in scenes
        ]

        if scene_rows:
            execute_values(
                cur,
                """
                INSERT INTO public.mistral_video_script_scenes (
                    run_id,
                    scene_number,
                    duration_seconds,
                    narration,
                    visual_type,
                    visual_prompt,
                    topic_references
                )
                VALUES %s
                """,
                scene_rows,
            )

            execute_values(
                cur,
                """
                INSERT INTO public.mistral_video_script_scenes (
                    run_id,
                    scene_number,
                    duration_seconds,
                    narration,
                    visual_type,
                    visual_prompt,
                    topic_references
                )
                VALUES %s
                """,
                scene_rows,
            )


def save_mistral_failure(
    conn,
    *,
    run_id: int,
    request_body: dict[str, Any] | None,
    input_payload: dict[str, Any] | None,
    result: MistralCompletionResult | None,
    status: str,
    validation_errors: list[str] | None = None,
    raw_response_text: str | None = None,
) -> None:
    if status not in {"api_failed", "validation_failed"}:
        raise ValueError("status must be 'api_failed' or 'validation_failed'")

    usage = result.usage if result is not None else {}
    raw_response_json = (
        result.raw_response
        if result is not None and isinstance(result.raw_response, dict)
        else None
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.mistral_video_scripts (
                run_id,
                status,
                model,
                request_id,
                finish_reason,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                service_tier,
                raw_response_text,
                raw_response_json,
                validation_status,
                validation_errors,
                script_json,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                NOW(), NOW()
            )
            ON CONFLICT (run_id) DO UPDATE
            SET
                status = EXCLUDED.status,
                model = EXCLUDED.model,
                request_id = EXCLUDED.request_id,
                finish_reason = EXCLUDED.finish_reason,
                latency_ms = EXCLUDED.latency_ms,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens,
                total_tokens = EXCLUDED.total_tokens,
                service_tier = EXCLUDED.service_tier,
                raw_response_text = EXCLUDED.raw_response_text,
                raw_response_json = EXCLUDED.raw_response_json,
                validation_status = EXCLUDED.validation_status,
                validation_errors = EXCLUDED.validation_errors,
                updated_at = NOW()
            """,
            (
                run_id,
                status,
                result.model if result is not None else None,
                result.request_id if result is not None else None,
                result.finish_reason if result is not None else None,
                result.latency_ms if result is not None else None,
                _usage_int(usage, "prompt_tokens"),
                _usage_int(usage, "completion_tokens"),
                _usage_int(usage, "total_tokens"),
                _usage_text(usage, "service_tier"),
                raw_response_text
                if raw_response_text is not None
                else (result.content if result is not None else None),
                Json(raw_response_json) if raw_response_json is not None else None,
                "failed" if status == "validation_failed" else "skipped",
                Json(validation_errors or []),
                Json({}),
            ),
        )

        cur.execute(
            """
            DELETE FROM public.mistral_video_script_scenes
            WHERE run_id = %s
            """,
            (run_id,),
        )