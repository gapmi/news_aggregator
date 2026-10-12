from __future__ import annotations

from typing import Any

from psycopg2.extras import Json, execute_values

from clustering.mistral_client import MistralCompletionResult
from clustering.mistral_validation import MistralVideoScript


_SCRIPT_COLUMNS = (
    "run_id",
    "status",
    "model",
    "request_id",
    "finish_reason",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "service_tier",
    "title",
    "description",
    "language",
    "estimated_duration_seconds",
    "editorial_angle",
    "full_voiceover",
    "coverage_summary",
    "fact_check_notes",
    "script_json",
    "raw_response_text",
    "raw_response_json",
    "validation_status",
    "validation_errors",
    "input_payload",
    "request_body",
)


def _validate_run_id(run_id: int) -> None:
    if (
        not isinstance(run_id, int)
        or isinstance(run_id, bool)
        or run_id <= 0
    ):
        raise ValueError("run_id must be a positive integer")


def _script_to_dict(script: MistralVideoScript) -> dict[str, Any]:
    return script.model_dump(mode="json")


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _usage_text(usage: dict[str, Any], key: str) -> str | None:
    value = usage.get(key)

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _optional_json(value: Any) -> Json | None:
    return Json(value) if value is not None else None


def _result_fields(
    result: MistralCompletionResult | None,
) -> dict[str, Any]:
    usage = (
        result.usage
        if result is not None and isinstance(result.usage, dict)
        else {}
    )

    raw_response = (
        result.raw_response
        if result is not None
        and isinstance(result.raw_response, dict)
        else None
    )

    return {
        "model": result.model if result is not None else None,
        "request_id": result.request_id if result is not None else None,
        "finish_reason": (
            result.finish_reason if result is not None else None
        ),
        "latency_ms": result.latency_ms if result is not None else None,
        "prompt_tokens": _usage_int(usage, "prompt_tokens"),
        "completion_tokens": _usage_int(usage, "completion_tokens"),
        "total_tokens": _usage_int(usage, "total_tokens"),
        "service_tier": _usage_text(usage, "service_tier"),
        "raw_response_json": _optional_json(raw_response),
    }


def _upsert_script(cur: Any, record: dict[str, Any]) -> None:
    columns_sql = ", ".join(_SCRIPT_COLUMNS)
    placeholders_sql = ", ".join(
        "%s" for _ in _SCRIPT_COLUMNS
    )
    updates_sql = ",\n".join(
        f"{column} = EXCLUDED.{column}"
        for column in _SCRIPT_COLUMNS
        if column != "run_id"
    )

    cur.execute(
        f"""
        INSERT INTO public.mistral_video_scripts (
            {columns_sql},
            created_at,
            updated_at
        )
        VALUES (
            {placeholders_sql},
            NOW(),
            NOW()
        )
        ON CONFLICT (run_id) DO UPDATE
        SET
            {updates_sql},
            updated_at = NOW()
        """,
        tuple(record[column] for column in _SCRIPT_COLUMNS),
    )


def _replace_scenes(
    cur: Any,
    *,
    run_id: int,
    scenes: list[dict[str, Any]],
) -> None:
    cur.execute(
        """
        DELETE FROM public.mistral_video_script_scenes
        WHERE run_id = %s
        """,
        (run_id,),
    )

    if not scenes:
        return

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


def save_mistral_video_script(
    conn,
    *,
    run_id: int,
    request_body: dict[str, Any],
    input_payload: dict[str, Any],
    result: MistralCompletionResult,
    script: MistralVideoScript,
) -> None:
    _validate_run_id(run_id)

    script_dict = _script_to_dict(script)
    metadata = script_dict["video_metadata"]

    record = {
        "run_id": run_id,
        "status": "success",
        **_result_fields(result),
        "title": metadata["title"],
        "description": metadata["description"],
        "language": metadata["language"],
        "estimated_duration_seconds": (
            metadata["estimated_duration_seconds"]
        ),
        "editorial_angle": metadata["editorial_angle"],
        "full_voiceover": (
            script_dict["narration_script"]["full_voiceover"]
        ),
        "coverage_summary": Json(script_dict["coverage_summary"]),
        "fact_check_notes": Json(script_dict["fact_check_notes"]),
        "script_json": Json(script_dict),
        "raw_response_text": result.content,
        "validation_status": "passed",
        "validation_errors": Json([]),
        "input_payload": Json(input_payload),
        "request_body": Json(request_body),
    }

    with conn.cursor() as cur:
        _upsert_script(cur, record)
        _replace_scenes(
            cur,
            run_id=run_id,
            scenes=script_dict["scenes"],
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
    _validate_run_id(run_id)

    if status not in {"api_failed", "validation_failed"}:
        raise ValueError(
            "status must be 'api_failed' or 'validation_failed'"
        )

    response_text = raw_response_text

    if response_text is None and result is not None:
        response_text = result.content

    model_from_request = (
        request_body.get("model")
        if isinstance(request_body, dict)
        else None
    )

    record = {
        "run_id": run_id,
        "status": status,
        **_result_fields(result),
        "title": None,
        "description": None,
        "language": None,
        "estimated_duration_seconds": None,
        "editorial_angle": None,
        "full_voiceover": None,
        "coverage_summary": None,
        "fact_check_notes": None,
        "script_json": Json({}),
        "raw_response_text": (
            response_text if response_text is not None else ""
        ),
        "validation_status": (
            "failed" if status == "validation_failed" else "skipped"
        ),
        "validation_errors": Json(validation_errors or []),
        "input_payload": _optional_json(input_payload),
        "request_body": _optional_json(request_body),
    }

    if record["model"] is None:
        record["model"] = model_from_request

    with conn.cursor() as cur:
        _upsert_script(cur, record)
        _replace_scenes(cur, run_id=run_id, scenes=[])