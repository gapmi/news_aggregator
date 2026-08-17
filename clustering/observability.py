from __future__ import annotations

import json
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger("pipeline")


@dataclass
class PipelineContext:
    run_id: int | None
    stage: str
    attempt: int = 1
    window_from: str | None = None
    window_to: str | None = None
    article_count: int | None = None
    cluster_count: int | None = None
    node_count: int | None = None
    threshold: float | None = None
    model_version: str | None = None


class PipelineStageError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        error_type: str | None = None,
        retryable: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type or self.__class__.__name__
        self.retryable = retryable
        self.extra = extra or {}


def _base_event(level: str, ctx: PipelineContext, **fields: Any) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        **asdict(ctx),
        **fields,
    }


def emit_stage_event(level: str, ctx: PipelineContext, message: str, **fields: Any) -> None:
    payload = _base_event(level, ctx, message=message, **fields)
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        json.dumps(payload, ensure_ascii=False, default=str),
    )


def capture_pipeline_error(ctx: PipelineContext, exc: Exception, **fields: Any) -> None:
    current_trace = traceback.format_exc()
    if current_trace.strip() == "NoneType: None":
        current_trace = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)
        )

    if isinstance(exc, PipelineStageError):
        payload = _base_event(
            "ERROR",
            ctx,
            message=str(exc),
            error_code=exc.error_code,
            error_type=exc.error_type,
            retryable=exc.retryable,
            extra=exc.extra,
            stacktrace=current_trace,
            **fields,
        )
    else:
        payload = _base_event(
            "ERROR",
            ctx,
            message=str(exc),
            error_code="UNCLASSIFIED_ERROR",
            error_type=type(exc).__name__,
            retryable=False,
            extra={},
            stacktrace=current_trace,
            **fields,
        )

    logger.error(json.dumps(payload, ensure_ascii=False, default=str))