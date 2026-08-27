from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


class MistralClientError(RuntimeError):
    """Base error for Mistral API client failures."""


class MistralConfigurationError(MistralClientError):
    """Raised when required local client configuration is missing or invalid."""


class MistralRequestError(MistralClientError):
    """Raised when Mistral rejects the request with a non-retryable HTTP error."""


class MistralRateLimitError(MistralClientError):
    """Raised after retryable rate-limit responses are exhausted."""


class MistralServerError(MistralClientError):
    """Raised after retryable Mistral server errors are exhausted."""


@dataclass(frozen=True)
class MistralCompletionResult:
    """
    Normalized result of one Mistral chat completion request.

    raw_response is retained for debugging and later persistence in the DB.
    content is the raw assistant message content; in JSON mode it should be a
    JSON string that the next pipeline stage validates with Pydantic.
    """

    request_id: str | None
    model: str | None
    finish_reason: str | None
    content: str
    usage: dict[str, Any]
    latency_ms: int
    rate_limit_remaining: str | None
    raw_response: dict[str, Any]

    def content_as_json(self) -> dict[str, Any]:
        """
        Parse assistant content returned in JSON mode.

        This validates JSON syntax only. The Pydantic output schema should
        validate the returned object in the next stage.
        """
        try:
            value = json.loads(self.content)
        except json.JSONDecodeError as exc:
            raise MistralClientError(
                "Mistral response content is not valid JSON"
            ) from exc

        if not isinstance(value, dict):
            raise MistralClientError(
                "Mistral response content must be a JSON object"
            )

        return value


def _get_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()

    if not api_key:
        raise MistralConfigurationError(
            "MISTRAL_API_KEY is not set in the application environment"
        )

    return api_key


def _validate_request_body(request_body: dict[str, Any]) -> None:
    required_fields = {
        "model",
        "messages",
        "response_format",
    }

    missing = sorted(required_fields - request_body.keys())

    if missing:
        raise MistralConfigurationError(
            f"Mistral request body is missing required fields: {', '.join(missing)}"
        )

    if not isinstance(request_body["model"], str) or not request_body["model"].strip():
        raise MistralConfigurationError(
            "Mistral request body field 'model' must be a non-empty string"
        )

    if not isinstance(request_body["messages"], list) or not request_body["messages"]:
        raise MistralConfigurationError(
            "Mistral request body field 'messages' must be a non-empty list"
        )

    if not isinstance(request_body["response_format"], dict):
        raise MistralConfigurationError(
            "Mistral request body field 'response_format' must be an object"
        )

    if request_body["response_format"].get("type") != "json_object":
        raise MistralConfigurationError(
            "Mistral request body must use response_format.type='json_object'"
        )


def _extract_error_message(response: httpx.Response) -> str:
    """
    Extract Mistral's structured error message without assuming every response
    is JSON. Mistral errors normally include message/type/code fields.
    """
    try:
        data = response.json()
    except json.JSONDecodeError:
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    if not isinstance(data, dict):
        return f"HTTP {response.status_code}"

    message = data.get("message")
    error_type = data.get("type")
    code = data.get("code")
    param = data.get("param")

    details = [
        str(part)
        for part in (message, error_type, code, param)
        if part
    ]

    return " | ".join(details) or f"HTTP {response.status_code}"


def _extract_content(data: dict[str, Any]) -> tuple[str, str | None]:
    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        raise MistralClientError(
            "Mistral response does not contain a non-empty choices array"
        )

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        raise MistralClientError("Mistral response choice has invalid format")

    message = first_choice.get("message")

    if not isinstance(message, dict):
        raise MistralClientError(
            "Mistral response choice does not contain a message object"
        )

    content = message.get("content")

    if isinstance(content, str):
        normalized_content = content
    elif isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])

        normalized_content = "".join(text_parts)
    else:
        raise MistralClientError(
            "Mistral response message content is missing or not textual"
        )

    if not normalized_content.strip():
        raise MistralClientError("Mistral response message content is empty")

    finish_reason = first_choice.get("finish_reason")

    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)

    return normalized_content, finish_reason


def _retry_delay_seconds(
    attempt: int,
    response: httpx.Response | None = None,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass

    return DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def call_mistral(
    request_body: dict[str, Any],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> MistralCompletionResult:
    """
    Sends a prepared request body to Mistral Chat Completions API.

    Responsibilities:
    - reads MISTRAL_API_KEY from environment;
    - posts HTTP request;
    - retries transient request failures, 429, and 5xx responses;
    - returns raw response and normalized assistant content;
    - does not validate the business JSON schema;
    - does not read or write PostgreSQL.

    The caller should parse result.content_as_json() and validate it against
    the dedicated Pydantic schema for the news video script.
    """
    _validate_request_body(request_body)

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    retryable_status_codes = {429, 500, 502, 503, 504}
    last_error: Exception | None = None

    started_at = time.perf_counter()

    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(1, max_attempts + 1):
            response: httpx.Response | None = None

            try:
                response = client.post(
                    MISTRAL_CHAT_COMPLETIONS_URL,
                    headers=headers,
                    json=request_body,
                )
            except httpx.TimeoutException as exc:
                last_error = exc

                if attempt == max_attempts:
                    raise MistralServerError(
                        f"Mistral request timed out after {max_attempts} attempts"
                    ) from exc

                time.sleep(_retry_delay_seconds(attempt))
                continue
            except httpx.RequestError as exc:
                last_error = exc

                if attempt == max_attempts:
                    raise MistralServerError(
                        f"Mistral request failed after {max_attempts} attempts: {exc}"
                    ) from exc

                time.sleep(_retry_delay_seconds(attempt))
                continue

            if response.status_code in retryable_status_codes:
                error_message = _extract_error_message(response)
                last_error = MistralClientError(
                    f"Mistral retryable HTTP {response.status_code}: {error_message}"
                )

                if attempt == max_attempts:
                    if response.status_code == 429:
                        raise MistralRateLimitError(
                            f"Mistral rate limit exceeded after {max_attempts} attempts: "
                            f"{error_message}"
                        )

                    raise MistralServerError(
                        f"Mistral server error after {max_attempts} attempts: "
                        f"HTTP {response.status_code} | {error_message}"
                    )

                time.sleep(_retry_delay_seconds(attempt, response))
                continue

            if response.status_code >= 400:
                error_message = _extract_error_message(response)

                raise MistralRequestError(
                    f"Mistral rejected request: HTTP {response.status_code} | "
                    f"{error_message}"
                )

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise MistralClientError(
                    "Mistral returned a successful response that is not valid JSON"
                ) from exc

            if not isinstance(data, dict):
                raise MistralClientError(
                    "Mistral returned a successful response that is not a JSON object"
                )

            content, finish_reason = _extract_content(data)

            latency_ms = round((time.perf_counter() - started_at) * 1000)

            usage = data.get("usage")

            if not isinstance(usage, dict):
                usage = {}

            return MistralCompletionResult(
                request_id=data.get("id"),
                model=data.get("model"),
                finish_reason=finish_reason,
                content=content,
                usage=usage,
                latency_ms=latency_ms,
                rate_limit_remaining=response.headers.get(
                    "X-RateLimit-Remaining"
                ),
                raw_response=data,
            )

    raise MistralServerError(
        f"Mistral request did not complete: {last_error}"
    )