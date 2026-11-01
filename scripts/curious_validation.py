"""Validation of screening responses without discarding valid records."""

from collections import Counter
from typing import Any


def validate_screening(
    answer: dict[str, Any],
    expected_ids: list[int],
) -> dict[str, Any]:
    if any(type(value) is not int for value in expected_ids):
        raise ValueError("Expected article IDs must be integers")

    expected = set(expected_ids)

    if len(expected) != len(expected_ids):
        raise ValueError("Duplicate article IDs in the input batch")

    valid = []
    errors = []

    items = answer.get("items") if isinstance(answer, dict) else None

    if not isinstance(items, list):
        return {
            "valid": [],
            "errors": [{"reason": "items must be a list"}],
            "unassessed_ids": sorted(expected),
        }

    counts = Counter(
        item["article_id"]
        for item in items
        if isinstance(item, dict)
        and type(item.get("article_id")) is int
    )

    for position, item in enumerate(items, start=1):
        reasons = []

        if not isinstance(item, dict):
            errors.append({
                "position": position,
                "reasons": ["item must be an object"],
                "item": item,
            })
            continue

        article_id = item.get("article_id")
        score = item.get("score")

        if type(article_id) is not int:
            reasons.append("article_id must be an integer")
        else:
            if article_id not in expected:
                reasons.append(
                    "article_id was not present in the input batch"
                )

            if counts[article_id] > 1:
                reasons.append("duplicate article_id in the response")

        if type(score) is not int or not 0 <= score <= 10:
            reasons.append("score must be an integer from 0 to 10")

        if item.get("decision") not in {"read", "maybe", "skip"}:
            reasons.append("decision must be read, maybe or skip")

        reason = item.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            reasons.append("reason must be a nonempty string")

        if reasons:
            errors.append({
                "position": position,
                "reasons": reasons,
                "item": item,
            })
        else:
            valid.append({
                "article_id": article_id,
                "decision": item["decision"],
                "score": score,
                "reason": reason.strip(),
            })

    accepted = {item["article_id"] for item in valid}

    return {
        "valid": valid,
        "errors": errors,
        "unassessed_ids": sorted(expected - accepted),
    }