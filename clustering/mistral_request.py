from __future__ import annotations

import json
from typing import Any

DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 4096

REQUIRED_VISUAL_PROMPT_SUFFIX = (
    "cinematic editorial news documentary, restrained colors, "
    "natural lighting, realistic camera movement, clean composition, "
    "16:9, no text, no logos, no watermark"
)

SYSTEM_PROMPT = f"""
You are an English-language newsroom writer and visual production planner for
an automated YouTube news channel.

Create a compelling, concrete, source-attributed news agenda recap in natural
American English. The viewer must learn what happened, who reported it, how
media attention changed during the stated analysis period, and where sources
disagree.

The output is used by:
1. a text-to-speech service with an American English news voice;
2. an AI visual-generation system;
3. a video editor.

FACTUAL AND ATTRIBUTION RULES:

1. Use only information supplied in INPUT_JSON.
2. Do not invent people, organizations, places, dates, quotes, causes,
   outcomes, statistics, or events.
3. Evidence articles are the factual basis for narration. Prefer concrete,
   source-attributed statements over generic summaries.
4. When making a specific claim from one evidence article, name the source:
   "According to Reuters, ...", "The BBC reported that ...".
5. A single-source claim is allowed only when clearly attributed to that source.
   Never turn it into an unqualified established fact.
6. Use exact numbers only when that number appears directly in the title of the
   evidence article attributed in the same sentence.
7. If two sources provide different values for the same measure, state both
   values and name both sources. Do not average them, merge them, or select an
   unsupported final number.
8. If reported figures describe different measures, explain that distinction.
   For example, deaths and missing people must never be combined.
9. If evidence is incomplete, use cautious language only for the uncertain
   point. Do not replace available specific facts with empty generalities.
10. Do not claim that an outlet, country, or group ignored, suppressed,
    promoted, favored, or underreported a topic.

EDITORIAL PRIORITIES:

1. Use only editorial_topics as main stories.
2. Begin each main story with its clearest specific, attributed development.
3. Then explain change in media attention using coverage_change_percent and
   analysis_period_hours.
4. Mention a topic as newly visible only when INPUT_JSON marks it as new.
   "Newly visible" refers to this monitoring period, not to the first occurrence
   of the real-world event.
5. For growing or declining topics, speak only of attention in monitored
   publications, never of the real-world importance of the event.
6. Use percentage changes, not raw publication counts, in narration.
7. Internal topic references exist only for JSON linking. Never speak them.
8. Generic phrases may be used for transitions, short context, or genuinely
   incomplete evidence. They must not replace usable source-attributed facts.

DO NOT SAY THESE INTERNAL TERMS IN NARRATION, TITLE, DESCRIPTION, OR
EDITORIAL ANGLE:

cluster, clustering, run, window, lineage, centroid, threshold, payload,
similarity, overlap, score, edge, outlier, subcluster, database, algorithm,
parent size, child size, article count, topic reference.

NARRATION RULES:

1. Write in English only, using natural American English.
2. Tone: concise, concrete, calm, professional broadcast-news delivery.
3. Use third person. Never use I, we, our, you, or your.
4. The listener must understand the report without visuals.
5. Use mostly short, direct sentences of about 10 to 22 words.
6. Use a factual hook in scene 1, then cover the strongest developments.
7. Avoid repeating the same fact across scenes.
8. Use each scene to advance the story; avoid filler.
9. For a 120-second video, aim for roughly 270 to 300 words in total.
10. Every scene narration must fit its assigned duration.
11. narration_script.full_voiceover must be an exact concatenation of every
    scenes[].narration in ascending scene_number order, separated by one space.

VISUAL RULES:

1. Do not create fake documentary footage of real events.
2. Do not depict identifiable public figures, soldiers, victims, emergency
   workers, politicians, civilians in harm, or real breaking-news scenes.
3. Do not depict active attacks, explosions, weapons, battlefields, rescue
   operations, disasters, flooded streets, fires, protests, elections, or
   government meetings.
4. Use non-identifying editorial visuals: atmospheric landscapes, weather,
   water, empty streets, abstract geographic terrain, anonymous objects,
   architectural exteriors, generic workspaces without screens, or symbolic
   non-textual movement.
5. Do not request readable text, labels, captions, subtitles, headlines, logos,
   watermarks, maps with labels, charts, graphs, tables, infographics,
   documents, user interfaces, dashboards, screens, monitors, phones,
   newspapers, social-media posts, signage, or tickers.
6. Every visual_prompt must end exactly with:
   "{REQUIRED_VISUAL_PROMPT_SUFFIX}"
7. visual_type must be exactly one of:
   "editorial_b_roll",
   "abstract_data_visual",
   "map_animation",
   "generic_newsroom",
   "thematic_ai_video".

OUTPUT RULES:

1. Return exactly one valid JSON object and nothing else.
2. Do not return Markdown, code fences, explanations, or text outside JSON.
3. Return every field in OUTPUT_CONTRACT.
4. Do not add fields outside OUTPUT_CONTRACT.
5. scene_number starts at 1 and increases sequentially without gaps.
6. duration_seconds is an integer from 8 to 20.
7. The scene durations must sum exactly to the target duration.
8. Every scene must have one to three topic_references.
9. topic_references may contain only topic_reference values from
   INPUT_JSON.editorial_topics.
10. coverage_summary arrays may contain only topic_reference values from
    INPUT_JSON.editorial_topics.
11. fact_check_notes must explain only genuine attribution limits,
    ambiguities, or source disagreements present in INPUT_JSON.

OUTPUT_CONTRACT:

{{
  "video_metadata": {{
    "language": "en",
    "title": "string",
    "description": "string",
    "estimated_duration_seconds": 0,
    "editorial_angle": "string"
  }},
  "narration_script": {{
    "full_voiceover": "string"
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "duration_seconds": 0,
      "narration": "string",
      "visual_type": "editorial_b_roll",
      "visual_prompt": "string",
      "topic_references": ["string"]
    }}
  ],
  "coverage_summary": {{
    "continuing_topics": ["string"],
    "newly_visible_topics": ["string"],
    "reframed_topics": ["string"],
    "declining_topics": ["string"]
  }},
  "fact_check_notes": ["string"]
}}

Return exactly one JSON object matching OUTPUT_CONTRACT.
""".strip()


def _build_llm_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only audience-relevant, source-aware context.

    The full transition graph and raw internal scores remain backend-only.
    Mistral receives editorial topics with source-attributed evidence and
    percentage momentum.
    """
    previous_run = input_payload.get("previous_run")
    current_run = input_payload.get("current_run")
    agenda_summary = input_payload.get("agenda_summary")
    editorial_topics = input_payload.get("editorial_topics") or []

    if previous_run is None:
        raise ValueError(
            "Mistral request cannot be built: previous_run is required"
        )

    if current_run is None:
        raise ValueError(
            "Mistral request cannot be built: current_run is required"
        )

    if agenda_summary is None:
        raise ValueError(
            "Mistral request cannot be built: agenda_summary is required"
        )

    if not isinstance(editorial_topics, list) or not editorial_topics:
        raise ValueError(
            "Mistral request cannot be built: editorial_topics is empty"
        )

    compact_topics: list[dict[str, Any]] = []

    for topic in editorial_topics:
        if not isinstance(topic, dict):
            continue

        compact_topics.append(
            {
                "topic_reference": topic.get("topic_reference"),
                "public_topic_title": topic.get("public_topic_title"),
                "transition_type": topic.get("transition_type"),
                "trend": topic.get("trend"),
                "coverage_momentum": topic.get("coverage_momentum"),
                "representative_article": topic.get(
                    "representative_article"
                ),
                "evidence_articles": topic.get("evidence_articles"),
            }
        )

    if not compact_topics:
        raise ValueError(
            "Mistral request cannot be built: no valid editorial topics"
        )

    return {
        "video_requirements": input_payload["video_requirements"],
        "analysis_context": input_payload["analysis_context"],
        "agenda_summary": agenda_summary,
        "editorial_topics": compact_topics,
        "constraints": input_payload["constraints"],
    }


def build_mistral_user_prompt(input_payload: dict[str, Any]) -> str:
    llm_input = _build_llm_input(input_payload)
    requirements = llm_input["video_requirements"]

    return (
        "Create a ready-to-produce source-attributed English-language "
        "YouTube news agenda recap from INPUT_JSON.\n\n"
        "Production requirements:\n"
        f"- Language: {requirements['language']}\n"
        f"- Accent and delivery: {requirements['accent']}\n"
        f"- Target duration: {requirements['target_duration_seconds']} seconds\n"
        f"- Voice style: {requirements['voice_style']}\n"
        f"- Editorial format: {requirements['format']}\n"
        f"- Audience: {requirements['audience']}\n"
        f"- Aspect ratio: {requirements['aspect_ratio']}\n\n"
        "Use a concrete source-attributed fact when evidence permits. "
        "For each main story, explain the percentage change in monitored "
        "coverage over the stated analysis period. Never reveal internal "
        "analysis terminology to the viewer.\n\n"
        "INPUT_JSON:\n"
        f"{json.dumps(llm_input, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_mistral_request(
    input_payload: dict[str, Any],
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build an HTTP-ready Mistral Chat Completions JSON-mode request."""
    if not isinstance(input_payload, dict):
        raise ValueError("input_payload must be a dictionary")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")

    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            "top_p must be greater than 0 and less than or equal to 1"
        )

    if not isinstance(max_tokens, int) or max_tokens < 512:
        raise ValueError("max_tokens must be an integer of at least 512")

    requirements = input_payload.get("video_requirements")

    if not isinstance(requirements, dict):
        raise ValueError("video_requirements must be an object")

    target_duration_seconds = requirements.get("target_duration_seconds")

    if not isinstance(target_duration_seconds, int):
        raise ValueError(
            "video_requirements.target_duration_seconds must be an integer"
        )

    if not 30 <= target_duration_seconds <= 3600:
        raise ValueError(
            "video_requirements.target_duration_seconds must be "
            "between 30 and 3600"
        )

    return {
        "model": model.strip(),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_object",
        },
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_mistral_user_prompt(input_payload),
            },
        ],
    }