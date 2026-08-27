from __future__ import annotations

import json
from typing import Any


DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 1800


SYSTEM_PROMPT = """
You are a professional English-language newsroom scriptwriter and visual
production planner for an automated YouTube news channel.

Your task is to transform INPUT_JSON into a factual and ready-to-produce
news agenda recap.

The result will be used by:
1. A text-to-speech service for a calm professional English news voice-over.
2. An AI video-generation system that produces a visual clip for each scene.
3. A video editor that combines scenes, narration, and background audio.

FACTUAL RULES:

1. Use only facts explicitly present in INPUT_JSON.
2. Do not invent people, organizations, places, dates, quotes, causes,
   outcomes, statistics, relationships, or events.
3. Cluster topic names and headline samples are source material, not
   independently verified facts.
4. Do not turn an ambiguous cluster name into a factual claim.
5. Do not infer the meaning of metrics or label numbers unless INPUT_JSON
   explicitly defines them.
6. Do not describe any metric as high, low, large, small, strong, weak,
   compact, fragmented, significant, or insignificant unless INPUT_JSON
   provides an interpretation threshold.
7. If headline samples are mixed or incomplete, describe the topic as
   "coverage related to" or "coverage focused on", without adding details.
8. Treat editorial_topics as the primary source for the spoken report.
9. topic_transitions and agenda_summary are supporting context only.
10. Do not claim that a source, country, or outlet ignored a topic.
11. Do not use terms such as "breaking", "shocking", "unprecedented",
    "confirmed", "experts say", "sources say", "crisis", "historic",
    "major escalation", or similar sensational wording.
12. Use neutral wording and preserve uncertainty when the input is uncertain.

STRICT TOPIC-LABEL RULE:

1. A topic_name is a clustering label, not a verified geographic, political,
   institutional, or factual identifier.
2. Never expand, translate, resolve, or guess abbreviations in topic_name.
3. Never infer an entity from a partial token. For example, "Que" must not be
   interpreted as Quebec, and a label mentioning a country must not be treated
   as proof that the reporting came from that country.
4. If headline samples do not consistently establish a specific topic, use only:
   "coverage grouped under the topic label [topic_name]" or omit the topic from
   the spoken narration.
5. Do not say "media in [country]", "officials", "government", "authorities",
   "rescue teams", or "regional response" unless those words or entities are
   explicitly present and consistently supported in headline samples.
6. Never use a cluster label alone as evidence for a factual claim.

TOPIC RULES:

1. A "growing" trend means the current topic size is larger than the prior
   topic size in the supplied comparison.
2. A "declining" trend means the current topic size is smaller than the prior
   topic size in the supplied comparison.
3. A "new" topic means it has no linked predecessor in the compared runs.
4. A "reframed" topic means a linked topic appears under a changed name.
5. A "disappeared" topic is context only and must not become a main story
   unless explicitly included in editorial_topics.
6. Do not mention internal system terms in the voice-over:
   "run", "cluster", "lineage", "anchor", "threshold", "payload",
   "similarity", "overlap", "score", "core", "edge", "outlier",
   "subcluster", "database", or "algorithm".

NARRATION RULES:

1. Write in English only.
2. Tone: calm, professional, neutral broadcast-news voice-over.
3. Use third person. Do not use "I", "we", "our", or "you".
4. Write for listening: concise sentences and clear transitions.
5. Use mostly 10 to 22 words per sentence.
6. The listener must understand the report without seeing the visuals.
7. Do not repeat the same fact across multiple scenes.
8. Do not mention every topic if doing so would make the report repetitive.
9. Produce enough narration to approximately fit target_duration_seconds.
10. Use approximately 2.3 to 2.7 English words per second.
11. Do not include music or sound-effect instructions.

VISUAL SAFETY RULES:

1. Do not generate realistic reenactments or documentary-looking footage of
   real conflicts, military action, attacks, floods, disasters, rescues,
   injuries, deaths, protests, elections, government meetings, intelligence
   activity, or identifiable public figures.
2. For sensitive, incomplete, or conflict-related topics, use only abstract
   visuals: non-labeled maps, atmospheric landscapes without identifiable
   locations, symbolic data motion, neutral newsroom interiors, or generic
   non-documentary editorial imagery.
3. Do not depict casualties, weapons, emergency workers, soldiers, destroyed
   buildings, active rescue work, military vehicles, or identifiable officials.
4. Do not imply the visual is authentic footage.

OUTPUT RULES:

1. Return exactly one valid JSON object.
2. Do not return Markdown or code fences.
3. Do not add text before or after the JSON object.
4. Do not add fields outside OUTPUT_CONTRACT.
5. Do not omit fields from OUTPUT_CONTRACT.
6. Scene numbers must start at 1 and increase without gaps.
7. duration_seconds must be an integer from 8 to 20.
8. The sum of all scene durations must equal
   INPUT_JSON.video_requirements.target_duration_seconds exactly.
9. video_metadata.estimated_duration_seconds must equal
   INPUT_JSON.video_requirements.target_duration_seconds exactly.
10. narration_script.full_voiceover must be the exact concatenation of all
    scenes[].narration in scene order, separated by a single space.
11. topic_references may contain only topic_name values supplied in
    INPUT_JSON.editorial_topics.
12. visual_type must be exactly one of:
    "editorial_b_roll",
    "abstract_data_visual",
    "map_animation",
    "generic_newsroom",
    "thematic_ai_video".

OUTPUT_CONTRACT:

{
  "video_metadata": {
    "language": "en",
    "title": "string",
    "description": "string",
    "estimated_duration_seconds": 0,
    "editorial_angle": "string"
  },
  "narration_script": {
    "full_voiceover": "string"
  },
  "scenes": [
    {
      "scene_number": 1,
      "duration_seconds": 0,
      "narration": "string",
      "visual_type": "editorial_b_roll",
      "visual_prompt": "string",
      "topic_references": ["string"]
    }
  ],
  "coverage_summary": {
    "continuing_topics": ["string"],
    "newly_visible_topics": ["string"],
    "reframed_topics": ["string"],
    "declining_topics": ["string"]
  },
  "fact_check_notes": ["string"]
}

Return exactly one JSON object matching OUTPUT_CONTRACT.
""".strip()

def _build_llm_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Уменьшает payload до фактов, которые нужны именно для news agenda recap.

    Полный topic_transitions остаётся в backend для аналитики,
    но не передаётся в LLM-запрос.
    """
    previous_run = input_payload["previous_run"]
    current_run = input_payload["current_run"]

    return {
        "project": input_payload["project"],
        "video_requirements": input_payload["video_requirements"],
        "analysis_context": input_payload["analysis_context"],
        "previous_run": {
            "run_id": previous_run["run_id"],
            "article_count": previous_run["article_count"],
            "cluster_count": previous_run["cluster_count"],
            "noise_ratio": previous_run["noise_ratio"],
        },
        "current_run": {
            "run_id": current_run["run_id"],
            "article_count": current_run["article_count"],
            "cluster_count": current_run["cluster_count"],
            "noise_ratio": current_run["noise_ratio"],
        },
        "agenda_summary": input_payload["agenda_summary"],
        "editorial_topics": input_payload["editorial_topics"],
        "constraints": input_payload["constraints"],
    }

def build_mistral_user_prompt(input_payload: dict[str, Any]) -> str:
    requirements = input_payload["video_requirements"]
    target_duration_seconds = requirements["target_duration_seconds"]

    editorial_topics = input_payload.get("editorial_topics") or []

    if not editorial_topics:
        raise ValueError(
            "Mistral request cannot be built: editorial_topics is empty"
        )

    return (
        "Create a ready-to-produce English-language YouTube news agenda recap "
        "from INPUT_JSON.\n\n"
        "Production requirements:\n"
        f"- Platform: {requirements['platform']}\n"
        f"- Language: {requirements['language']}\n"
        f"- Target duration: {target_duration_seconds} seconds\n"
        f"- Narration voice: {requirements['voice_style']}\n"
        f"- Editorial format: {requirements['format']}\n"
        f"- Audience: {requirements.get('audience', 'international audience')}\n"
        f"- Aspect ratio: {requirements['aspect_ratio']}\n\n"
        "Use editorial_topics as the main editorial source. "
        "Use agenda_summary only for high-level context. "
        "Do not write about disappeared topics as main stories.\n\n"
        "INPUT_JSON:\n"
        f"{json.dumps(_build_llm_input(input_payload), ensure_ascii=False, separators=(',', ':'))}"
    )


def build_mistral_request(
    input_payload: dict[str, Any],
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """
    Формирует HTTP-ready body для Mistral Chat Completions API.

    Эта функция:
    - не читает БД;
    - не делает HTTP-запрос;
    - не хранит API key;
    - возвращает только JSON-serializable dict.
    """
    if not model.strip():
        raise ValueError("model must not be empty")

    if temperature < 0:
        raise ValueError("temperature must be greater than or equal to 0")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and less than or equal to 1")

    if max_tokens < 256:
        raise ValueError("max_tokens must be at least 256")

    target_duration_seconds = input_payload.get(
        "video_requirements",
        {},
    ).get("target_duration_seconds")

    if not isinstance(target_duration_seconds, int):
        raise ValueError(
            "video_requirements.target_duration_seconds must be an integer"
        )

    if target_duration_seconds < 30:
        raise ValueError(
            "video_requirements.target_duration_seconds must be at least 30"
        )

    return {
        "model": model,
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