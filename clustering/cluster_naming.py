from __future__ import annotations

from collections import Counter
import re

try:
    import yake  # type: ignore[import-not-found]
except ImportError:
    yake = None


MULTI_STOPWORDS = {
    "en": {
        "the", "and", "for", "with", "from", "that", "this", "after", "before",
        "into", "over", "under", "news", "said", "says", "will", "have", "has",
        "who", "what", "when", "where", "why", "how", "about", "against", "asks",
        "ask", "report", "reports", "file", "files", "new", "more", "than",
    },
    "pt": {
        "que", "quem", "sobre", "para", "como", "com", "sem", "das", "dos", "nas",
        "nos", "uma", "uns", "umas", "de", "do", "da", "e", "em", "por", "após",
        "depois", "antes", "mais", "menos", "hoje", "ontem", "amanhã", "diz",
        "disse", "veja", "todos", "todas", "candidato", "candidatos",
    },
    "ru": {
        "что", "как", "для", "после", "перед", "при", "без", "или", "это", "эта",
        "эти", "кто", "где", "куда", "откуда", "из", "в", "во", "на", "по", "о",
        "об", "про", "под", "над", "и", "но", "а", "ли", "же", "уже", "еще",
        "ещё", "сегодня", "завтра", "вчера", "стало", "известно", "сообщили",
        "почему", "который", "которая", "которые",
    },
}

GENERIC_NEWS_WORDS = {
    "president", "presidência", "presidencial", "presidente",
    "campaign", "campanha", "election", "eleição", "eleicoes",
    "debate", "candidato", "candidatos", "candidate", "candidates",
    "report", "analysis", "entrevista", "interview", "news",
    "globo", "bbc", "lenta", "aljazeera", "english",
}

SOURCE_WORDS = {
    "bbc", "brasil", "al", "jazeera", "english", "lenta", "ura", "breitbart",
    "news", "reuters", "associated", "press",
}


def _detect_title_language(text: str) -> str:
    low = text.lower()

    cyr = sum(1 for ch in low if "а" <= ch <= "я" or ch == "ё")
    if cyr >= 3:
        return "ru"

    pt_markers = [
        " que ", " não ", "ção", "ções", "presidência", "eleição", "entrevista",
        "candidato", "campanha", "sobre", "uma ", " à ", " por que ",
    ]
    if any(marker in f" {low} " for marker in pt_markers):
        return "pt"

    return "en"


def _detect_cluster_language(titles: list[str]) -> str | None:
    if not titles:
        return None

    counts = Counter(_detect_title_language(title) for title in titles if title)
    if not counts:
        return None

    language, score = counts.most_common(1)[0]
    if score <= 0:
        return None
    return language


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cleanup_keyword(phrase: str) -> str:
    phrase = _normalize_space(phrase)
    phrase = phrase.strip(".,:;!?-–—\"'()[]{}")
    return phrase


def _all_stopwords() -> set[str]:
    stop: set[str] = set()
    for words_set in MULTI_STOPWORDS.values():
        stop.update(words_set)
    return stop


def _is_bad_keyword(phrase: str, language_code: str | None) -> bool:
    if not phrase:
        return True

    p = _cleanup_keyword(phrase).lower()
    if not p:
        return True

    if len(p) < 3:
        return True

    if p.isdigit():
        return True

    words = re.findall(r"[\wÀ-ÿЁёА-Яа-я-]+", p)
    if not words:
        return True

    if all(word in SOURCE_WORDS for word in words):
        return True

    stop = _all_stopwords()

    if all(word in stop for word in words):
        return True

    if len(words) == 1 and words[0] in GENERIC_NEWS_WORDS:
        return True

    return False


def _dedupe_phrases(phrases: list[str]) -> list[str]:
    result: list[str] = []

    for phrase in phrases:
        cleaned = _cleanup_keyword(phrase)
        if not cleaned:
            continue

        key = cleaned.lower()

        duplicate = False
        for existing in result:
            e = existing.lower()
            if key == e or key in e or e in key:
                duplicate = True
                break

        if not duplicate:
            result.append(cleaned)

    return result


def _extract_title_phrases(titles: list[str]) -> list[str]:
    phrase_counts: Counter[str] = Counter()

    for title in titles:
        if not title:
            continue

        cleaned = re.sub(r"[\"“”‘’]", "", title)
        chunks = re.split(r"[,:;|()\[\]—–-]", cleaned)

        for chunk in chunks:
            chunk = _normalize_space(chunk)
            if not chunk:
                continue

            words = re.findall(r"[\wÀ-ÿЁёА-Яа-я]+", chunk)
            if not words:
                continue

            for n in (3, 2):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n]).strip()
                    if len(phrase) < 5:
                        continue
                    phrase_counts[phrase] += 1

    phrases = [phrase for phrase, _ in phrase_counts.most_common(40)]
    return _dedupe_phrases(phrases)


def _extract_frequent_tokens(titles: list[str]) -> list[str]:
    token_counts: Counter[str] = Counter()
    stop = _all_stopwords()

    for title in titles:
        if not title:
            continue

        tokens = re.findall(r"[\wÀ-ÿЁёА-Яа-я-]{3,}", title.lower())
        for token in tokens:
            if token in stop:
                continue
            if token in SOURCE_WORDS:
                continue
            token_counts[token] += 1

    ranked = [token for token, _ in token_counts.most_common(30)]
    return _dedupe_phrases(ranked)


def _extract_yake_keywords(text: str, language_code: str | None) -> list[str]:
    if yake is None:
        return []

    try:
        kw_extractor = yake.KeywordExtractor(
            lan=language_code if language_code in {"en", "pt", "ru"} else "en",
            n=3,
            dedupLim=0.35,
            dedupFunc="seqm",
            windowsSize=2,
            top=20,
            features=None,
        )
        extracted = kw_extractor.extract_keywords(text)
        return [kw for kw, _score in extracted]
    except Exception:
        return []


def _score_candidates(
    titles: list[str],
    yake_keywords: list[str],
    heuristic_phrases: list[str],
    token_candidates: list[str],
    language_code: str | None,
) -> list[str]:
    title_lows = [title.lower() for title in titles if title]
    ranked: list[tuple[float, str]] = []

    candidates = _dedupe_phrases(yake_keywords + heuristic_phrases + token_candidates)

    for phrase in candidates:
        cleaned = _cleanup_keyword(phrase)
        low = cleaned.lower()

        if _is_bad_keyword(cleaned, language_code):
            continue

        article_hits = sum(1 for title in title_lows if low in title)
        word_count = len(low.split())

        score = 0.0
        score += article_hits * 3.0
        score += min(word_count, 3) * 1.2

        if word_count >= 2:
            score += 2.5

        if low in GENERIC_NEWS_WORDS:
            score -= 3.0

        ranked.append((score, cleaned))

    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    return _dedupe_phrases([phrase for _score, phrase in ranked])


def _build_display_names(
    candidates: list[str],
    cluster_id: int,
) -> tuple[str, str, list[str], list[str]]:
    if not candidates:
        fallback = f"Cluster {cluster_id}"
        return fallback, fallback, [], []

    tags = candidates[:5]
    concepts = candidates[:7]

    if len(tags) >= 2:
        name_short = " · ".join(tags[:2])
    else:
        name_short = tags[0]

    if len(tags) >= 3:
        name_title = " · ".join(tags[:3])
    else:
        name_title = name_short

    return name_short[:80], name_title[:160], tags, concepts


def generate_cluster_name(conn, cluster_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.title, a.source
            FROM cluster_articles ca
            JOIN articles a ON a.id = ca.article_id
            WHERE ca.cluster_id = %s
            ORDER BY a.published DESC NULLS LAST, a.id DESC
            """,
            (cluster_id,),
        )
        rows = cur.fetchall()

    if not rows:
        fallback = f"Cluster {cluster_id}"
        return {
            "name_short": fallback,
            "name_title": fallback,
            "tags": [],
            "language_code": None,
            "concepts": [],
        }

    titles = [row[0] for row in rows if row[0]]
    if not titles:
        fallback = f"Cluster {cluster_id}"
        return {
            "name_short": fallback,
            "name_title": fallback,
            "tags": [],
            "language_code": None,
            "concepts": [],
        }

    language_code = _detect_cluster_language(titles) or "en"
    text = "\n".join(titles)

    yake_keywords = _extract_yake_keywords(text, language_code)
    heuristic_phrases = _extract_title_phrases(titles)
    token_candidates = _extract_frequent_tokens(titles)

    candidates = _score_candidates(
        titles=titles,
        yake_keywords=yake_keywords,
        heuristic_phrases=heuristic_phrases,
        token_candidates=token_candidates,
        language_code=language_code,
    )

    name_short, name_title, tags, concepts = _build_display_names(candidates, cluster_id)

    return {
        "name_short": name_short,
        "name_title": name_title,
        "tags": tags,
        "language_code": language_code,
        "concepts": concepts,
    }


def upsert_cluster_name(conn, cluster_id: int) -> None:
    payload = generate_cluster_name(conn, cluster_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cluster_names (
                cluster_id,
                name_short,
                name_title,
                tags,
                language_code,
                concepts
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE
            SET
                name_short = EXCLUDED.name_short,
                name_title = EXCLUDED.name_title,
                tags = EXCLUDED.tags,
                language_code = EXCLUDED.language_code,
                concepts = EXCLUDED.concepts
            """,
            (
                cluster_id,
                payload["name_short"],
                payload["name_title"],
                payload["tags"],
                payload["language_code"],
                payload["concepts"],
            ),
        )


def upsert_cluster_names_for_run(conn, run_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM clusters WHERE run_id = %s ORDER BY id",
            (run_id,),
        )
        cluster_ids = [row[0] for row in cur.fetchall()]

    for cluster_id in cluster_ids:
        upsert_cluster_name(conn, cluster_id)

    return len(cluster_ids)