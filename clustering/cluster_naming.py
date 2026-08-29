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
    "de": {
        "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder",
        "mit", "ohne", "von", "zu", "zum", "zur", "im", "in", "am", "an", "auf",
        "über", "unter", "nach", "vor", "für", "gegen", "wie", "was", "wer", "wo",
        "warum", "dass", "sagte", "sagt", "bericht", "berichte",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "avec", "sans",
        "dans", "sur", "sous", "pour", "contre", "après", "avant", "que", "qui",
        "quoi", "où", "pourquoi", "comment", "dit", "selon", "rapport", "rapports",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "y", "o",
        "con", "sin", "en", "sobre", "para", "contra", "después", "antes", "que",
        "quien", "qué", "dónde", "por", "cómo", "dijo", "según", "reporte", "reportes",
    },
    "ja": {
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある",
        "いる", "する", "した", "として", "について", "による", "など", "という",
    },
    "ko": {
        "의", "에", "이", "가", "을", "를", "은", "는", "와", "과", "도", "로", "으로",
        "에서", "에게", "하다", "했다", "대한", "관련", "라고", "이라고",
    },
    "zh": {
        "的", "了", "在", "是", "和", "与", "对", "就", "都", "而", "及", "或", "被",
        "将", "把", "由", "关于", "有关", "一个", "一种", "表示", "称",
    },
    "sv": {
        "och", "det", "att", "som", "för", "med", "utan", "från", "till", "på", "i",
        "av", "om", "mot", "efter", "före", "vad", "vem", "var", "hur", "sade",
    },
    "no": {
        "og", "det", "at", "som", "for", "med", "uten", "fra", "til", "på", "i",
        "av", "om", "mot", "etter", "før", "hva", "hvem", "hvor", "hvordan", "sa",
    },
    "da": {
        "og", "det", "at", "som", "for", "med", "uden", "fra", "til", "på", "i",
        "af", "om", "mod", "efter", "før", "hvad", "hvem", "hvor", "hvordan", "sagde",
    },
    "ar": {
        "في", "من", "على", "إلى", "عن", "مع", "دون", "بعد", "قبل", "هذا", "هذه",
        "ذلك", "الذي", "التي", "ما", "ماذا", "من", "أين", "كيف", "قال", "بحسب",
    },
    "pl": {
        "i", "oraz", "z", "ze", "w", "we", "na", "do", "od", "po", "przed", "pod",
        "nad", "o", "u", "bez", "dla", "jak", "co", "kto", "gdzie", "dlaczego",
        "który", "która", "które", "powiedział", "według", "raport",
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

LEADING_FUNCTION_WORDS = {
    "en": {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "from"},
    "pt": {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "na", "no", "nas", "nos", "à", "ao", "e", "que", "por", "para", "com", "sem"},
    "ru": {"и", "в", "во", "на", "по", "о", "об", "от", "до", "из", "у", "к", "ко", "с", "со", "за", "под", "над", "при", "что", "как"},
    "de": {"der", "die", "das", "ein", "eine", "und", "oder", "von", "zu", "im", "in", "am", "an", "auf", "für", "mit"},
    "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "dans", "sur", "pour", "avec"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "y", "o", "en", "para", "con", "sin"},
    "sv": {"och", "det", "att", "som", "för", "med", "från", "till", "på", "i", "av", "om"},
    "no": {"og", "det", "at", "som", "for", "med", "fra", "til", "på", "i", "av", "om"},
    "da": {"og", "det", "at", "som", "for", "med", "fra", "til", "på", "i", "af", "om"},
    "ar": {"في", "من", "على", "إلى", "عن", "مع", "بعد", "قبل", "هذا", "هذه", "الذي", "التي"},
    "pl": {"i", "oraz", "z", "ze", "w", "we", "na", "do", "od", "po", "o", "u", "dla", "jak", "co"},
}

TRAILING_FUNCTION_WORDS = {
    "en": {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "from"},
    "pt": {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "na", "no", "nas", "nos", "à", "ao", "e", "que", "por", "para", "com", "sem"},
    "ru": {"и", "в", "во", "на", "по", "о", "об", "от", "до", "из", "у", "к", "ко", "с", "со", "за", "под", "над", "при", "что", "как"},
    "de": {"der", "die", "das", "ein", "eine", "und", "oder", "von", "zu", "im", "in", "am", "an", "auf", "für", "mit"},
    "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "dans", "sur", "pour", "avec"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "y", "o", "en", "para", "con", "sin"},
    "sv": {"och", "det", "att", "som", "för", "med", "från", "till", "på", "i", "av", "om"},
    "no": {"og", "det", "at", "som", "for", "med", "fra", "til", "på", "i", "av", "om"},
    "da": {"og", "det", "at", "som", "for", "med", "fra", "til", "på", "i", "af", "om"},
    "ar": {"في", "من", "على", "إلى", "عن", "مع", "بعد", "قبل", "هذا", "هذه", "الذي", "التي"},
    "pl": {"i", "oraz", "z", "ze", "w", "we", "na", "do", "od", "po", "o", "u", "dla", "jak", "co"},
}

BANNED_PHRASE_PATTERNS = [
    r"^\w{1,2}\s+\w{1,3}$",
    r"^(o|a|os|as|um|uma|que)\s+",
    r"^(the|a|an|of|in|on|to)\s+",
    r"^(и|в|на|по|что|как)\s+",
    r"^(der|die|das|ein|eine|und|mit|von|zu)\s+",
    r"^(le|la|les|un|une|des|de|du|et|avec)\s+",
    r"^(el|la|los|las|un|una|de|del|y|con)\s+",
    r"^(i|oraz|z|ze|w|we|na|do|od)\s+",
    r"\s+(de|da|do|dos|das|na|no|nas|nos|à|ao)$",
    r"\s+(of|in|on|to|for|by|with|from)$",
    r"\s+(в|на|по|о|об|с|со|из|от)$",
    r"\s+(der|die|das|mit|von|zu|im|am|an|auf)$",
    r"\s+(de|du|des|dans|sur|avec|pour)$",
    r"\s+(de|del|en|para|con|sin)$",
    r"\s+(i|oraz|z|ze|w|we|na|do|od|po)$",
    r"^(disse|says|said|сообщили|заявили|sagte|dit|dijo|powiedział|قال)\b",
    r"\b(disse o candidato|que disse o|o que|à presidência|said the candidate|что сказал|lo que|ce que)\b",
]

CJK_BANNED_EXACT = {
    "について", "として", "による", "という", "有关", "关于", "表示", "称", "관련", "대한",
}


def _detect_title_language(text: str) -> str:
    low = text.lower()

    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"

    cyr = sum(1 for ch in low if "а" <= ch <= "я" or ch == "ё")
    if cyr >= 3:
        return "ru"

    pt_markers = [
        " que ", " não ", "ção", "ções", "presidência", "eleição", "entrevista",
        "candidato", "campanha", "sobre", "uma ", " à ", " por que ",
    ]
    if any(marker in f" {low} " for marker in pt_markers):
        return "pt"

    de_markers = [" der ", " die ", " das ", " und ", " mit ", " von ", " sagt "]
    if any(marker in f" {low} " for marker in de_markers):
        return "de"

    fr_markers = [" le ", " la ", " les ", " des ", " avec ", " pour ", " dit "]
    if any(marker in f" {low} " for marker in fr_markers):
        return "fr"

    es_markers = [" el ", " la ", " los ", " las ", " para ", " con ", " dijo "]
    if any(marker in f" {low} " for marker in es_markers):
        return "es"

    sv_markers = [" och ", " för ", " med ", " från ", " sade "]
    if any(marker in f" {low} " for marker in sv_markers):
        return "sv"

    no_markers = [" og ", " for ", " med ", " fra ", " sa "]
    if any(marker in f" {low} " for marker in no_markers):
        return "no"

    da_markers = [" og ", " for ", " med ", " fra ", " sagde "]
    if any(marker in f" {low} " for marker in da_markers):
        return "da"

    pl_markers = [" oraz ", " powiedział ", " według ", " który ", " która "]
    if any(marker in f" {low} " for marker in pl_markers):
        return "pl"

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


def _looks_like_named_entity(phrase: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿЁёА-Яа-я]+", phrase) if w]
    if not words:
        return False
    significant = [w for w in words if len(w) >= 3]
    if not significant:
        return False
    titled = [w for w in significant if w[:1].isupper()]
    return len(titled) >= 1 and len(titled) >= max(1, len(significant) // 2)


def _has_weak_inner_words(phrase: str) -> bool:
    words = _tokenize_phrase(phrase.lower())
    if len(words) < 3:
        return False
    stop = _all_stopwords()
    inner = words[1:-1]
    return any(w in stop for w in inner)


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


def _tokenize_phrase(text: str) -> list[str]:
    if re.search(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff\u0600-\u06ff]", text):
        if " " in text:
            return [t for t in text.split() if t]
        return [text]
    return re.findall(r"[\wÀ-ÿЁёА-Яа-я-]+", text.lower())


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

    words = _tokenize_phrase(p)
    if not words:
        return True

    if all(word in SOURCE_WORDS for word in words):
        return True

    stop = _all_stopwords()

    if all(word in stop for word in words):
        return True

    if len(words) == 1 and words[0] in GENERIC_NEWS_WORDS:
        return True

    if p in CJK_BANNED_EXACT:
        return True

    return False


def _has_mixed_scripts(text: str) -> bool:
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
    has_arabic = bool(re.search(r"[\u0600-\u06ff]", text))
    has_cjk = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", text))

    groups = sum([has_latin, has_cyrillic, has_arabic, has_cjk])
    return groups >= 2


def _strong_word_count(phrase: str, language_code: str | None) -> int:
    words = _tokenize_phrase(phrase.lower())
    if not words:
        return 0

    stop = _all_stopwords()
    strong = [
        w for w in words
        if len(w) >= 4 and w not in stop and w not in SOURCE_WORDS
    ]
    return len(strong)


def _document_frequency(phrase: str, titles: list[str]) -> int:
    low = phrase.lower()
    return sum(1 for title in titles if low in title.lower())


def _is_junk_phrase(phrase: str, language_code: str | None) -> bool:
    p = _cleanup_keyword(phrase).lower()
    if not p:
        return True

    words = _tokenize_phrase(p)
    if not words:
        return True

    lang = language_code if language_code in {
        "en", "pt", "ru", "de", "fr", "es", "sv", "no", "da", "ar", "pl"
    } else None

    if len(words) == 2 and all(len(w) <= 3 for w in words):
        return True

    if lang:
        if len(words) >= 1 and words[0] in LEADING_FUNCTION_WORDS.get(lang, set()):
            return True
        if len(words) >= 1 and words[-1] in TRAILING_FUNCTION_WORDS.get(lang, set()):
            return True

    for pattern in BANNED_PHRASE_PATTERNS:
        if re.search(pattern, p, flags=re.IGNORECASE):
            return True

    if language_code in {"ja", "ko", "zh"}:
        if p in CJK_BANNED_EXACT:
            return True
        if len(p) <= 2:
            return True

    if language_code == "ar":
        if p in {"قال", "بحسب", "هذا", "هذه"}:
            return True

    if _has_mixed_scripts(p):
        return True

    stop = _all_stopwords()
    if words[0] in stop or words[-1] in stop:
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

            if re.search(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff\u0600-\u06ff]", chunk):
                if len(chunk) >= 4:
                    phrase_counts[chunk] += 1
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

        if re.search(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff\u0600-\u06ff]", title):
            if len(title.strip()) >= 4:
                token_counts[title.strip()] += 1
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

    lang_for_yake = language_code if language_code in {
        "en", "pt", "ru", "de", "fr", "es", "sv", "no", "da", "ar"
    } else "en"

    try:
        kw_extractor = yake.KeywordExtractor(
            lan=lang_for_yake,
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
    ranked: list[tuple[float, str]] = []

    candidates = _dedupe_phrases(yake_keywords + heuristic_phrases + token_candidates)

    for phrase in candidates:
        cleaned = _cleanup_keyword(phrase)
        low = cleaned.lower()

        if _is_bad_keyword(cleaned, language_code):
            continue

        if _is_junk_phrase(cleaned, language_code):
            continue

        article_hits = _document_frequency(cleaned, titles)
        word_count = len(_tokenize_phrase(low))

        score = 0.0
        score += article_hits * 3.0
        score += min(word_count, 3) * 1.2

        if article_hits >= 2:
            score += 4.0
        else:
            score -= 2.5

        if word_count >= 2:
            score += 2.5

        if low in GENERIC_NEWS_WORDS:
            score -= 3.0

        if _looks_like_named_entity(cleaned):
            score += 2.5

        if _has_weak_inner_words(cleaned):
            score -= 2.0

        ranked.append((score, cleaned))

    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    return _dedupe_phrases([phrase for _score, phrase in ranked])


def _build_display_names(
    candidates: list[str],
    cluster_id: int,
    language_code: str | None = None,
) -> tuple[str, str, list[str], list[str]]:
    if not candidates:
        fallback = f"Cluster {cluster_id}"
        return fallback, fallback, [], []

    tags = candidates[:5]
    concepts = candidates[:7]

    strong_candidates = [
        c for c in candidates
        if _strong_word_count(c, language_code) >= 2 and not _has_mixed_scripts(c)
    ]

    supported_candidates = [
        c for c in strong_candidates
        if c in tags[:3]
    ]
    if supported_candidates:
        strong_candidates = supported_candidates

    topical = [c for c in strong_candidates if not _looks_like_named_entity(c)]
    entities = [c for c in strong_candidates if _looks_like_named_entity(c)]

    if topical and entities:
        short_parts = [topical[0], entities[0]]
    elif len(strong_candidates) >= 2:
        short_parts = strong_candidates[:2]
    elif len(strong_candidates) == 1:
        short_parts = [strong_candidates[0]]
    elif len(tags) >= 2:
        short_parts = tags[:2]
    elif len(tags) == 1:
        short_parts = [tags[0]]
    else:
        short_parts = [f"Cluster {cluster_id}"]

    name_short = " · ".join(short_parts)

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

    name_short, name_title, tags, concepts = _build_display_names(
        candidates,
        cluster_id,
        language_code,
    )

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