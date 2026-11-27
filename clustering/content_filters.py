from __future__ import annotations

GAMBLING_PROMO_PATTERNS = (
    r"\bpromo(?:tional)?\s+codes?\b",
    r"\b(?:bonus|free)\s+bets?\b",
    r"\b(?:sign[\s-]?up|welcome|new[\s-]?user|first[\s-]?deposit)"
    r"\s+(?:bonus|offer|promo)\w*",
    r"\b(?:bet365|draftkings|fanduel|betmgm|espn\s+bet|"
    r"fanatics\s+sportsbook|caesars\s+sportsbook)\b"
    r".{0,120}\b(?:bonus|offer|promo|code|deposit|odds\s+boost)\b",
    r"\b(?:best|top)\s+(?:online\s+)?"
    r"(?:sportsbooks?|betting\s+(?:sites|apps)|casinos?)\b",
    r"\bc[oó]digo\s+promocional\b",
    r"\bbono\s+de\s+bienvenida\b",
    r"\bapuestas?\b.{0,80}\b(?:bono|promoci[oó]n|c[oó]digo)\b",
    r"\bпромокод\w*\b",
    r"\bбонус\w*\s+(?:за\s+регистрацию|новым\s+игрокам)\b",
    r"\bбукмекер\w*\b.{0,80}\b(?:бонус|промокод)\b",
)

GAMBLING_TOPIC_PATTERNS = (
    r"\b(?:online\s+)?casinos?\b",
    r"\bsportsbooks?\b",
    r"\bbetting\s+(?:sites?|apps?|odds)\b",
    r"\bcasinos?\s+en\s+l[ií]nea\b",
    r"\btragamonedas\b",
    r"\b(?:онлайн[\s-]*)?казино\b",
)

MATERIAL_GAMBLING_NEWS_PATTERNS = (
    r"\b(?:shooting|robbery|arrest\w*|fraud|fined?|fines|lawsuit|"
    r"banned?|licen[sc]e|regulator\w*|legislation|tax|closure|"
    r"bankrupt\w*|killed|police)\b",
    r"\b(?:ограблен\w*|арест\w*|штраф\w*|закон\w*|суд\w*|"
    r"запрет\w*|лиценз\w*|полици\w*)\b",
    r"\b(?:robo|arresto|multa|ley|licencia|polic[ií]a|clausura)\b",
)

GAMBLING_PROMO_COMBINED = "|".join(
    f"(?:{pattern})" for pattern in GAMBLING_PROMO_PATTERNS
)