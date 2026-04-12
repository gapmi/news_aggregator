"""Article deduplication processor."""
import logging
from typing import List

from scrapers.base import Article

logger = logging.getLogger(__name__)


def deduplicate(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles by URL."""
    seen: set[str] = set()
    unique: List[Article] = []
    for article in articles:
        if article.url and article.url not in seen:
            seen.add(article.url)
            unique.append(article)
    removed = len(articles) - len(unique)
    if removed:
        logger.info("Deduplicator: removed %d duplicates", removed)
    return unique
