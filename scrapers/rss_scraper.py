"""RSS feed scraper."""
import logging
from typing import List

import feedparser
import requests

from .base import Article, BaseScraper
from config import RSSSource

logger = logging.getLogger(__name__)


class RSSScraper(BaseScraper):
    """Scraper for RSS/Atom feeds."""

    def __init__(self, source: RSSSource, timeout: int = 10, user_agent: str = "") -> None:
        self.source = source
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self) -> List[Article]:
        """Fetch and parse an RSS feed."""
        articles: List[Article] = []
        try:
            headers = {"User-Agent": self.user_agent} if self.user_agent else {}
            response = requests.get(self.source.url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                articles.append(Article(
                    title=entry.get("title", "No title"),
                    url=entry.get("link", ""),
                    source=self.source.name,
                    description=entry.get("summary", ""),
                    published=entry.get("published", None),
                ))
            logger.info("RSS [%s]: collected %d articles", self.source.name, len(articles))
        except requests.RequestException as e:
            logger.error("RSS [%s]: network error — %s", self.source.name, e)
        except Exception as e:
            logger.error("RSS [%s]: unexpected error — %s", self.source.name, e)
        return articles
