"""HTML page scraper."""
import logging
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .base import Article, BaseScraper
from config import HTMLSource

logger = logging.getLogger(__name__)


class HTMLScraper(BaseScraper):
    """Scraper for HTML pages using CSS selectors."""

    def __init__(self, source: HTMLSource, timeout: int = 10, user_agent: str = "") -> None:
        self.source = source
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self) -> List[Article]:
        """Fetch and parse an HTML page."""
        articles: List[Article] = []
        try:
            headers = {"User-Agent": self.user_agent} if self.user_agent else {}
            response = requests.get(self.source.url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for block in soup.select(self.source.article_selector):
                title_tag = block.select_one(self.source.title_selector)
                link_tag = block.select_one(self.source.link_selector)
                desc_tag = block.select_one(self.source.description_selector) if self.source.description_selector else None

                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                url = urljoin(self.source.url, link_tag.get("href", "")) if link_tag else ""
                description = desc_tag.get_text(strip=True) if desc_tag else ""

                articles.append(Article(
                    title=title,
                    url=url,
                    source=self.source.name,
                    description=description,
                ))
            logger.info("HTML [%s]: collected %d articles", self.source.name, len(articles))
        except requests.RequestException as e:
            logger.error("HTML [%s]: network error — %s", self.source.name, e)
        except Exception as e:
            logger.error("HTML [%s]: parse error — %s", self.source.name, e)
        return articles
