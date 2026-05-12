"""News Aggregator — entry point."""
import logging
import sys
import time

from config import Config
from processors import deduplicate
from scrapers import HTMLScraper, RSSScraper
from scrapers.base import Article
from storage.pg_storage import PGStorage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def run_once() -> None:
    cfg = Config()
    all_articles: list[Article] = []

    for src in cfg.rss_sources:
        scraper = RSSScraper(
            src,
            timeout=cfg.request_timeout,
            user_agent=cfg.user_agent,
        )
        all_articles.extend(scraper.fetch())

    for src in cfg.html_sources:
        scraper = HTMLScraper(
            src,
            timeout=cfg.request_timeout,
            user_agent=cfg.user_agent,
        )
        all_articles.extend(scraper.fetch())

    all_articles = deduplicate(all_articles)

    storage = PGStorage()
    storage.save(all_articles)

    logger.info("Done! Total unique articles: %d", len(all_articles))


def main() -> None:
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("Collector iteration failed")
        time.sleep(3600)


if __name__ == "__main__":
    main()