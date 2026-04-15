"""News Aggregator — entry point."""
import logging
import sys
import time

from config import Config
from scrapers import RSSScraper, HTMLScraper
from scrapers.base import Article
from processors import deduplicate
# from storage import JSONStorage
from storage.pg_storage import PGStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config()
    while True:
        all_articles: list[Article] = []

        # RSS sources
        for src in cfg.rss_sources:
            scraper = RSSScraper(src, timeout=cfg.request_timeout, user_agent=cfg.user_agent)
            all_articles.extend(scraper.fetch())

        # HTML sources
        for src in cfg.html_sources:
            scraper = HTMLScraper(src, timeout=cfg.request_timeout, user_agent=cfg.user_agent)
            all_articles.extend(scraper.fetch())

        # Process
        all_articles = deduplicate(all_articles)

        # Store
        # storage = JSONStorage(cfg.output_file)
        storage = PGStorage()
        storage.save(all_articles)
            
        logger.info("Done! Total unique articles: %d", len(all_articles))
            
        # Теперь sleep будет срабатывать после каждой итерации
        time.sleep(3600)

if __name__ == "__main__":
    main()
