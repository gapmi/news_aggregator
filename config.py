"""Configuration for News Aggregator."""
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
ARCHIVE_DIR = PROJECT_ROOT / "dev_tempFile"


def _archive_old_output_files() -> None:
    """Move previous current_news_output_*.json to dev_tempFile/ renamed as news_output_*.json."""
    ARCHIVE_DIR.mkdir(exist_ok=True)

    for old_path in PROJECT_ROOT.glob("current_news_output_*.json"):
        new_name = old_path.name.replace("current_news_output_", "news_output_", 1)
        dest_path = ARCHIVE_DIR / new_name

        # Если файл с таким именем уже есть — добавляем суффикс
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = ARCHIVE_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        old_path.replace(dest_path)


def get_output_filename() -> str:
    """
    Move any existing current_news_output_*.json to dev_tempFile/ as news_output_*.json,
    then return a new current_news_output_<datetime>.json filepath.
    """
    _archive_old_output_files()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(PROJECT_ROOT / f"current_news_output_{timestamp}.json")


@dataclass
class RSSSource:
    """RSS feed source configuration."""
    name: str
    url: str


@dataclass
class HTMLSource:
    """HTML scraping source configuration."""
    name: str
    url: str
    article_selector: str
    title_selector: str
    link_selector: str
    description_selector: str = ""


@dataclass
class Config:
    """Main application configuration."""
    output_file: str = field(default_factory=get_output_filename)
    request_timeout: int = 10
    user_agent: str = "NewsAggregator/1.0"

    rss_sources: List[RSSSource] = field(default_factory=lambda: [
        RSSSource(name="BBC News", url="http://feeds.bbci.co.uk/news/rss.xml"),
        RSSSource(name="Reuters", url="https://feeds.reuters.com/reuters/topNews"),
        RSSSource(name="The Guardian", url="https://www.theguardian.com/world/rss"),
        RSSSource(name="Al Jazeera", url="https://www.aljazeera.com/xml/rss/all.xml"),
        RSSSource(name="Fox News", url="https://moxie.foxnews.com/google-publisher/world.xml"),
        RSSSource(name="The Economist", url="https://www.economist.com/the-world-this-week/rss.xml"),
        RSSSource(name="SCMP (Asia)", url="https://www.scmp.com/rss/91/feed"),
        RSSSource(name="Deutsche Welle", url="https://rss.dw.com/atom/rss-en-all"),
        RSSSource(name="France 24", url="https://www.france24.com/en/rss"),
        RSSSource(name="Foreign Policy", url="https://foreignpolicy.com/feed/"),
        RSSSource(name="Project Syndicate", url="https://www.project-syndicate.org/rss")
    ])

    html_sources: List[HTMLSource] = field(default_factory=lambda: [
        HTMLSource(
            name="Hacker News",
            url="https://news.ycombinator.com/",
            article_selector=".athing",
            title_selector=".titleline > a",
            link_selector=".titleline > a",
            description_selector="",
        ),
    ])