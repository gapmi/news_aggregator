"""Configuration for News Aggregator."""
from dataclasses import dataclass, field
from typing import List


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
    output_file: str = "news_output.json"
    request_timeout: int = 10
    user_agent: str = "NewsAggregator/1.0"

    rss_sources: List[RSSSource] = field(default_factory=lambda: [
        RSSSource(name="BBC News", url="http://feeds.bbci.co.uk/news/rss.xml"),
        RSSSource(name="Reuters", url="https://www.rss-bridge.org/bridge01/?action=display&bridge=FilterBridge&url=https%3A%2F%2Fwww.reuters.com&format=Atom"),
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
