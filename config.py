"""Configuration for News Aggregator."""
from dataclasses import dataclass, field
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
    name: str
    url: str


@dataclass
class HTMLSource:
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