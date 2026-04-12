"""JSON file storage implementation."""
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

from scrapers.base import Article
from .base import BaseStorage

logger = logging.getLogger(__name__)


class JSONStorage(BaseStorage):
    """Store articles as a JSON file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)

    def save(self, articles: List[Article]) -> None:
        data = [asdict(a) for a in articles]
        self.filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d articles to %s", len(articles), self.filepath)

    def load(self) -> List[Article]:
        if not self.filepath.exists():
            return []
        data = json.loads(self.filepath.read_text(encoding="utf-8"))
        return [Article(**item) for item in data]
