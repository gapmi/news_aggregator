"""Base scraper interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Article:
    """Unified article model."""
    title: str
    url: str
    source: str
    description: str = ""
    published: Optional[str] = None
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    @abstractmethod
    def fetch(self) -> List[Article]:
        """Fetch articles from the source."""
        ...
