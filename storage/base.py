"""Abstract storage interface."""
from abc import ABC, abstractmethod
from typing import List

from scrapers.base import Article


class BaseStorage(ABC):
    """Storage interface — swap JSON for DB by implementing this."""

    @abstractmethod
    def save(self, articles: List[Article]) -> None:
        ...

    @abstractmethod
    def load(self) -> List[Article]:
        ...
