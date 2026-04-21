from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import SentenceTransformer


@dataclass
class ArticleText:
    id: int
    title: str | None = None
    content: str | None = None


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def build_text(self, article: ArticleText) -> str:
        parts: list[str] = []

        if article.title:
            parts.append(article.title.strip())

        if article.content:
            parts.append(article.content.strip())

        text = "\n\n".join(part for part in parts if part)

        if not text:
            raise ValueError(f"Article {article.id} has no text for embedding")

        return text

    def encode_text(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return vector.tolist()

    def encode_article(self, article: ArticleText) -> list[float]:
        text = self.build_text(article)
        return self.encode_text(text)

    def encode_batch(
        self,
        articles: Sequence[ArticleText],
        batch_size: int = 32
    ) -> list[list[float]]:
        texts = [self.build_text(article) for article in articles]

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True
        )

        return [vector.tolist() for vector in vectors]