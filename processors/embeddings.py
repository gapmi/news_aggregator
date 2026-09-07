from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Sequence


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_TEXT_VERSION = "title-rss-v1"
MAX_CONTEXT_CHARS = 700


@dataclass
class ArticleText:
    id: int
    title: str | None = None
    description: str | None = None
    content: str | None = None


def normalize_text(value: str | None) -> str:
    text = unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class EmbeddingService:
    _model = None

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.model_name = model_name

    @classmethod
    def _get_model(cls, model_name: str):
        if cls._model is None:
            print(f"Loading embedding model: {model_name}")
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(model_name)
        return cls._model

    def build_text(self, article: ArticleText) -> str:
        title = normalize_text(article.title)
        description = normalize_text(article.description)
        content = normalize_text(article.content)

        context = description or content[:MAX_CONTEXT_CHARS]

        if title and context:
            return f"Заголовок: {title}\nКонтекст: {context[:MAX_CONTEXT_CHARS]}"

        if title:
            return f"Заголовок: {title}"

        if context:
            return f"Контекст: {context[:MAX_CONTEXT_CHARS]}"

        raise ValueError(f"Article {article.id} has no text for embedding")

    def encode_text(self, text: str) -> list[float]:
        model = self._get_model(self.model_name)
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def encode_article(self, article: ArticleText) -> list[float]:
        return self.encode_text(self.build_text(article))

    def encode_batch(
        self,
        articles: Sequence[ArticleText],
        batch_size: int = 8,
    ) -> list[list[float]]:
        texts = [self.build_text(article) for article in articles]
        model = self._get_model(self.model_name)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]