from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from processors.embeddings import EmbeddingService, ArticleText
from .definitions import SCALES, ScaleDef


class ScaleEmbeddingService:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self._scale_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        self._init_scale_vectors()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        # используем уже существующий сервис
        vectors = [
            self.embedding_service.encode_text(t)
            for t in texts
        ]
        return np.array(vectors, dtype="float32")

    def _init_scale_vectors(self):
        for scale in SCALES:
            left_vecs = self._encode_texts(scale.left_anchors)
            right_vecs = self._encode_texts(scale.right_anchors)
            left_center = left_vecs.mean(axis=0)
            right_center = right_vecs.mean(axis=0)
            axis_vec = right_center - left_center
            axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-8)

            self._scale_vectors[scale.id] = {
                "left_center": left_center,
                "right_center": right_center,
                "axis": axis_vec,
            }

    def score_article_embedding(self, article_vec: List[float]) -> List[Dict[str, Any]]:
        v = np.array(article_vec, dtype="float32")
        v = v / (np.linalg.norm(v) + 1e-8)

        results: List[Dict[str, Any]] = []
        for scale in SCALES:
            data = self._scale_vectors[scale.id]
            axis = data["axis"]
            # косинус проекции на ось — значение шкалы
            score = float(np.dot(v, axis).clip(-1.0, 1.0))
            strength = abs(score)
            results.append({
                "scale_id": scale.id,
                "label": scale.label,
                "left_label": scale.left_label,
                "right_label": scale.right_label,
                "score": score,
                "strength": strength,
            })

        return results

    def score_article_text(self, article: ArticleText, vector: List[float]) -> List[Dict[str, Any]]:
        # вариант, если у тебя уже есть embedding (vector)
        return self.score_article_embedding(vector)