"""Переранжирование кандидатов cross-encoder'ом.

Первый этап (векторный поиск) кодирует вопрос и фрагмент независимо и быстро
отбирает кандидатов. Cross-encoder читает пару «вопрос — фрагмент» целиком и
оценивает её точнее, но дорого: один прогон трансформера на каждую пару. Поэтому
он применяется только к top-N кандидатам первого этапа.

Модель: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 — многоязычный MiniLM,
обученный на mMARCO (MS MARCO, переведённый в том числе на русский).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Sequence

from .retrieval import ChunkHit


class CrossEncoderReranker:
    def __init__(self, model_path: str = "models/reranker", max_length: int = 512, batch_size: int = 16,
                 device: str = 'cpu'):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, device=device, max_length=max_length, local_files_only=True)
        self.batch_size = batch_size
        self.name = f"cross-encoder:{model_path.rstrip('/').rsplit('/', 1)[-1]}"

    def score(self, query: str, hits: Sequence[ChunkHit]) -> list[float]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.embedding_text()) for hit in hits]
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]

    def rerank(self, query: str, hits: Sequence[ChunkHit], top_k: int | None = None) -> list[ChunkHit]:
        scores = self.score(query, hits)
        # при равных оценках сохраняется порядок первого этапа
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        reranked = [replace(hits[i], score=scores[i], rank=rank) for rank, i in enumerate(order, start=1)]
        return reranked[:top_k] if top_k else reranked


class RerankingRetriever:
    """Двухэтапный поиск: `candidates` фрагментов первого этапа → cross-encoder → top_k."""

    def __init__(self, first_stage, reranker: CrossEncoderReranker, candidates: int = 20):
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{first_stage.name}+rerank@{candidates}"
        self.last_timings: dict = {}

    def search(self, query: str, top_k: int = 10) -> list[ChunkHit]:
        started = time.perf_counter()
        hits = self.first_stage.search(query, top_k=max(self.candidates, top_k))
        middle = time.perf_counter()
        reranked = self.reranker.rerank(query, hits, top_k=top_k)
        self.last_timings = {"first_stage_s": middle - started, "rerank_s": time.perf_counter() - middle}
        return reranked

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[ChunkHit]]:
        return [self.search(q, top_k=top_k) for q in queries]
