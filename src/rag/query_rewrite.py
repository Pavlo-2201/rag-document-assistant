"""Переформулировка вопроса перед поиском (HyDE: hypothetical document embeddings).

Главное узкое место поиска (docs/evaluation.md, «Ошибки по этапам»): вопросы
заданы бытовыми словами («рисовать графику прямо в браузере»), а ответ лежит на
странице, названной по API (`<canvas>`). Та же Qwen без документов часто знает
нужный термин (17 из 36 test-вопросов по ключевому термину). Поэтому до поиска
модель пишет короткий гипотетический абзац документации, и ищется уже он.

Абзац может быть неверным: он нужен только для поиска. В промпт ответа идут
найденные фрагменты, а cross-encoder сравнивает их с исходным вопросом, а не с
абзацем модели.

Режимы:

* `concat` — вопрос и абзац одним запросом к векторному поиску;
* `rrf` — два поиска (вопрос и абзац) и слияние выдач по Reciprocal Rank
  Fusion: фрагмент, найденный по вопросу, не вытесняется ошибкой абзаца.
"""

from __future__ import annotations

import time
from typing import Sequence

from .retrieval import ChunkHit

MODES = ("concat", "rrf")
RRF_K = 60  # константа RRF из статьи Cormack et al. (2009); на dev не подбиралась

# Первая версия перечисляла «HTML-элементы, CSS-свойства, методы JavaScript»,
# и модель называла что-то из каждого пункта («<span> … text-break … split()»);
# лимит 64 токена: нужный термин модель пишет в первом предложении, а код после
# него только удлиняет генерацию (до 15 с при 128 токенах). На первых 6 dev-вопросах
# нужный термин назван в 6 абзацах против 5 у первой версии; промпт больше не подбирался.
HYDE_SYSTEM = (
    "Ты — автор документации MDN Web Docs. На вопрос пользователя напиши одно-два предложения "
    "из документации, которые на него отвечают, и назови точное название подходящего "
    "элемента, свойства, метода или API. Пиши по-русски, без кода."
)


def hyde_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": HYDE_SYSTEM}, {"role": "user", "content": question.strip()}]


def rrf_merge(rankings: Sequence[Sequence[ChunkHit]], top_k: int, k: int = RRF_K) -> list[ChunkHit]:
    """Слить выдачи: сумма 1 / (k + позиция). При равенстве — порядок первой выдачи."""
    scores: dict[str, float] = {}
    hits: dict[str, ChunkHit] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            hits.setdefault(chunk_id, hit)
    order = sorted(hits, key=lambda cid: -scores[cid])[:top_k]
    return [ChunkHit(hits[cid].chunk, scores[cid], rank) for rank, cid in enumerate(order, start=1)]


class HydeRetriever:
    """Первый этап поиска с гипотетическим абзацем; интерфейс как у ChunkRetriever."""

    def __init__(self, first_stage, llm, mode: str = "concat", max_tokens: int = 64, depth: int = 50):
        if mode not in MODES:
            raise ValueError(f"режим HyDE '{mode}' неизвестен; есть {MODES}")
        self.first_stage = first_stage
        self.llm = llm
        self.mode = mode
        self.max_tokens = max_tokens
        self.depth = depth          # глубина каждой выдачи перед слиянием (rrf)
        self.name = f"{first_stage.name}+hyde-{mode}"
        self.last_expansion = ""
        self.last_timings: dict = {}

    def expand(self, question: str) -> str:
        return self.llm.chat(hyde_messages(question), max_tokens=self.max_tokens).strip()

    def search(self, query: str, top_k: int = 10) -> list[ChunkHit]:
        started = time.perf_counter()
        expansion = self.expand(query)
        middle = time.perf_counter()
        self.last_expansion = expansion
        if not expansion:            # модель ничего не написала — обычный поиск по вопросу
            hits = self.first_stage.search(query, top_k=top_k)
        elif self.mode == "concat":
            hits = self.first_stage.search(f"{query.strip()}\n{expansion}", top_k=top_k)
        else:
            depth = max(self.depth, top_k)
            rankings = [self.first_stage.search(query, top_k=depth), self.first_stage.search(expansion, top_k=depth)]
            hits = rrf_merge(rankings, top_k)
        self.last_timings = {"hyde_s": middle - started, "search_s": time.perf_counter() - middle}
        return hits

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[ChunkHit]]:
        return [self.search(q, top_k=top_k) for q in queries]
