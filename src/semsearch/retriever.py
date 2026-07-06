"""Общий интерфейс поиска.

Интерфейс специально узкий: запрос и top_k на входе, упорядоченный список
результатов на выходе. Так лексический и векторный поиск взаимозаменяемы в
CLI, в оценке и в следующем проекте (RAG), которому нужен только `search`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int          # с единицы: удобно для метрик и для вывода
    title: str = ""
    url: str = ""
    snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "title": self.title,
            "url": self.url,
        }


@runtime_checkable
class Retriever(Protocol):
    """Минимальный контракт поиска."""

    name: str

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        ...

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[SearchHit]]:
        ...


class BaseRetriever:
    """Реализация `search_batch` через `search` — её достаточно для baseline.

    Векторный поиск переопределит `search_batch`: кодировать запросы батчем
    существенно быстрее, чем по одному.
    """

    name: str = "base"

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:  # pragma: no cover
        raise NotImplementedError

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[SearchHit]]:
        return [self.search(query, top_k=top_k) for query in queries]


def make_snippet(text: str, max_chars: int = 200) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[:max_chars].rsplit(" ", 1)[0] + "…"
