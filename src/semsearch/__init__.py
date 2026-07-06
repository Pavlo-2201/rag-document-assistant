"""Семантический поиск по русскоязычной технической документации.

Публичный интерфейс пакета — то, что нужно внешнему коду (в том числе
следующему проекту с RAG): корпус, результат поиска и функция, которая открывает
готовый индекс.

    from semsearch import open_retriever

    retriever = open_retriever("data/corpus/mdn-ru.jsonl", "data/indexes/lsa-300-flat")
    for hit in retriever.search("как узнать длину массива", top_k=5):
        print(hit.rank, hit.title, hit.url)

Контракт поиска описан в `semsearch.retriever.Retriever`: `search(query, top_k)`
и `search_batch(queries, top_k)` возвращают упорядоченные `SearchHit`.
"""

from __future__ import annotations

from pathlib import Path

from .corpus import Document, load_corpus
from .retriever import Retriever, SearchHit

__version__ = "1.0.0"

__all__ = [
    "Document",
    "Retriever",
    "SearchHit",
    "load_corpus",
    "open_retriever",
]


def open_retriever(
    corpus_path: str | Path,
    index_dir: str | Path | None = None,
    kind: str = "faiss",
    model: str | None = None,
    models_config: str | Path = "configs/models.yaml",
) -> Retriever:
    """Открыть поиск по корпусу.

    По умолчанию загружает готовый индекс FAISS из `index_dir` — это самый
    быстрый путь: корпус не кодируется заново. Если индекса нет, можно собрать
    поиск на месте: `kind="bm25"` или `kind="dense"` с `model`.
    """
    from .pipeline import build_retriever

    documents = load_corpus(corpus_path)
    return build_retriever(
        kind,
        documents,
        model=model,
        models_config=models_config,
        index_dir=index_dir,
    )
