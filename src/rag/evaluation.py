"""Оценка поиска отдельно от генерации.

Метрики документов (Recall@K, MRR@K, nDCG@K) взяты из semsearch.evaluation —
те же определения, что в semantic-search-engine, поэтому числа сравнимы с его
таблицей. Выдача фрагментов перед подсчётом сворачивается в выдачу документов.
"""

from __future__ import annotations

import time
from typing import Sequence

from semsearch.evaluation import LabeledQuery, ndcg_at_k, recall_at_k, reciprocal_rank_at_k

from .retrieval import ChunkHit, documents_from_chunks

DOC_K_VALUES = (1, 5, 10)


def document_metrics(chunk_hits: Sequence[ChunkHit], query: LabeledQuery) -> dict[str, float]:
    doc_hits = documents_from_chunks(chunk_hits, top_k=max(DOC_K_VALUES))
    metrics = {}
    for k in DOC_K_VALUES:
        metrics[f"recall@{k}"] = recall_at_k(doc_hits, query, k)
    metrics["mrr@10"] = reciprocal_rank_at_k(doc_hits, query, 10)
    metrics["ndcg@10"] = ndcg_at_k(doc_hits, query, 10)
    return metrics


def average(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}


def evaluate_document_retrieval(retriever, queries: Sequence[LabeledQuery], depth: int = 100) -> dict:
    """Прогнать запросы, посчитать метрики документов; depth — сколько фрагментов брать."""
    started = time.perf_counter()
    batches = retriever.search_batch([q.query for q in queries], top_k=depth)
    seconds = time.perf_counter() - started
    per_query = []
    for query, hits in zip(queries, batches):
        metrics = document_metrics(hits, query)
        docs = documents_from_chunks(hits, top_k=10)
        per_query.append({
            "query_id": query.query_id,
            "query": query.query,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "relevant": dict(query.slugs),
            "top_documents": [{"doc_id": d.doc_id, "title": d.title, "grade": query.relevance.get(d.doc_id, 0)} for d in docs],
            "top_chunks": [h.chunk.chunk_id for h in hits[:5]],
        })
    return {
        "retriever": retriever.name,
        "queries": len(queries),
        "metrics": average([r["metrics"] for r in per_query]),
        "seconds_per_query": round(seconds / max(len(queries), 1), 4),
        "per_query": per_query,
    }


# --- уровень доказательств -------------------------------------------------
#
# Для RAG важна не страница, а фрагмент, в котором есть ответ: страница может
# быть найдена, а нужный абзац — нет. Фрагмент покрывает цитату-доказательство,
# если он из того же документа и содержит не меньше половины её символов
# (длинная цитата может попасть на границу двух фрагментов).

EVIDENCE_K_VALUES = (1, 3, 5, 10)
MIN_EVIDENCE_OVERLAP = 0.5


def covers(chunk, evidence, min_fraction: float = MIN_EVIDENCE_OVERLAP) -> bool:
    if chunk.doc_id != evidence.doc_id:
        return False
    overlap = min(chunk.end, evidence.end) - max(chunk.start, evidence.start)
    return overlap >= min_fraction * (evidence.end - evidence.start)


def evidence_rank(chunk_hits: Sequence[ChunkHit], item) -> int | None:
    """Позиция первого фрагмента, покрывающего любую цитату вопроса (с 1)."""
    for position, hit in enumerate(chunk_hits, start=1):
        if any(covers(hit.chunk, ev) for ev in item.evidence):
            return position
    return None


def evidence_metrics(chunk_hits: Sequence[ChunkHit], item) -> dict[str, float]:
    rank = evidence_rank(chunk_hits, item)
    metrics = {f"evidence@{k}": float(rank is not None and rank <= k) for k in EVIDENCE_K_VALUES}
    metrics["evidence_mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    return metrics


def evaluate_evidence_retrieval(retriever, items, depth: int = 10) -> dict:
    """Нашёл ли поиск фрагмент с ответом: только вопросы с цитатами-доказательствами."""
    items = [i for i in items if i.answerable and i.evidence]
    batches = retriever.search_batch([i.question for i in items], top_k=depth)
    per_query = []
    for item, hits in zip(items, batches):
        per_query.append({
            "id": item.id,
            "metrics": evidence_metrics(hits, item),
            "evidence_rank": evidence_rank(hits, item),
        })
    return {"queries": len(items), "metrics": average([r["metrics"] for r in per_query]), "per_query": per_query}
