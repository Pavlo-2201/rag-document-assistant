"""Набор запросов с разметкой и метрики качества поиска.

Определения метрик (K — глубина, по которой считается метрика):

* **Recall@K** — доля релевантных документов запроса, попавших в первые K
  результатов: `|relevant ∩ top-K| / |relevant|`. Усредняется по запросам
  (macro-average). Показывает, сколько нужного поиск вообще достаёт.
* **MRR@K** — средний обратный ранг первого релевантного документа:
  `mean(1 / rank_первого_релевантного)`, причём если релевантных в первых K нет,
  вклад запроса равен нулю. Показывает, насколько высоко стоит первый полезный
  ответ.
* **nDCG@K** — нормированный дисконтированный накопленный выигрыш с градуированной
  релевантностью: `DCG@K = Σ_{i=1..K} (2^grade_i − 1) / log2(i + 1)`, где
  `grade_i` — оценка релевантности документа на позиции i (0, если документ не
  размечен), а нормировка — на `IDCG@K`, то есть на DCG идеального порядка
  размеченных документов. Учитывает и порядок, и разную степень релевантности.

Градации релевантности в разметке: 2 — страница прямо отвечает на запрос, 1 —
страница по теме и полезна, но отвечает частично. Неразмеченные документы
считаются нерелевантными (grade 0).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .corpus import Document, make_doc_id
from .retriever import SearchHit

DEFAULT_K_VALUES = (1, 5, 10)


@dataclass(frozen=True)
class LabeledQuery:
    query_id: str
    query: str
    #: doc_id -> градация релевантности (1 или 2)
    relevance: Mapping[str, int]
    #: slug'и в том виде, в котором они записаны в файле разметки
    slugs: Mapping[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def relevant_ids(self) -> set[str]:
        return {doc_id for doc_id, grade in self.relevance.items() if grade > 0}


def load_queries(
    path: str | Path,
    documents: Sequence[Document] | None = None,
) -> list[LabeledQuery]:
    """Прочитать файл разметки.

    В файле запросы ссылаются на страницы по slug'у: он читаем человеком и не
    меняется при повторной загрузке корпуса. doc_id считается из slug'а той же
    функцией, что и при сборке корпуса, поэтому разметка и корпус не могут
    разойтись незаметно. Если передан корпус, проверяется, что все размеченные
    страницы в нём есть.
    """
    known_ids = {d.doc_id for d in documents} if documents is not None else None
    queries: list[LabeledQuery] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            record = json.loads(line)
            query_id = record["query_id"]
            if query_id in seen_ids:
                raise ValueError(f"повторяющийся query_id: {query_id}")
            seen_ids.add(query_id)

            slugs: dict[str, int] = {}
            relevance: dict[str, int] = {}
            for item in record["relevant"]:
                slug = item["slug"]
                grade = int(item.get("grade", 2))
                if grade not in (1, 2):
                    raise ValueError(f"градация должна быть 1 или 2, строка {line_number}")
                doc_id = make_doc_id(slug)
                if known_ids is not None and doc_id not in known_ids:
                    raise ValueError(
                        f"запрос {query_id}: страницы '{slug}' нет в корпусе "
                        "(проверьте slug или пересоберите корпус)"
                    )
                slugs[slug] = grade
                relevance[doc_id] = grade
            if not relevance:
                raise ValueError(f"запрос {query_id} без размеченных страниц")
            queries.append(
                LabeledQuery(
                    query_id=query_id,
                    query=record["query"],
                    relevance=relevance,
                    slugs=slugs,
                    note=record.get("note", ""),
                )
            )
    if not queries:
        raise ValueError(f"в файле {path} нет запросов")
    return queries


def recall_at_k(hits: Sequence[SearchHit], query: LabeledQuery, k: int) -> float:
    relevant = query.relevant_ids
    if not relevant:
        return 0.0
    found = {hit.doc_id for hit in hits[:k]} & relevant
    return len(found) / len(relevant)


def reciprocal_rank_at_k(hits: Sequence[SearchHit], query: LabeledQuery, k: int) -> float:
    relevant = query.relevant_ids
    for position, hit in enumerate(hits[:k], start=1):
        if hit.doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(hits: Sequence[SearchHit], query: LabeledQuery, k: int) -> float:
    def gain(grade: int) -> float:
        return (2.0**grade) - 1.0

    dcg = sum(
        gain(query.relevance.get(hit.doc_id, 0)) / math.log2(position + 1)
        for position, hit in enumerate(hits[:k], start=1)
    )
    ideal_grades = sorted(query.relevance.values(), reverse=True)[:k]
    idcg = sum(
        gain(grade) / math.log2(position + 1)
        for position, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / idcg if idcg else 0.0


@dataclass
class QueryResult:
    query_id: str
    query: str
    metrics: dict[str, float]
    retrieved: list[dict]
    relevant_slugs: dict[str, int]
    first_relevant_rank: int | None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "metrics": self.metrics,
            "first_relevant_rank": self.first_relevant_rank,
            "relevant": self.relevant_slugs,
            "retrieved": self.retrieved,
        }


def evaluate_queries(
    retriever,
    queries: Sequence[LabeledQuery],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    keep_top: int = 10,
) -> tuple[dict[str, float], list[QueryResult]]:
    """Посчитать метрики по набору запросов.

    Возвращает усреднённые метрики и результат по каждому запросу: подробности
    нужны для разбора провальных запросов.
    """
    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(max(k_values), keep_top)
    batch = retriever.search_batch([q.query for q in queries], top_k=max_k)

    per_query: list[QueryResult] = []
    totals: dict[str, float] = {}
    for query, hits in zip(queries, batch):
        metrics: dict[str, float] = {}
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(hits, query, k)
            metrics[f"mrr@{k}"] = reciprocal_rank_at_k(hits, query, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(hits, query, k)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value

        relevant = query.relevant_ids
        first_rank = next((hit.rank for hit in hits if hit.doc_id in relevant), None)
        per_query.append(
            QueryResult(
                query_id=query.query_id,
                query=query.query,
                metrics={name: round(value, 4) for name, value in metrics.items()},
                retrieved=[
                    {
                        "rank": hit.rank,
                        "doc_id": hit.doc_id,
                        "title": hit.title,
                        "score": round(float(hit.score), 6),
                        "grade": query.relevance.get(hit.doc_id, 0),
                    }
                    for hit in hits[:keep_top]
                ],
                relevant_slugs=dict(query.slugs),
                first_relevant_rank=first_rank,
            )
        )

    n = len(queries)
    averaged = {name: round(value / n, 4) for name, value in totals.items()}
    return averaged, per_query
