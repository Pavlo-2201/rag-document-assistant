"""Точный векторный поиск.

Начинается с точного поиска (полный перебор, numpy): он даёт верхнюю границу
качества для выбранной модели, и относительно него потом измеряется потеря от
приближённого индекса FAISS.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .corpus import Document
from .embedding import TextEncoder, build_document_text
from .retriever import BaseRetriever, SearchHit, make_snippet


def encode_corpus(
    encoder: TextEncoder,
    documents: Sequence[Document],
    progress: bool = False,
) -> np.ndarray:
    texts = [
        build_document_text(d.title, d.text, encoder.config.document_max_chars)
        for d in documents
    ]
    if progress:
        print(f"кодирование {len(texts)} документов моделью {encoder.config.name}…")
    vectors = encoder.encode_documents(texts)
    if vectors.shape[0] != len(documents):
        raise RuntimeError("кодировщик вернул не столько векторов, сколько документов")
    return np.ascontiguousarray(vectors, dtype=np.float32)


class ExactDenseRetriever(BaseRetriever):
    """Поиск по косинусной близости полным перебором.

    Векторы нормированы, поэтому скалярное произведение равно косинусу.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        encoder: TextEncoder,
        embeddings: np.ndarray | None = None,
    ):
        self.documents = list(documents)
        self.encoder = encoder
        self.name = f"dense-exact:{encoder.config.name}"
        self.embeddings = (
            encode_corpus(encoder, self.documents) if embeddings is None else embeddings
        )
        if self.embeddings.shape[0] != len(self.documents):
            raise ValueError("число векторов не совпадает с числом документов")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def _hits_from_scores(self, scores: np.ndarray, top_k: int) -> list[SearchHit]:
        top_k = min(top_k, len(self.documents))
        if top_k <= 0:
            return []
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        hits = []
        for rank, index in enumerate(candidates, start=1):
            document = self.documents[index]
            hits.append(
                SearchHit(
                    doc_id=document.doc_id,
                    score=float(scores[index]),
                    rank=rank,
                    title=document.title,
                    url=document.url,
                    snippet=make_snippet(document.text),
                )
            )
        return hits

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[SearchHit]]:
        query_vectors = self.encoder.encode_queries(list(queries))
        scores = query_vectors @ self.embeddings.T
        return [self._hits_from_scores(row, top_k) for row in scores]


class FaissDenseRetriever(BaseRetriever):
    """Поиск по сохранённому индексу FAISS.

    Документы нужны только для метаданных результата (заголовок, ссылка,
    фрагмент); ранжирование целиком выполняет индекс. Соответствие позиций
    индекса и doc_id хранится в самом индексе, поэтому порядок документов в
    корпусе на результат не влияет.
    """

    def __init__(self, documents: Sequence[Document], encoder: TextEncoder, index):
        self.documents_by_id = {d.doc_id: d for d in documents}
        self.encoder = encoder
        self.index = index
        self.name = f"faiss-{index.spec.index_type}:{encoder.config.name}"
        unknown = set(index.doc_ids) - set(self.documents_by_id)
        if unknown:
            raise ValueError(
                f"в индексе есть {len(unknown)} doc_id, которых нет в корпусе; "
                "индекс собран на другом корпусе"
            )

    @property
    def dim(self) -> int:
        return int(self.index.dim)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[SearchHit]]:
        query_vectors = self.encoder.encode_queries(list(queries))
        scores, doc_id_rows = self.index.search(query_vectors, top_k)
        results = []
        for score_row, doc_ids in zip(scores, doc_id_rows):
            hits = []
            for rank, (score, doc_id) in enumerate(zip(score_row, doc_ids), start=1):
                document = self.documents_by_id[doc_id]
                hits.append(
                    SearchHit(
                        doc_id=doc_id,
                        score=float(score),
                        rank=rank,
                        title=document.title,
                        url=document.url,
                        snippet=make_snippet(document.text),
                    )
                )
            results.append(hits)
        return results
