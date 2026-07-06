"""Лексический baseline: TF-IDF на scikit-learn и собственный BM25 Okapi.

Оба поиска работают по одному и тому же представлению документа — заголовок плюс
текст — и по одному анализатору из `semsearch.text`, поэтому разница в метриках
относится к модели ранжирования, а не к препроцессингу.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Document
from .retriever import BaseRetriever, SearchHit, make_snippet
from .text import analyze

# Заголовок дублируется, чтобы он весил больше, чем одно упоминание в тексте:
# в документации название страницы — самый точный её признак.
TITLE_REPEATS = 3


def document_text(document: Document, title_repeats: int = TITLE_REPEATS) -> str:
    return " ".join([document.title] * title_repeats + [document.text])


class LexicalRetrieverBase(BaseRetriever):
    def __init__(self, documents: Sequence[Document], title_repeats: int = TITLE_REPEATS):
        self.documents = list(documents)
        self.doc_ids = [d.doc_id for d in self.documents]
        self.title_repeats = title_repeats

    def _hits(self, scores: np.ndarray, top_k: int) -> list[SearchHit]:
        top_k = min(top_k, len(self.documents))
        if top_k <= 0:
            return []
        # argpartition берёт top_k без полной сортировки — на 2.5k документов это
        # незаметно, но на большом корпусе разница существенная
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


class TfidfRetriever(LexicalRetrieverBase):
    """Косинусная близость TF-IDF векторов."""

    name = "tfidf"

    def __init__(
        self,
        documents: Sequence[Document],
        title_repeats: int = TITLE_REPEATS,
        min_df: int = 2,
        max_df: float = 0.6,
        sublinear_tf: bool = True,
    ):
        super().__init__(documents, title_repeats=title_repeats)
        self.vectorizer = TfidfVectorizer(
            analyzer=analyze,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm="l2",
        )
        corpus_texts = [document_text(d, title_repeats) for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query_vector = self.vectorizer.transform([query])
        # обе матрицы нормированы по L2, поэтому скалярное произведение = косинус
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        return self._hits(scores, top_k)


class Bm25Retriever(LexicalRetrieverBase):
    """BM25 Okapi.

    Формула: score(q, d) = sum_t idf(t) * tf(t, d) * (k1 + 1) /
    (tf(t, d) + k1 * (1 - b + b * |d| / avgdl)), где
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).

    В отличие от TF-IDF, BM25 насыщает вклад частоты термина (k1) и нормирует на
    длину документа (b); в MDN длины страниц различаются в сотни раз, поэтому это
    важно.
    """

    name = "bm25"

    def __init__(
        self,
        documents: Sequence[Document],
        title_repeats: int = TITLE_REPEATS,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__(documents, title_repeats=title_repeats)
        self.k1 = k1
        self.b = b

        self._postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths = np.zeros(len(self.documents), dtype=np.float32)
        for index, document in enumerate(self.documents):
            tokens = analyze(document_text(document, title_repeats))
            self.doc_lengths[index] = len(tokens)
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, []).append((index, count))

        n_docs = max(len(self.documents), 1)
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.documents) else 0.0
        self.idf = {
            term: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }
        # знаменатель зависит только от документа, поэтому считается один раз
        self._length_norm = self.k1 * (
            1.0 - self.b + self.b * self.doc_lengths / (self.avg_doc_length or 1.0)
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        for term in analyze(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            indices = np.fromiter((i for i, _ in postings), dtype=np.int64, count=len(postings))
            freqs = np.fromiter((c for _, c in postings), dtype=np.float32, count=len(postings))
            scores[indices] += idf * freqs * (self.k1 + 1.0) / (freqs + self._length_norm[indices])
        return self._hits(scores, top_k)


LEXICAL_RETRIEVERS = {"tfidf": TfidfRetriever, "bm25": Bm25Retriever}


def build_lexical_retriever(kind: str, documents: Sequence[Document], **kwargs):
    if kind not in LEXICAL_RETRIEVERS:
        raise ValueError(f"неизвестный лексический поиск: {kind}; есть {sorted(LEXICAL_RETRIEVERS)}")
    return LEXICAL_RETRIEVERS[kind](documents, **kwargs)
