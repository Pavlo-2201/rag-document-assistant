"""Поиск по фрагментам на компонентах semantic-search-engine.

BM25, LSA, sentence-transformers и индекс FAISS берутся из пакета `semsearch`
без изменений: они там проверены тестами и измерены. Чтобы их переиспользовать,
фрагмент представляется как `semsearch.corpus.Document`, у которого `doc_id` —
это `chunk_id`, заголовок — контекст фрагмента (документ и раздел), текст —
сам фрагмент. Результат поиска возвращается к `Chunk` по `chunk_id`.

Оценка по запросам semantic-search-engine идёт на уровне документов: выдача
фрагментов сворачивается в выдачу документов (документ получает позицию своего
лучшего фрагмента). Так результаты сравнимы с поиском по страницам целиком.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from semsearch.corpus import Document, text_checksum
from semsearch.embedding import EncoderConfig, load_model_configs
from semsearch.faiss_index import IndexSpec, VectorIndex
from semsearch.lexical import Bm25Retriever
from semsearch.pipeline import build_encoder_for_corpus, load_encoder, save_encoder
from semsearch.dense import encode_corpus
from semsearch.corpus import corpus_fingerprint
from semsearch.dense import FaissDenseRetriever
from semsearch.retriever import SearchHit

from .chunking import Chunk

RRF_K = 60


@dataclass(frozen=True)
class ChunkHit:
    chunk: Chunk
    score: float
    rank: int

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


def chunk_as_document(chunk: Chunk) -> Document:
    return Document(
        doc_id=chunk.chunk_id,
        slug=chunk.chunk_id,
        title=chunk.header,
        text=chunk.text,
        url=chunk.url,
        source_path=chunk.doc_id,
        source_repo="",
        source_ref="",
        license="",
        checksum=text_checksum(chunk.embedding_text()),
        char_len=len(chunk.text),
    )


class ChunkRetriever:
    """Обёртка над поиском semsearch: на входе запрос, на выходе ChunkHit."""

    def __init__(self, chunks: Sequence[Chunk], backend, name: str):
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.backend = backend
        self.name = name

    def _convert(self, hits: Sequence[SearchHit]) -> list[ChunkHit]:
        return [ChunkHit(self.chunks_by_id[h.doc_id], float(h.score), h.rank) for h in hits]

    def search(self, query: str, top_k: int = 10) -> list[ChunkHit]:
        return self._convert(self.backend.search(query, top_k=top_k))

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[ChunkHit]]:
        return [self._convert(hits) for hits in self.backend.search_batch(list(queries), top_k=top_k)]


def rrf_merge(rankings: Sequence[Sequence[ChunkHit]], top_k: int, k: int = RRF_K) -> list[ChunkHit]:
    """Reciprocal rank fusion; один голос фрагмента от каждой выдачи."""
    if k < 1 or top_k < 0:
        raise ValueError("RRF: k >= 1, top_k >= 0")
    scores: dict[str, float] = {}
    hits: dict[str, ChunkHit] = {}
    for ranking in rankings:
        seen = set()
        for position, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + position)
            hits.setdefault(cid, hit)
    order = sorted(hits, key=lambda cid: -scores[cid])[:top_k]
    return [ChunkHit(hits[cid].chunk, scores[cid], rank) for rank, cid in enumerate(order, 1)]


class HybridRetriever:
    """BM25 + dense: RRF до общего cross-encoder, без сложения разных шкал."""

    def __init__(self, dense, lexical, depth: int = 100, k: int = RRF_K, fixed_depth: bool = False):
        if depth < 1 or k < 1:
            raise ValueError("depth и k должны быть положительными")
        self.dense, self.lexical, self.depth, self.k = dense, lexical, depth, k
        self.fixed_depth = fixed_depth
        self.name = f"{dense.name}+bm25-rrf{depth}-k{k}"

    def search(self, query: str, top_k: int = 10) -> list[ChunkHit]:
        depth = self.depth if self.fixed_depth else max(self.depth, top_k)
        return rrf_merge([self.dense.search(query, top_k=depth),
                          self.lexical.search(query, top_k=depth)], top_k, self.k)

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> list[list[ChunkHit]]:
        queries = list(queries)
        depth = self.depth if self.fixed_depth else max(self.depth, top_k)
        dense = self.dense.search_batch(queries, top_k=depth)
        lexical = self.lexical.search_batch(queries, top_k=depth)
        if len(dense) != len(queries) or len(lexical) != len(queries):
            raise ValueError("retriever вернул неверное число выдач")
        return [rrf_merge([d, b], top_k, self.k) for d, b in zip(dense, lexical)]


def resolve_model(name: str, models_config: str | Path = "configs/retrieval.yaml") -> EncoderConfig:
    configs = load_model_configs(models_config)
    if name not in configs:
        raise ValueError(f"модель '{name}' не найдена в {models_config}; есть {sorted(configs)}")
    return configs[name]


def build_bm25(chunks: Sequence[Chunk], title_repeats: int = 3) -> ChunkRetriever:
    """BM25 по фрагментам; заголовок фрагмента (документ и раздел) повторяется
    `title_repeats` раз, как в semantic-search-engine. Значение 3 выбрано на dev
    (nDCG@10 0.267 против 0.252 при 1); разница в пределах шума 36 запросов,
    поэтому оставлено значение проверенного проекта."""
    documents = [chunk_as_document(c) for c in chunks]
    return ChunkRetriever(chunks, Bm25Retriever(documents, title_repeats=title_repeats), "bm25")


def build_dense_index(
    chunks: Sequence[Chunk],
    config: EncoderConfig,
    output_dir: str | Path,
    spec: IndexSpec | None = None,
    device: str | None = None,
    precision: str = "float32",
) -> dict:
    """Закодировать фрагменты, собрать индекс FAISS и сохранить его вместе с кодировщиком."""
    documents = [chunk_as_document(c) for c in chunks]
    started = time.perf_counter()
    if precision not in ("float32", "float16"):
        raise ValueError("precision: float32 | float16")
    encoder = build_encoder_for_corpus(config, documents)
    model = getattr(encoder, "model", None)
    if device:
        if model is None:
            raise ValueError("device поддерживается только для sentence-transformers")
        model.to(device)
    if precision == "float16":
        if model is None or model.device.type != "cuda":
            raise ValueError("float16 требует CUDA sentence-transformers")
        model.half()
    vectors = encode_corpus(encoder, documents, progress=True)
    weights = Path(config.local_path) / 'model.safetensors' if config.local_path else None
    weights_sha = None
    if weights is not None and weights.exists():
        with weights.open('rb') as handle:
            weights_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    index = VectorIndex.build(vectors, [d.doc_id for d in documents], spec=spec or IndexSpec("flat-ip"),
        model={**config.to_dict(), "build_precision": precision, "weights_sha256": weights_sha},
        corpus_fingerprint=corpus_fingerprint(documents))
    seconds = time.perf_counter() - started
    output_dir = Path(output_dir)
    index.save(output_dir)
    save_encoder(encoder, output_dir)
    size = sum(p.stat().st_size for p in output_dir.iterdir())
    stats = {"model": config.name, "vectors": index.size, "dim": index.dim,
             "build_seconds": round(seconds, 1), "index_bytes": size,
             "device": str(model.device) if model is not None else "cpu", "precision": precision}
    stats['weights_sha256'] = weights_sha
    (output_dir / "build.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def load_dense(chunks: Sequence[Chunk], index_dir: str | Path) -> ChunkRetriever:
    """Открыть сохранённый индекс; проверить, что он собран на этих же фрагментах."""
    from semsearch.corpus import corpus_fingerprint

    index = VectorIndex.load(index_dir)
    documents = [chunk_as_document(c) for c in chunks]
    index.check_matches_corpus([d.doc_id for d in documents], fingerprint=corpus_fingerprint(documents))
    config = EncoderConfig.from_dict(index.model["name"], index.model)
    if index.model.get('weights_sha256') and config.local_path:
        weights = Path(config.local_path) / 'model.safetensors'
        with weights.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != index.model['weights_sha256']:
            raise ValueError('веса эмбеддера изменились после сборки индекса; пересоберите индекс')
    encoder = load_encoder(config, index_dir)
    backend = FaissDenseRetriever(documents, encoder, index)
    return ChunkRetriever(chunks, backend, f"faiss:{config.name}")


def documents_from_chunks(hits: Sequence[ChunkHit], top_k: int) -> list[SearchHit]:
    """Свернуть выдачу фрагментов в выдачу документов (позиция лучшего фрагмента)."""
    seen: dict[str, SearchHit] = {}
    for hit in hits:
        if hit.doc_id not in seen:
            seen[hit.doc_id] = SearchHit(
                doc_id=hit.doc_id, score=hit.score, rank=len(seen) + 1,
                title=hit.chunk.title, url=hit.chunk.url.split("#")[0],
            )
            if len(seen) == top_k:
                break
    return list(seen.values())
