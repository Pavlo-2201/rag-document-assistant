"""Сборка ассистента из одного конфига (configs/rag.yaml).

Конфиг описывает весь путь: корпус → нарезка → индекс → reranker → LLM →
параметры ответа. Скрипты экспериментов задают те же параметры явно; здесь
они собраны в одном месте для повседневного запуска (`python -m rag`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .cache import CachedLLM
from .chunking import ChunkingConfig, chunk_documents, load_chunks, write_chunks
from .generation import GenerationConfig, RAGPipeline
from .ingest import load_documents
from .llm import make_backend
from .retrieval import HybridRetriever, build_bm25, build_dense_index, load_dense, resolve_model


@dataclass
class Paths:
    documents: Path
    chunks: Path
    index: Path | None      # None для BM25: он строится в памяти за секунды


def load_config(path: str | Path = "configs/rag.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def paths_for(config: dict) -> Paths:
    data = Path(config.get("data_dir", "data"))
    chunking = ChunkingConfig(config["chunking"]["chunk_size"], config["chunking"]["overlap"])
    chunks = data / "processed" / f"chunks-{chunking.name}.jsonl"
    retriever = config["retriever"]["model"]
    index = None if retriever == "bm25" else data / "indexes" / f"{chunks.stem}__{retriever}"
    return Paths(data / "processed" / "documents.jsonl", chunks, index)


def build_artifacts(config: dict, log=print) -> Paths:
    """Собрать недостающие фрагменты и индекс; готовые не пересобираются."""
    paths = paths_for(config)
    if not paths.documents.exists():
        raise FileNotFoundError(
            f"нет {paths.documents}: сначала python scripts/download_corpus.py и python scripts/ingest.py")
    if not paths.chunks.exists():
        started = time.perf_counter()
        chunking = ChunkingConfig(config["chunking"]["chunk_size"], config["chunking"]["overlap"])
        write_chunks(paths.chunks, chunk_documents(load_documents(paths.documents), chunking))
        log(f"фрагменты: {paths.chunks} ({time.perf_counter() - started:.1f} с)")
    if paths.index is not None and not (paths.index / "build.json").exists():
        log(f"индекс: {paths.index} (e5-small на CPU — около 15 минут) ...")
        model = resolve_model(config["retriever"]["model"], config["retriever"].get("models_config", "configs/retrieval.yaml"))
        stats = build_dense_index(load_chunks(paths.chunks), model, paths.index)
        log(f"индекс готов за {stats['build_seconds']} с")
    return paths


def build_retriever(config: dict, paths: Paths | None = None, llm=None):
    """`llm` нужен только для переформулировки вопроса (query_rewrite.hyde)."""
    paths = paths or paths_for(config)
    chunks = load_chunks(paths.chunks)
    retriever = build_bm25(chunks) if paths.index is None else load_dense(chunks, paths.index)
    hybrid = config["retriever"].get("hybrid") or {}
    if hybrid.get("enabled"):
        if paths.index is None:
            raise ValueError("hybrid требует dense-модель")
        retriever = HybridRetriever(retriever, build_bm25(chunks),
                                    depth=hybrid.get("depth", 100), k=hybrid.get("rrf_k", 60),
                                    fixed_depth=hybrid.get("fixed_depth", False))
    hyde = (config.get("query_rewrite") or {}).get("hyde")
    if hyde:
        from .query_rewrite import HydeRetriever

        if llm is None:
            raise ValueError("query_rewrite.hyde требует LLM: build_retriever(config, llm=...)")
        retriever = HydeRetriever(retriever, llm, mode=hyde)
    rerank = config.get("reranker") or {}
    if rerank.get("candidates"):
        from .reranker import CrossEncoderReranker, RerankingRetriever

        reranker = CrossEncoderReranker(rerank.get("model_path", "models/reranker"))
        retriever = RerankingRetriever(retriever, reranker, candidates=rerank["candidates"])
    return retriever


def build_llm(config: dict, backend: str | None = None):
    llm_config = dict(config["llm"])
    if backend:
        llm_config["backend"] = backend
    llm = make_backend(llm_config)
    cache = config.get("cache", {})
    if cache.get("llm") and not llm.is_mock:
        llm = CachedLLM(llm, Path(cache.get("dir", "data/cache")) / "llm.sqlite")
    return llm


def build_pipeline(config: dict, backend: str | None = None, retriever=None) -> RAGPipeline:
    generation = GenerationConfig(**config.get("generation", {}))
    llm = build_llm(config, backend)
    return RAGPipeline(retriever or build_retriever(config, llm=llm), llm, generation)
