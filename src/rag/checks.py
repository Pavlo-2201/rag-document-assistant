"""Проверки окружения и данных: `python -m rag check`.

Каждая проверка возвращает (название, статус, подробности); статус — ok,
fail или skip (нечего проверять: например, веса ещё не скачаны и проверка
выключена флагом). Проверяется то, что ломается незаметно: подменённые веса,
другой корпус, индекс от других фрагментов, разметка, разошедшаяся с текстом.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .app import Paths, paths_for
from .chunking import chunks_fingerprint, load_chunks
from .ingest import documents_fingerprint, load_documents


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def check_models(config: dict) -> list[tuple]:
    results = []
    for name, spec in config.get("models", {}).items():
        path = Path(spec["path"])
        if not path.exists():
            results.append((f"веса {name}", "fail", f"нет {path} (README, раздел «Модели»)"))
            continue
        actual = sha256_file(path)
        ok = actual == spec["sha256"]
        results.append((f"веса {name}", "ok" if ok else "fail", f"sha256 {actual[:12]}…" + ("" if ok else f" ≠ {spec['sha256'][:12]}…")))
    return results


def check_corpus() -> list[tuple]:
    corpus_config = yaml.safe_load(Path("configs/corpus.yaml").read_text(encoding="utf-8"))
    meta_path = Path(corpus_config["paths"]["raw_dir"]) / "meta.json"
    if not meta_path.exists():
        return [("корпус", "fail", f"нет {meta_path}: python scripts/download_corpus.py")]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    reference = corpus_config["reference_fingerprint_sha256"]
    ok = meta.get("semsearch_fingerprint_sha256") == reference
    return [("корпус = корпус semantic-search-engine", "ok" if ok else "fail",
             f"документов: {meta.get('documents')}, commit {meta['source']['ref'][:12]}")]


def check_documents(paths: Paths) -> tuple[list[tuple], list]:
    if not paths.documents.exists():
        return [("документы", "fail", f"нет {paths.documents}: python scripts/ingest.py")], []
    documents = load_documents(paths.documents)
    report_path = paths.documents.with_suffix(".report.json")
    if not report_path.exists():
        return [("отчёт ingest", "fail", f"нет {report_path}: python scripts/ingest.py")], documents
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ok = documents_fingerprint(documents) == report["fingerprint_sha256"]
    detail = f"документов: {len(documents)}" + ("" if ok else f": текст отличается от {report_path} (другой парсер?)")
    return [("документы = отчёт ingest", "ok" if ok else "fail", detail)], documents


def check_chunks_and_index(paths: Paths) -> list[tuple]:
    if not paths.chunks.exists():
        return [("фрагменты", "fail", f"нет {paths.chunks}: python -m rag build")]
    chunks = load_chunks(paths.chunks)
    results = []
    report_path = Path("data") / f"{paths.chunks.stem}.report.json"
    if report_path.exists():
        ok = chunks_fingerprint(chunks) == json.loads(report_path.read_text(encoding="utf-8"))["fingerprint_sha256"]
        results.append(("фрагменты = версия в репозитории", "ok" if ok else "fail", f"фрагментов: {len(chunks)}"))
    if paths.index is None:
        results.append(("индекс", "skip", "BM25 строится в памяти"))
        return results
    if not (paths.index / "build.json").exists():
        return results + [("индекс", "fail", f"нет {paths.index}: python -m rag build")]
    from .retrieval import load_dense

    try:
        load_dense(chunks, paths.index)     # сверяет id и отпечаток фрагментов
        results.append(("индекс собран на этих фрагментах", "ok", str(paths.index)))
    except Exception as error:  # noqa: BLE001 — показываем причину как есть
        results.append(("индекс собран на этих фрагментах", "fail", str(error)))
    return results


def check_eval_set(documents) -> list[tuple]:
    from .qa import load_qa

    try:
        items = load_qa("data/eval/qa.jsonl", documents)
    except ValueError as error:
        return [("цитаты-доказательства есть в документах", "fail", str(error))]
    quotes = sum(len(i.evidence) for i in items)
    return [("цитаты-доказательства есть в документах", "ok", f"цитат: {quotes}, вопросов: {len(items)}")]


def check_parser(documents) -> list[tuple]:
    from collections import Counter

    from .error_analysis import parser_artifacts

    kinds, docs = Counter(), 0
    for document in documents:
        found = parser_artifacts([document.text])
        kinds.update(a["kind"] for a in found)
        docs += bool(found)
    detail = ", ".join(f"{kind}: {n}" for kind, n in kinds.most_common()) or "нет"
    return [("следы ошибок парсера (информация)", "ok", f"документов со следами: {docs} из {len(documents)}; {detail}")]


def check_pipeline() -> list[tuple]:
    """Связность без весов: BM25 + mock-бэкенд, источники только из контекста."""
    from .generation import RAGPipeline
    from .llm import MockBackend
    from .retrieval import build_bm25

    paths = paths_for({"chunking": {"chunk_size": 1000, "overlap": 150}, "retriever": {"model": "bm25"}})
    if not paths.chunks.exists():
        return [("конвейер с mock-бэкендом", "skip", f"нет {paths.chunks}")]
    answer = RAGPipeline(build_bm25(load_chunks(paths.chunks)), MockBackend()).answer("что значит код ответа 404")
    ok = bool(answer.sources) and {s.chunk_id for s in answer.sources} <= set(answer.context)
    return [("конвейер с mock-бэкендом", "ok" if ok else "fail", answer.sources[0].title if answer.sources else "нет источников")]


def run_checks(config: dict, models: bool = True) -> list[tuple]:
    paths = paths_for(config)
    results = check_models(config) if models else [("веса моделей", "skip", "--skip-models")]
    results += check_corpus()
    doc_results, documents = check_documents(paths)
    results += doc_results
    results += check_chunks_and_index(paths)
    if documents:
        results += check_eval_set(documents) + check_parser(documents)
    results += check_pipeline()
    return results
