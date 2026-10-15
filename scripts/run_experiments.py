#!/usr/bin/env python3
"""Контролируемые эксперименты: нарезка, перекрытие, модель поиска, reranker, top-k.

    python scripts/run_experiments.py retrieval --split dev --split test
    python scripts/run_experiments.py generation --split dev            # все варианты из конфига
    python scripts/run_experiments.py generation --split dev --variant top3
    python scripts/run_experiments.py retrieval --config configs/experiments-hyde.yaml         --output results/experiments-hyde --split dev --split test

Параметры — configs/experiments.yaml. Недостающие нарезки и индексы
собираются (время сборки записывается). Посчитанные строки поиска и ответы
модели сохраняются и при повторном запуске не пересчитываются.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.evaluation import load_queries  # noqa: E402

from rag.answer_eval import score_answer, summarize_scores  # noqa: E402
from rag.cache import CachedLLM  # noqa: E402
from rag.chunking import ChunkingConfig, chunk_documents, load_chunks, write_chunks  # noqa: E402
from rag.evaluation import average, document_metrics, evidence_metrics  # noqa: E402
from rag.generation import GenerationConfig, RAGPipeline  # noqa: E402
from rag.ingest import load_documents  # noqa: E402
from rag.llm import make_backend  # noqa: E402
from rag.qa import load_qa  # noqa: E402
from rag.query_rewrite import HydeRetriever  # noqa: E402
from rag.retrieval import build_bm25, build_dense_index, load_dense, resolve_model  # noqa: E402

DOCUMENTS = "data/processed/documents.jsonl"
GGUF = "models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def chunks_path(chunk_size: int, overlap: int) -> Path:
    return Path("data/processed") / f"chunks-{ChunkingConfig(chunk_size, overlap).name}.jsonl"


def ensure_chunks(documents, chunk_size: int, overlap: int):
    path = chunks_path(chunk_size, overlap)
    if not path.exists():
        write_chunks(path, chunk_documents(documents, ChunkingConfig(chunk_size, overlap)))
    return path, load_chunks(path)


def ensure_dense(chunks, path: Path, model: str) -> tuple[object, dict]:
    index_dir = Path("data/indexes") / f"{path.stem}__{model}"
    if not (index_dir / "build.json").exists():
        print(f"сборка индекса {index_dir} ...", flush=True)
        build_dense_index(chunks, resolve_model(model), index_dir)
    build = json.loads((index_dir / "build.json").read_text(encoding="utf-8"))
    return load_dense(chunks, index_dir), build


_LLM = None


def experiment_llm():
    """Одна модель на все варианты; ответы кешируются, поэтому абзацы HyDE,
    посчитанные в поиске, не генерируются заново в генерации."""
    global _LLM
    if _LLM is None:
        _LLM = CachedLLM(make_backend({"backend": "llama_cpp", "model_path": GGUF}), "data/cache/llm.sqlite")
    return _LLM


def open_retriever(spec: str, chunks, path: Path):
    """bm25 | <модель>[+hyde-concat|+hyde-rrf][+rerank<N>]"""
    build = {}
    if spec == "bm25":
        started = time.perf_counter()
        retriever = build_bm25(chunks)
        build = {"build_seconds": round(time.perf_counter() - started, 1)}
        return retriever, build
    first, _, rerank = spec.partition("+rerank")
    model, _, hyde = first.partition("+hyde-")
    retriever, build = ensure_dense(chunks, path, model)
    if hyde:
        retriever = HydeRetriever(retriever, experiment_llm(), mode=hyde)
    if rerank:
        from rag.reranker import CrossEncoderReranker, RerankingRetriever

        retriever = RerankingRetriever(retriever, CrossEncoderReranker(), candidates=int(rerank))
    return retriever, build


def evaluate_retrieval(retriever, split: str, documents, items) -> dict:
    queries = {q.query_id: q for q in load_queries(f"data/eval/sse/{split}.jsonl", documents)}
    answerable = [i for i in items if i.answerable]
    started = time.perf_counter()
    batches = retriever.search_batch([i.question for i in answerable], top_k=20)
    seconds = (time.perf_counter() - started) / len(answerable)
    docs = [document_metrics(hits, queries[i.id]) for i, hits in zip(answerable, batches)]
    evidence = [evidence_metrics(hits, i) for i, hits in zip(answerable, batches)]
    top5_chars = [sum(len(h.chunk.text) for h in hits[:5]) for hits in batches]
    return {
        "documents": average(docs),
        "evidence": average(evidence),
        "per_query_evidence": {i.id: e for i, e in zip(answerable, evidence)},
        "query_ms": round(1000 * seconds, 1),
        "top5_context_chars_mean": round(statistics.mean(top5_chars)),
    }


def retrieval_plan(config) -> list[tuple[dict, list[str]]]:
    """[(нарезка, [поиск, ...]), ...] в порядке конфига."""
    base = config["baseline"]
    plan = []
    for chunking in config["retrieval"]["chunkings"]:
        is_base = chunking["chunk_size"] == base["chunk_size"] and chunking["overlap"] == base["overlap"]
        plan.append((chunking, ["bm25", base["retriever"]] + (config["retrieval"]["extra_retrievers"] if is_base else [])))
    for extra in config["retrieval"].get("combined", []):
        plan.append(({"chunk_size": extra["chunk_size"], "overlap": extra["overlap"]}, [extra["retriever"]]))
    return plan


def run_retrieval(args, config) -> int:
    """Уже посчитанные строки <output>/retrieval.json не пересчитываются."""
    documents = load_documents(DOCUMENTS)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "retrieval.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    done = {(r["chunking"], r["retriever"], r["split"]) for r in rows}
    for chunking, specs in retrieval_plan(config):
        path = chunks_path(chunking["chunk_size"], chunking["overlap"])
        todo = [s for s in specs if any((path.stem, s, split) not in done for split in args.split)]
        if not todo:
            continue
        path, chunks = ensure_chunks(documents, chunking["chunk_size"], chunking["overlap"])
        lengths = [len(c.text) for c in chunks]
        chunk_stats = {"chunks": len(chunks), "chunk_chars_median": int(statistics.median(lengths)),
                       "duplication_ratio": round(sum(lengths) / sum(d.char_len for d in documents), 4)}
        for spec in todo:
            retriever, build = open_retriever(spec, chunks, path)
            for split in args.split:
                if (path.stem, spec, split) in done:
                    continue
                items = load_qa("data/eval/qa.jsonl", documents, split=split)
                result = evaluate_retrieval(retriever, split, documents, items)
                rows.append({"chunking": path.stem, **chunking, "retriever": spec, "split": split,
                             **chunk_stats, "build": build, **result})
                e, d = result["evidence"], result["documents"]
                print(f"{path.stem:22s} {spec:22s} {split:4s} ev@1={e['evidence@1']:.3f} ev@5={e['evidence@5']:.3f} "
                      f"ev@10={e['evidence@10']:.3f} R@5={d['recall@5']:.3f} R@10={d['recall@10']:.3f} "
                      f"{result['query_ms']} мс", flush=True)
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def run_generation(args, config) -> int:
    documents = load_documents(DOCUMENTS)
    gen = config["generation"]
    names = args.variant or list(gen["variants"])
    output = Path(args.output) / "generation"
    output.mkdir(parents=True, exist_ok=True)
    llm = None
    summary_path = output / "summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for split in args.split:
        items = load_qa("data/eval/qa.jsonl", documents, split=split)
        for name in names:
            variant = {**config["baseline"], **gen["variants"][name]}
            answers_path = output / f"answers__{name}__{split}.jsonl"
            if answers_path.exists():
                answers = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line]
            else:
                path, chunks = ensure_chunks(documents, variant["chunk_size"], variant["overlap"])
                spec = (variant["retriever"] + (f"+hyde-{variant['hyde']}" if variant.get("hyde") else "")
                        + (f"+rerank{variant['rerank']}" if variant["rerank"] else ""))
                retriever, _ = open_retriever(spec, chunks, path)
                llm = llm or experiment_llm()
                pipeline = RAGPipeline(retriever, llm, GenerationConfig(
                    top_k=variant["top_k"], max_context_tokens=gen["max_context_tokens"]))
                answers = []
                with answers_path.open("w", encoding="utf-8") as handle:
                    for item in items:
                        answer = {"id": item.id, **pipeline.answer(item.question).to_dict()}
                        answers.append(answer)
                        handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
                        print(f"{name} {item.id}: refused={answer['refused']} {answer['answer'][:60]!r}", flush=True)
            by_id = {i.id: i for i in items}
            scores = [score_answer(a, by_id[a["id"]]) for a in answers]
            summaries[f"{name}__{split}"] = {"variant": variant, "split": split,
                                             **summarize_scores(scores, answers)}
            (output / f"scores__{name}__{split}.jsonl").write_text(
                "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores), encoding="utf-8")
            print(json.dumps(summaries[f"{name}__{split}"], ensure_ascii=False), flush=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["retrieval", "generation"])
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output", default="results/experiments",
                        help="каталог результатов; готовые строки и ответы в нём не пересчитываются")
    parser.add_argument("--split", action="append")
    parser.add_argument("--variant", action="append", help="вариант генерации из конфига (по умолчанию все)")
    args = parser.parse_args()
    args.split = args.split or ["dev"]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    return run_retrieval(args, config) if args.what == "retrieval" else run_generation(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
