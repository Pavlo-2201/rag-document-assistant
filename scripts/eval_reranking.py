#!/usr/bin/env python3
"""Retrieval до и после reranking на запросах semantic-search-engine.

Для каждого первого этапа (BM25, e5-small) берутся top-N фрагментов и
переранжируются cross-encoder'ом; метрики страниц считаются по выдаче до и
после. Порядок кандидатов первого этапа одинаковый, поэтому разница относится
только к reranker'у. Записывается и время reranking'а на запрос.

    python scripts/eval_reranking.py --split dev --split test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.evaluation import load_queries  # noqa: E402

from rag.chunking import load_chunks  # noqa: E402
from rag.evaluation import average, document_metrics  # noqa: E402
from rag.ingest import load_documents  # noqa: E402
from rag.reranker import CrossEncoderReranker  # noqa: E402
from rag.retrieval import build_bm25, load_dense  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--split", action="append")
    parser.add_argument("--candidates", type=int, action="append", help="размер top-N для reranking'а")
    parser.add_argument("--output", default="results/reranking")
    args = parser.parse_args()

    documents = load_documents("data/processed/documents.jsonl")
    chunks = load_chunks(args.chunks)
    first_stages = {
        "bm25": build_bm25(chunks),
        "faiss:e5-small": load_dense(chunks, Path("data/indexes") / f"{Path(args.chunks).stem}__e5-small"),
    }
    reranker = CrossEncoderReranker()
    candidate_sizes = args.candidates or [20, 50]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in args.split or ["dev"]:
        queries = load_queries(f"data/eval/sse/{split}.jsonl", documents)
        for stage_name, retriever in first_stages.items():
            candidates = retriever.search_batch([q.query for q in queries], top_k=max(candidate_sizes))
            before = [document_metrics(hits, q) for hits, q in zip(candidates, queries)]
            rows.append({"split": split, "first_stage": stage_name, "reranked": False, "candidates": max(candidate_sizes),
                         "metrics": average(before)})
            for size in candidate_sizes:
                after, seconds, per_query = [], [], []
                for hits, query in zip(candidates, queries):
                    started = time.perf_counter()
                    reranked = reranker.rerank(query.query, hits[:size])
                    seconds.append(time.perf_counter() - started)
                    metrics = document_metrics(reranked, query)
                    after.append(metrics)
                    per_query.append({"query_id": query.query_id, "before": document_metrics(hits[:size], query),
                                      "after": metrics})
                rows.append({
                    "split": split, "first_stage": stage_name, "reranked": True, "candidates": size,
                    "metrics": average(after),
                    "before_same_candidates": average([p["before"] for p in per_query]),
                    "rerank_ms_p50": round(1000 * statistics.median(seconds), 1),
                    "rerank_ms_max": round(1000 * max(seconds), 1),
                    "per_query": per_query,
                })
                m = rows[-1]["metrics"]
                print(f"{split} {stage_name}+rerank@{size}: R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
                      f"R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f} "
                      f"({rows[-1]['rerank_ms_p50']} мс)", flush=True)
            m = rows[-1 - len(candidate_sizes)]["metrics"]
            print(f"{split} {stage_name}: R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
                  f"R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f}", flush=True)

    (output / "reranking.json").write_text(json.dumps({"reranker": reranker.name, "runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
