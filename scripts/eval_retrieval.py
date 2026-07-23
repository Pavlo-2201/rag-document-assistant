#!/usr/bin/env python3
"""Качество поиска по фрагментам на запросах semantic-search-engine (уровень документов).

    python scripts/eval_retrieval.py --chunks data/processed/chunks-c1000-o150.jsonl \
        --retriever bm25 --retriever faiss:lsa-300 --retriever faiss:e5-small --split dev --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.evaluation import load_queries  # noqa: E402

from rag.chunking import load_chunks  # noqa: E402
from rag.evaluation import evaluate_document_retrieval  # noqa: E402
from rag.ingest import load_documents  # noqa: E402
from rag.retrieval import build_bm25, load_dense  # noqa: E402


def open_retriever(spec: str, chunks, chunks_path: str):
    if spec == "bm25":
        return build_bm25(chunks)
    if spec.startswith("faiss:"):
        model = spec.split(":", 1)[1]
        return load_dense(chunks, Path("data/indexes") / f"{Path(chunks_path).stem}__{model}")
    raise ValueError(f"неизвестный поиск: {spec}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/processed/documents.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--retriever", action="append", required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--queries-dir", default="data/eval/sse")
    parser.add_argument("--output", default="results/retrieval")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    chunks = load_chunks(args.chunks)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for spec in args.retriever:
        retriever = open_retriever(spec, chunks, args.chunks)
        for split in args.split or ["dev"]:
            queries = load_queries(Path(args.queries_dir) / f"{split}.jsonl", documents)
            result = evaluate_document_retrieval(retriever, queries)
            result.update({"split": split, "chunks": Path(args.chunks).stem})
            name = f"{Path(args.chunks).stem}__{spec.replace(':', '-')}__{split}.json"
            (output / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{spec:22s} {split:5s} " + " ".join(f"{k}={v:.3f}" for k, v in result["metrics"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
