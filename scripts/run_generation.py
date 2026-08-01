#!/usr/bin/env python3
"""Прогнать RAG по списку вопросов и сохранить ответы с источниками (JSONL).

    python scripts/run_generation.py --questions data/eval/sse/dev.jsonl --limit 10 \
        --retriever faiss:e5-small --output results/generation/stage4-dev-sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import load_chunks  # noqa: E402
from rag.generation import GenerationConfig, RAGPipeline  # noqa: E402
from rag.llm import make_backend  # noqa: E402
from rag.retrieval import build_bm25, load_dense  # noqa: E402


def read_questions(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({"id": row.get("query_id") or row.get("id"), "question": row.get("question") or row["query"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--retriever", default="faiss:e5-small")
    parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "mock"])
    parser.add_argument("--model-path", default="models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if args.retriever == "bm25":
        retriever = build_bm25(chunks)
    else:
        retriever = load_dense(chunks, Path("data/indexes") / f"{Path(args.chunks).stem}__{args.retriever.split(':', 1)[1]}")
    llm = make_backend({"backend": args.backend, "model_path": args.model_path})
    pipeline = RAGPipeline(retriever, llm, GenerationConfig(args.top_k, args.max_context_tokens))

    questions = read_questions(args.questions)[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as handle:
        for row in questions:
            answer = pipeline.answer(row["question"])
            record = {"id": row["id"], "retriever": retriever.name, **answer.to_dict()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{row['id']}: {answer.timings} {answer.answer[:100]!r}", flush=True)
    print(f"всего {len(questions)} вопросов за {time.perf_counter() - started:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
