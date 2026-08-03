#!/usr/bin/env python3
"""Задать вопрос по документации.

    python scripts/ask.py "как узнать длину массива"
    python scripts/ask.py --backend mock --retriever bm25 "что значит код 404"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import load_chunks  # noqa: E402
from rag.generation import GenerationConfig, RAGPipeline, format_answer  # noqa: E402
from rag.llm import make_backend  # noqa: E402
from rag.retrieval import build_bm25, load_dense  # noqa: E402

DEFAULT_GGUF = "models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--retriever", default="faiss:e5-small", help="bm25 или faiss:<модель>")
    parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "mock"])
    parser.add_argument("--model-path", default=DEFAULT_GGUF)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--min-score", type=float, default=None,
                        help="порог близости лучшего фрагмента (косинус e5); ниже — отказ без вызова модели")
    parser.add_argument("--refusal", default="strict", choices=["strict", "lenient"], help="правило отказа в промпте")
    parser.add_argument("--no-retry-empty", action="store_true",
                        help="не переспрашивать модель, если она ответила одними номерами фрагментов")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if args.retriever == "bm25":
        retriever = build_bm25(chunks)
    else:
        model = args.retriever.split(":", 1)[1]
        retriever = load_dense(chunks, Path("data/indexes") / f"{Path(args.chunks).stem}__{model}")
    llm = make_backend({"backend": args.backend, "model_path": args.model_path})
    config = GenerationConfig(top_k=args.top_k, max_context_tokens=args.max_context_tokens,
                              min_retrieval_score=args.min_score, refusal=args.refusal,
                              retry_empty=not args.no_retry_empty)
    pipeline = RAGPipeline(retriever, llm, config)
    answer = pipeline.answer(args.question)
    print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2) if args.json else format_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
