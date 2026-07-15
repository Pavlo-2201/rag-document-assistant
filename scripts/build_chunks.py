#!/usr/bin/env python3
"""Нарезка разобранных документов на фрагменты.

    python scripts/build_chunks.py --chunk-size 1000 --overlap 150
    # -> data/processed/chunks-c1000-o150.jsonl и отчёт .report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import ChunkingConfig, chunk_documents, chunks_fingerprint, write_chunks  # noqa: E402
from rag.ingest import load_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/processed/documents.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = ChunkingConfig(args.chunk_size, args.overlap)
    documents = load_documents(args.documents)
    started = time.perf_counter()
    chunks = chunk_documents(documents, config)
    seconds = time.perf_counter() - started

    output = Path(args.output_dir) / f"chunks-{config.name}.jsonl"
    write_chunks(output, chunks)
    lengths = [len(c.text) for c in chunks]
    report = {
        "config": {"chunk_size": config.chunk_size, "overlap": config.overlap},
        "documents": len(documents),
        "chunks": len(chunks),
        "chunks_per_document_median": statistics.median(
            [sum(1 for _ in g) for g in _group_counts(chunks)]
        ),
        "chunk_chars_median": int(statistics.median(lengths)),
        "chunk_chars_p95": int(sorted(lengths)[int(0.95 * (len(lengths) - 1))]),
        "chunk_chars_max": max(lengths),
        "total_chunk_chars": sum(lengths),
        "duplication_ratio": round(sum(lengths) / sum(d.char_len for d in documents), 4),
        "seconds": round(seconds, 2),
        "fingerprint_sha256": chunks_fingerprint(chunks),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _group_counts(chunks):
    from itertools import groupby

    for _, group in groupby(chunks, key=lambda c: c.doc_id):
        yield list(group)


if __name__ == "__main__":
    raise SystemExit(main())
