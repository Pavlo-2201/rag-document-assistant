#!/usr/bin/env python3
"""Индекс FAISS по фрагментам.

    python scripts/build_index.py --chunks data/processed/chunks-c1000-o150.jsonl --model lsa-300
    python scripts/build_index.py --chunks data/processed/chunks-c1000-o150.jsonl --model e5-small
    # -> data/indexes/<chunks>__<model>/ (index.faiss, meta.json, encoder.pkl, build.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.faiss_index import IndexSpec  # noqa: E402

from rag.chunking import load_chunks  # noqa: E402
from rag.retrieval import build_dense_index, resolve_model  # noqa: E402


def index_dir_for(chunks_path: str, model: str, root: str = "data/indexes") -> Path:
    return Path(root) / f"{Path(chunks_path).stem}__{model}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--model", default="lsa-300")
    parser.add_argument("--models-config", default="configs/retrieval.yaml")
    parser.add_argument("--index-type", default="flat-ip", choices=["flat-ip", "hnsw", "ivf-flat"])
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    config = resolve_model(args.model, args.models_config)
    output = Path(args.output) if args.output else index_dir_for(args.chunks, args.model)
    stats = build_dense_index(chunks, config, output, IndexSpec(args.index_type),
                              device=args.device, precision=args.precision)
    print(json.dumps({"index": str(output), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
