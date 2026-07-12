#!/usr/bin/env python3
"""Разбор загруженного корпуса: data/raw/mdn-ru → data/processed/documents.jsonl.

    python scripts/ingest.py --config configs/corpus.yaml
    python scripts/ingest.py --input-dir my_docs/ --output data/processed/my_docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingest import documents_fingerprint, ingest_directory, ingest_mdn, write_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corpus.yaml")
    parser.add_argument("--input-dir", help="каталог с .md/.txt вместо корпуса MDN")
    parser.add_argument("--base-url", help="адрес, от которого строятся ссылки на файлы каталога")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.input_dir:
        documents = ingest_directory(args.input_dir, base_url=args.base_url)
        output = Path(args.output or "data/processed/documents.jsonl")
    else:
        documents = ingest_mdn(config["paths"]["raw_dir"], config["source"])
        output = Path(args.output or config["paths"]["documents"])

    count = write_documents(output, documents)
    lengths = [d.char_len for d in documents]
    report = {
        "documents": count,
        "total_chars": sum(lengths),
        "median_chars": int(statistics.median(lengths)) if lengths else 0,
        "max_chars": max(lengths, default=0),
        "sections": sum(len(d.sections) for d in documents),
        "documents_with_code": sum("```" in d.text for d in documents),
        "fingerprint_sha256": documents_fingerprint(documents),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
