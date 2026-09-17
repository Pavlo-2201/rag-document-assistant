#!/usr/bin/env python3
"""Опубликовать только одобренные вопросы и сохранить происхождение разметки."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.ingest import load_documents
from rag.qa import load_qa
from generate_synthetic import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--output", default="data/eval/qa-v2.jsonl")
    p.add_argument("--minimum", type=int, default=150)
    args = p.parse_args()
    docs = load_documents("data/processed/documents.jsonl")
    items = load_qa(args.candidates, docs)
    source = [json.loads(s) for s in Path(args.candidates).read_text(encoding="utf-8").splitlines()]
    reviews = list(csv.DictReader(open(args.review, encoding="utf-8-sig")))
    if len({r["id"] for r in reviews}) != len(reviews) or len({i.id for i in items}) != len(items):
        raise ValueError("повтор id")
    indexed = {r["id"]: r for r in reviews}
    accepted, seen, doc_splits = [], set(), {}
    for row, item in zip(source, items):
        review = indexed.get(item.id, {})
        if review.get("accept", "").lower() not in ("yes", "да", "1"):
            continue
        if not review.get("reviewer") or review.get("reviewer_type") not in ("human", "assistant"):
            raise ValueError(f"{item.id}: нет происхождения разметки")
        # Вопрос и эталон в review можно исправлять. Цитата остаётся привязанной к корпусу.
        for field in ("question", "reference_answer"):
            if review.get(field):
                row[field] = review[field].strip()
        if review.get("answer_terms"):
            terms = json.loads(review["answer_terms"])
            if not isinstance(terms, list) or (item.answerable and not terms) or not all(isinstance(t, str) and t for t in terms):
                raise ValueError(f"{item.id}: неверные answer_terms")
            row["answer_terms"] = terms
        if review.get('quote'):
            if len(item.evidence)!=1:
                raise ValueError(f'{item.id}: исправление quote поддержано для одной цитаты')
            document=next(d for d in docs if d.doc_id==item.evidence[0].doc_id)
            if review['quote'] not in document.text:
                raise ValueError(f'{item.id}: исправленной цитаты нет в документе')
            row['evidence'][0]['quote']=review['quote']
        key = normalize(row["question"])
        if key in seen:
            raise ValueError(f"{item.id}: повтор вопроса")
        seen.add(key)
        for ev in item.evidence:
            if ev.doc_id in doc_splits and doc_splits[ev.doc_id] != item.split:
                raise ValueError(f"{item.id}: страница в двух split")
            doc_splits[ev.doc_id] = item.split
        accepted.append(row)
    if len(accepted) < args.minimum:
        raise ValueError(f"одобрено {len(accepted)}, нужно {args.minimum}")
    out = Path(args.output)
    with out.open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
    metadata = {"accepted": len(accepted), "reviewers": sorted({r.get("reviewer", "") for r in reviews if r.get("accept", "").lower() in ("yes", "да", "1")}),
                "reviewer_types": sorted({r.get("reviewer_type", "") for r in reviews if r.get("accept", "").lower() in ("yes", "да", "1")}),
                "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.candidates, args.review, args.output)}}
    out.with_suffix(".provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
