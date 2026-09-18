#!/usr/bin/env python3
"""Независимый судья и blind-лист с текстами источников; каппа по полным парам.

export сохраняет существующую разметку: повторная запись запрещена.
agree различает human/assistant и не выдаёт разметку LLM за ручную.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.chunking import load_chunks
from rag.judge import Judge
from rag.llm import make_backend
from rag.qa import load_qa
from rag.statistics import cohen_kappa


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def agreement(reviews, verdicts):
    by_id = {r["id"]: r for r in verdicts}
    if len(by_id) != len(verdicts) or len({r["id"] for r in reviews}) != len(reviews):
        raise ValueError("повтор id")
    if {r["id"] for r in reviews} != set(by_id):
        raise ValueError("лист проверки и вердикты должны покрывать один и тот же набор id")
    result = {}
    for kind in sorted({r.get("reviewer_type", "unspecified") or "unspecified" for r in reviews}):
        for metric in ("correct", "grounded"):
            pairs, missing, invalid, excluded = [], [], [], {}
            for row in reviews:
                if (row.get("reviewer_type") or "unspecified") != kind:
                    continue
                if row.get(f'exclude_{metric}'):
                    excluded[row['id']] = row[f'exclude_{metric}']
                    continue
                label = row.get(f"manual_{metric} (да/частично/нет)", "").strip().upper()
                judge = by_id.get(row["id"], {}).get(f"judge_{metric}", "")
                verdict_row = by_id.get(row["id"], {})
                if verdict_row.get(f"model_scored_{metric}", verdict_row.get("model_scored")) is False:
                    excluded[row['id']] = 'not scored by LLM: refusal or no cited fragments'
                    continue  # простые правила отказа не маскируют качество LLM-судьи
                if not label or not judge:
                    missing.append(row["id"])
                elif label not in ("ДА", "ЧАСТИЧНО", "НЕТ") or judge not in ("ДА", "ЧАСТИЧНО", "НЕТ"):
                    invalid.append(row["id"])
                else:
                    pairs.append((label, judge))
            result[f"{kind}__{metric}"] = {"reviewer_type": kind, "paired": len(pairs),
                                         "missing": missing, "invalid": invalid,
                                         "excluded": excluded,
                                         "statistics": cohen_kappa(*zip(*pairs)) if pairs else None}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["export", "judge", "agree"])
    p.add_argument("--answers", default="results/experiments/generation/answers__rerank20__test.jsonl")
    p.add_argument("--qa", default="data/eval/qa.jsonl")
    p.add_argument("--split", default="test")
    p.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    p.add_argument("--judge-config", default="configs/judge.yaml")
    p.add_argument("--review", default="results/quality/review/manual_review.csv")
    p.add_argument("--verdicts", default="results/quality/review/judge.jsonl")
    p.add_argument("--output", default="results/quality/review/agreement.json")
    args = p.parse_args()
    if args.action == "agree":
        rows = list(csv.DictReader(open(args.review, encoding="utf-8-sig")))
        result = agreement(rows, jsonl(args.verdicts))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    items = {i.id: i for i in load_qa(args.qa, split=args.split)}
    answers = jsonl(args.answers)
    if {a["id"] for a in answers} != set(items) or len(answers) != len(items):
        raise ValueError("ответы должны покрывать split целиком, без повторов")
    chunks = {c.chunk_id: c for c in load_chunks(args.chunks)}
    if args.action == "export":
        path = Path(args.review)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "question", "reference_answer", "answer", "cited_fragments",
                  "manual_correct (да/частично/нет)", "manual_grounded (да/частично/нет)",
                  "manual_comment", "reviewer", "reviewer_type"]
        with path.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for a in answers:
                i = items[a["id"]]
                reference = i.reference_answer if i.answerable else "Ожидается отказ: ответа в корпусе нет."
                fragments = "\n\n".join(f"[{s['number']}] {chunks[s['chunk_id']].embedding_text()}" for s in a["sources"])
                writer.writerow({"id": i.id, "question": i.question, "reference_answer": reference,
                                 "answer": a["answer"], "cited_fragments": fragments})
    else:
        cfg = yaml.safe_load(Path(args.judge_config).read_text(encoding="utf-8"))
        if cfg.get('style') == 'structured':
            from rag.judge_v2 import StructuredJudge
            judge_class = StructuredJudge
        else:
            judge_class = Judge
        judge = judge_class(make_backend(cfg["llm"]), max_tokens=cfg.get("max_tokens", 16))
        path = Path(args.verdicts)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**judge.config(),
                    "answers_sha256": hashlib.sha256(Path(args.answers).read_bytes()).hexdigest(),
                    "qa_sha256": hashlib.sha256(Path(args.qa).read_bytes()).hexdigest(),
                    "refusal_correctness": "deterministic; excluded from LLM-only kappa", "refusal_groundedness": "N/A"}
        path.with_suffix(".config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        with path.open("x", encoding="utf-8") as handle:
            for a in answers:
                i = items[a["id"]]
                row = {"id": i.id, "refused": a["refused"], "model_scored": not a["refused"]}
                row["model_scored_correct"] = not a["refused"]
                row["model_scored_grounded"] = not a["refused"] and bool(a["sources"])
                if a["refused"]:
                    row.update(judge_correct="НЕТ" if i.answerable else "ДА", judge_grounded="")
                else:
                    reference = i.reference_answer if i.answerable else "Следует отказаться: ответа на вопрос в корпусе нет."
                    correct = judge.correctness(i.question, a["answer"], reference)
                    grounded = judge.groundedness(i.question, a["answer"], [chunks[s["chunk_id"]].text for s in a["sources"]])
                    row.update(judge_correct=correct.verdict, judge_grounded=grounded.verdict,
                               correct_raw=correct.raw, grounded_raw=grounded.raw)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(row["id"], row.get("judge_correct"), row.get("judge_grounded"), flush=True)


if __name__ == "__main__":
    main()
