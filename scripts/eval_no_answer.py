#!/usr/bin/env python3
"""Вопросы с ответом и без ответа: отказывается ли система и не выдумывает ли источники.

Один прогон модели на split'е без порога поиска; решение порога по оценке
лучшего фрагмента детерминировано, поэтому эффект любого порога считается по
тем же ответам без повторного запуска модели. Порог выбирается только на dev.

    python scripts/eval_no_answer.py --split dev --refusal strict
    python scripts/eval_no_answer.py --split dev --refusal lenient
    python scripts/eval_no_answer.py --split dev --refusal strict --retry-empty
    python scripts/eval_no_answer.py --split test --refusal strict --threshold 0.83
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import load_chunks  # noqa: E402
from rag.generation import GenerationConfig, RAGPipeline  # noqa: E402
from rag.llm import make_backend  # noqa: E402
from rag.retrieval import load_dense  # noqa: E402


def load_questions(split: str) -> list[dict]:
    rows = []
    for line in Path(f"data/eval/sse/{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"id": row["query_id"], "question": row["query"], "answerable": True})
    for line in Path("data/eval/unanswerable.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == split:
            rows.append({"id": row["id"], "question": row["question"], "answerable": False, "kind": row["kind"]})
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decide(record: dict, threshold: float | None) -> dict:
    """Итог для вопроса при заданном пороге (поверх ответа модели без порога)."""
    gated = threshold is not None and record["top_score"] is not None and record["top_score"] < threshold
    refused = gated or record["refused"]
    # при пороге модель для такого вопроса не вызывалась бы: отказ на этапе поиска
    reason = "retrieval" if gated else record["refusal_reason"]
    return {"refused": refused, "reason": reason,
            "answered_with_sources": not refused and bool(record["sources"])}


def summarize(records: list[dict], threshold: float | None) -> dict:
    result = {}
    for answerable in (True, False):
        group = [r for r in records if r["answerable"] == answerable]
        decisions = [decide(r, threshold) for r in group]
        key = "answerable" if answerable else "unanswerable"
        reasons = {"retrieval": 0, "model": 0, "empty": 0}
        for d in decisions:
            if d["refused"]:
                reasons[d["reason"]] += 1
        answered = [r for r, d in zip(group, decisions) if not d["refused"]]
        result[key] = {
            "questions": len(group),
            "refused": sum(d["refused"] for d in decisions),
            "refused_by": reasons,
            "answered_with_sources": sum(d["answered_with_sources"] for d in decisions),
            "answered_without_sources": sum(not d["refused"] and not d["answered_with_sources"] for d in decisions),
            "answered_after_removing_refusal_sentence": sum(r.get("partial_refusal", False) for r in answered),
            "retried_after_citation_only_answer": sum(r.get("retried", False) for r in group),
        }
    # проверка «не выдумывает источники»: каждый показанный источник был в промпте
    result["sources_outside_context"] = sum(
        1 for r in records for s in r["sources"] if s["chunk_id"] not in r["context"]
    )
    result["invalid_citations_removed"] = sum(len(r["invalid_citations"]) for r in records)
    return result


def balanced_accuracy(summary: dict) -> float:
    """Средняя доля верных решений: ответ на вопросы с ответом, отказ на остальные."""
    a, u = summary["answerable"], summary["unanswerable"]
    return round(((a["questions"] - a["refused"]) / a["questions"] + u["refused"] / u["questions"]) / 2, 4)


def select_threshold(records: list[dict]) -> tuple[float | None, list[dict]]:
    """Порог по оценке лучшего фрагмента, максимизирующий balanced accuracy.

    Кандидаты — «без порога» и середины между соседними оценками; при равенстве
    берётся меньший порог (меньше лишних отказов на вопросы с ответом).
    """
    scores = sorted({r["top_score"] for r in records if r["top_score"] is not None})
    candidates = [None] + [round((x + y) / 2, 4) for x, y in zip(scores, scores[1:])]
    sweep = []
    for threshold in candidates:
        s = summarize(records, threshold)
        sweep.append({"threshold": threshold,
                      "answerable_refused": s["answerable"]["refused"],
                      "unanswerable_refused": s["unanswerable"]["refused"],
                      "balanced_accuracy": balanced_accuracy(s)})
    best = max(sweep, key=lambda row: (row["balanced_accuracy"], -(row["threshold"] or 0)))
    return best["threshold"], sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    parser.add_argument("--index", default="data/indexes/chunks-c1000-o150__e5-small")
    parser.add_argument("--model-path", default="models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--refusal", default="strict", choices=["strict", "lenient"],
                        help="правило отказа в промпте (rag/prompt.py)")
    parser.add_argument("--retry-empty", action="store_true",
                        help="переспрашивать модель, если она ответила одними номерами фрагментов")
    parser.add_argument("--threshold", type=float, help="порог, выбранный на dev (для test)")
    parser.add_argument("--output", default="results/no_answer")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variant = f"{args.split}__{args.refusal}" + ("__retry" if args.retry_empty else "")
    answers_path = output / f"answers__{variant}.jsonl"
    if answers_path.exists():
        records = read_jsonl(answers_path)
    else:
        # повтор меняет только ответы из одних ссылок; остальные ответы берутся из
        # прогона без повтора: жадная генерация детерминирована (проверено: два
        # прогона на dev дали одинаковые ответы)
        base_path = output / f"answers__{args.split}__{args.refusal}.jsonl"
        base = {r["id"]: r for r in read_jsonl(base_path)} if args.retry_empty and base_path.exists() else {}
        chunks = load_chunks(args.chunks)
        pipeline = RAGPipeline(load_dense(chunks, args.index),
                               make_backend({"backend": "llama_cpp", "model_path": args.model_path}),
                               GenerationConfig(top_k=5, refusal=args.refusal, retry_empty=args.retry_empty))
        records = []
        with answers_path.open("w", encoding="utf-8") as handle:
            for row in load_questions(args.split):
                previous = base.get(row["id"])
                if previous is not None and previous["refusal_reason"] != "empty":
                    record = previous
                else:
                    answer = pipeline.answer(row["question"])
                    record = {**row, **answer.to_dict()}
                    print(f"{row['id']}: refused={answer.refused} top={answer.top_score} {answer.answer[:70]!r}", flush=True)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {"split": args.split, "refusal_rule": args.refusal, "retry_empty": args.retry_empty, "retriever": "faiss:e5-small", "llm": records[0]["backend"] if records else "",
              "without_threshold": summarize(records, None)}
    report["without_threshold"]["balanced_accuracy"] = balanced_accuracy(report["without_threshold"])
    best, sweep = select_threshold(records)
    report["threshold_sweep"] = sweep
    if args.threshold is None:
        # подбор порога — только на dev; на test порог передаётся явно
        report["best_threshold_on_this_split"] = best
    else:
        report["threshold"] = args.threshold
        report["with_threshold"] = summarize(records, args.threshold)
        report["with_threshold"]["balanced_accuracy"] = balanced_accuracy(report["with_threshold"])
    report["scores"] = {
        "answerable": sorted(r["top_score"] for r in records if r["answerable"]),
        "unanswerable": sorted(r["top_score"] for r in records if not r["answerable"]),
    }
    (output / f"report__{variant}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("threshold_sweep", "scores")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
