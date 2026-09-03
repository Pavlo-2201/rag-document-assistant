#!/usr/bin/env python3
"""Оценка ответов: правильность, обоснованность, ссылки и разбор ошибок по этапам.

    python scripts/run_experiments.py generation --split test --variant rerank20   # ответы
    python scripts/evaluate_answers.py --split test --variant rerank20 --judge --closed-book

Что считается (results/evaluation/<variant>__<split>/):

* автоматически: ключевой термин в ответе, ссылка на фрагмент с
  цитатой-доказательством, отказы (`rag/answer_eval.py`);
* разбор ошибок по этапам по позициям цитаты в выдаче первого этапа и в
  промпте, следы ошибок парсера в контексте (`rag/error_analysis.py`);
* --judge: LLM-судья (та же Qwen2.5-1.5B) — обоснованность ответа
  процитированными фрагментами и совпадение с эталоном; модель, промпты и
  критерии записываются в judge_config.json;
* --closed-book: та же модель без документов — чтобы увидеть, когда поиск
  улучшает ответ, а когда мешает;
* manual_review.csv — лист для ручной проверки человеком (сами ответы,
  источники, автоматические оценки и пустые колонки для вердикта).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.answer_eval import score_answer, summarize_scores  # noqa: E402
from rag.chunking import load_chunks  # noqa: E402
from rag.error_analysis import attribute, parser_artifacts, summarize_errors  # noqa: E402
from rag.evaluation import evidence_rank  # noqa: E402
from rag.ingest import load_documents  # noqa: E402
from rag.llm import make_backend  # noqa: E402
from rag.qa import load_qa, mentions_answer_term  # noqa: E402

GGUF = "models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf"
CANDIDATES = 20

CLOSED_BOOK_SYSTEM = "Ты — помощник по веб-разработке. Ответь на вопрос по-русски, 1–3 предложения."


def first_stage(variant: dict, chunks, chunks_path: Path):
    from rag.retrieval import build_bm25, load_dense

    if variant["retriever"] == "bm25":
        return build_bm25(chunks)
    return load_dense(chunks, Path("data/indexes") / f"{chunks_path.stem}__{variant['retriever']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default="baseline", help="вариант из configs/experiments.yaml")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--closed-book", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    variant = {**config["baseline"], **config["generation"]["variants"][args.variant]}
    answers_path = Path("results/experiments/generation") / f"answers__{args.variant}__{args.split}.jsonl"
    answers = {a["id"]: a for a in map(json.loads, answers_path.read_text(encoding="utf-8").splitlines())}
    output = Path("results/evaluation") / f"{args.variant}__{args.split}"
    output.mkdir(parents=True, exist_ok=True)

    documents = load_documents("data/processed/documents.jsonl")
    items = load_qa("data/eval/qa.jsonl", documents, split=args.split)
    chunks_path = Path("data/processed") / f"chunks-c{variant['chunk_size']}-o{variant['overlap']}.jsonl"
    chunks = load_chunks(chunks_path)
    chunk_text = {c.chunk_id: c.text for c in chunks}
    retriever = first_stage(variant, chunks, chunks_path)

    rows = []
    for item in items:
        answer = answers[item.id]
        score = score_answer(answer, item)
        rank = None
        if item.answerable:
            rank = evidence_rank(retriever.search(item.question, top_k=CANDIDATES), item)
        rows.append({
            **score,
            "question": item.question,
            "answer": answer["answer"],
            "first_stage_evidence_rank": rank,
            "stage": attribute(score, rank, CANDIDATES, variant["top_k"]),
            "parser_artifacts": parser_artifacts([chunk_text[c] for c in answer["context"]]),
        })

    llm = None
    judge_config = None
    if args.judge or args.closed_book:
        llm = make_backend({"backend": "llama_cpp", "model_path": GGUF})
    if args.judge:
        from rag.judge import Judge

        judge = Judge(llm)
        judge_config = judge.config()
        for row, item in zip(rows, items):
            if row["refused"]:
                continue
            cited = [chunk_text[s["chunk_id"]] for s in answers[item.id]["sources"]]
            row["judge_grounded"] = judge.groundedness(item.question, row["answer"], cited).verdict
            if item.answerable:
                row["judge_correct"] = judge.correctness(item.question, row["answer"], item.reference_answer).verdict
            print(f"{item.id}: grounded={row.get('judge_grounded')} correct={row.get('judge_correct')}", flush=True)
    if args.closed_book:
        for row, item in zip(rows, items):
            if not item.answerable:
                continue
            reply = llm.chat([{"role": "system", "content": CLOSED_BOOK_SYSTEM},
                              {"role": "user", "content": item.question}], max_tokens=200)
            row["closed_book_answer"] = reply
            row["closed_book_term_hit"] = mentions_answer_term(reply, item.answer_terms)
            if args.judge:
                row["closed_book_judge_correct"] = judge.correctness(item.question, reply, item.reference_answer).verdict
            print(f"{item.id}: closed-book term_hit={row['closed_book_term_hit']}", flush=True)

    report = {"variant": variant, "split": args.split, "answers": str(answers_path),
              "automatic": summarize_scores(rows, list(answers.values())),
              "errors": summarize_errors(rows)}
    answered = [r for r in rows if r["answerable"] and not r["refused"]]
    if args.judge:
        report["judge"] = {
            "answered_groundedness": _count(r["judge_grounded"] for r in rows if not r["refused"]),
            "answered_correctness": _count(r["judge_correct"] for r in answered),
            # согласие судьи с механическими метриками на одних и тех же ответах
            "agreement_correct_vs_term_hit": _agreement(answered, "judge_correct", "term_hit"),
            "agreement_grounded_vs_cited_evidence": _agreement(answered, "judge_grounded", "cited_evidence"),
        }
        (output / "judge_config.json").write_text(json.dumps(judge_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.closed_book:
        answerable = [r for r in rows if r["answerable"]]
        report["closed_book"] = {
            "system_prompt": CLOSED_BOOK_SYSTEM,
            "term_hit": sum(r["closed_book_term_hit"] for r in answerable),
            "rag_term_hit": sum(r["term_hit"] for r in answerable),
            "rag_only": sum(r["term_hit"] and not r["closed_book_term_hit"] for r in answerable),
            "closed_book_only": sum(r["closed_book_term_hit"] and not r["term_hit"] for r in answerable),
            "both": sum(r["term_hit"] and r["closed_book_term_hit"] for r in answerable),
            "neither": sum(not r["term_hit"] and not r["closed_book_term_hit"] for r in answerable),
        }
        if args.judge:
            report["closed_book"]["judge_correctness"] = _count(r["closed_book_judge_correct"] for r in answerable)

    (output / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_sheet(output / "manual_review.csv", rows, items, answers)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _count(values) -> dict:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _agreement(rows, judge_key: str, auto_key: str) -> dict:
    """Таблица сопряжённости: вердикт судьи × механическая метка."""
    table: dict[str, dict[str, int]] = {}
    for r in rows:
        cell = table.setdefault(r[judge_key], {"true": 0, "false": 0})
        cell["true" if r[auto_key] else "false"] += 1
    return table


def write_review_sheet(path: Path, rows, items, answers) -> None:
    """Лист для ручной проверки: колонки manual_* заполняет человек."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "reference_answer", "answer", "sources", "refused", "term_hit",
                         "cited_evidence", "stage", "judge_correct", "judge_grounded",
                         "manual_correct (да/частично/нет)", "manual_grounded (да/частично/нет)", "manual_comment"])
        for row, item in zip(rows, items):
            sources = "; ".join(f"[{s['number']}] {s['title']} {s['url']}" for s in answers[item.id]["sources"])
            writer.writerow([item.id, item.question, item.reference_answer, row["answer"], sources, row["refused"],
                             row.get("term_hit", ""), row.get("cited_evidence", ""), row["stage"],
                             row.get("judge_correct", ""), row.get("judge_grounded", ""), "", "", ""])


if __name__ == "__main__":
    raise SystemExit(main())
