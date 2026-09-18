#!/usr/bin/env python3
"""Локальная LLM → кандидаты вопросов с точной цитатой. Это ещё не gold-набор.

eval: одна страница на вопрос, 2:1 dev_v2/test_v2; страницы train не пересекаются
с любыми eval-кандидатами (включая отклонённые). Решение человека хранится отдельно.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from uuid import uuid4
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.chunking import load_chunks
from rag.ingest import load_documents
from rag.llm import make_backend
from rag.qa import load_qa

SYSTEM = """Составь один конкретный бытовой вопрос новичка по фрагменту MDN на русском.
Вопрос описывает задачу своими словами, не называет нужный API и не ссылается на текст.
Ответ должен полностью следовать из фрагмента. Не добавляй знания извне.
Верни JSON с полями question (вопрос), reference_answer (ответ в 1 предложении),
answer_terms (массив точных названий API или ключевых понятий из ответа).
Не используй заполнители вроде 'точное имя API'."""


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def parse_candidate(raw, chunk, doc, qid, split, fixed_quote=None):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("нет JSON")
    value = json.loads(match.group())
    question, reference = (value[k].strip() for k in ("question", "reference_answer"))
    quote = fixed_quote if fixed_quote is not None else value["quote"].strip()
    terms = value["answer_terms"]
    if not 15 <= len(question) <= 260 or len(reference) < 15 or len(quote) < 25:
        raise ValueError("слишком короткий или длинный текст")
    if quote not in chunk.text or not isinstance(terms, list) or not terms or not all(isinstance(t, str) and t.strip() for t in terms):
        raise ValueError("нет точной цитаты или терминов")
    if not any(t.casefold() in reference.casefold() for t in terms):
        raise ValueError("термин не упомянут в эталоне")
    return {"id": qid, "split": split, "question": question, "answerable": True,
            "reference_answer": reference, "answer_terms": terms,
            "relevant": [{"slug": doc.slug, "grade": 2}],
            "evidence": [{"slug": doc.slug, "doc_id": doc.doc_id, "quote": quote}]}


def reserved_docs(paths):
    return {e.doc_id for path in paths for item in load_qa(path) for e in item.evidence}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--purpose", choices=["eval", "train"], required=True)
    p.add_argument("--count", type=int, default=180)
    p.add_argument("--model", default="models/qwen3b/qwen2.5-3b-instruct-q4_k_m.gguf")
    p.add_argument("--server", help="локальный llama-server, например http://127.0.0.1:8091")
    p.add_argument("--chunks", default="data/processed/chunks-c1000-o150.jsonl")
    p.add_argument("--exclude", action="append", default=["data/eval/qa.jsonl"])
    p.add_argument("--reserved-docs", action="append", default=[])
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.count < 1:
        p.error("--count > 0")
    if args.purpose == "train" and len(args.exclude) < 2:
        p.error("для train укажите --exclude с новым eval-набором кандидатов")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    documents = {d.doc_id: d for d in load_documents("data/processed/documents.jsonl")}
    reserved = reserved_docs(args.exclude)
    for path in args.reserved_docs:
        reserved.update(json.loads(Path(path).read_text(encoding="utf-8")))
    chunks = [c for c in load_chunks(args.chunks) if c.doc_id not in reserved and 300 < len(c.text) < 1300
              and re.search(r"[а-яА-Я]{4}", c.text) and "```" not in c.text]
    random.Random(args.seed).shuffle(chunks)
    # Резервируем страницы ДО генерации: rejected-кандидаты тоже не попадут в train.
    by_doc = {}
    for chunk in sorted(chunks, key=lambda c: c.start):
        for paragraph in chunk.text.split("\n\n"):
            if 120 <= len(paragraph) <= 650 and len(re.findall(r"[а-яА-Я]{3,}", paragraph)) >= 12 and not any(s in paragraph for s in ("<code", "|", "{{", "\\u")):
                by_doc.setdefault(chunk.doc_id, (chunk, paragraph))
                break
    selected = list(by_doc.values())
    random.Random(args.seed).shuffle(selected)
    selected = selected[:args.count * 3]
    (output / "reserved_docs.json").write_text(json.dumps([c.doc_id for c, _ in selected]), encoding="utf-8")
    llm = make_backend({"backend": "llama_server" if args.server else "llama_cpp", "model_path": args.model,
                        "base_url": args.server, "n_ctx": 4096, "n_threads": 6})
    system = SYSTEM
    if args.purpose == "train":
        system += """\nОсобенно важно: вопрос описывает практическую задачу без имени нужного метода.
Пример: фрагмент 'Array.filter создаёт массив элементов, прошедших проверку' →
question: 'Как оставить в списке только подходящие элементы?', reference_answer:
'Метод filter создаёт новый массив элементов, прошедших проверку.', answer_terms: ['filter'].
Пример: фрагмент 'border-radius задаёт скругление углов' → question:
'Как сделать уголки карточки круглыми?', reference_answer: 'Используйте border-radius.',
answer_terms: ['border-radius']. Придумай аналогичный вопрос по данному фрагменту."""
    provenance = {"purpose": args.purpose,
                  "seed": args.seed, "generator": llm.fingerprint, "prompt": system,
                  "exclude_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.exclude},
                  "target": args.count, "review_status": "unreviewed", "answerability": "answerable only"}
    run_id = uuid4().hex[:12]
    (output / f"provenance-{run_id}.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    seen = {normalize(i.question) for path in args.exclude for i in load_qa(path)}
    questions_path, attempts_path = output / "questions.jsonl", output / "attempts.jsonl"
    accepted = [json.loads(s) for s in questions_path.read_text(encoding="utf-8").splitlines()] if questions_path.exists() else []
    attempts = [json.loads(s) for s in attempts_path.read_text(encoding="utf-8").splitlines()] if attempts_path.exists() else []
    seen.update(normalize(row["question"]) for row in accepted)
    attempted = {row["chunk_id"] for row in attempts}
    with questions_path.open("a", encoding="utf-8") as out, attempts_path.open("a", encoding="utf-8") as log:
        for chunk, paragraph in selected:
            if len(accepted) >= args.count:
                break
            if chunk.chunk_id in attempted:
                continue
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": f"{chunk.header}\n\n{paragraph}"}]
            raw = llm.json_chat(messages=messages, max_tokens=200, schema={"type": "object",
                    "properties": {"question": {"type": "string"}, "reference_answer": {"type": "string"},
                                   "answer_terms": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
                    "required": ["question", "reference_answer", "answer_terms"]})
            try:
                number = len(accepted) + 1
                split = "train" if args.purpose == "train" else ("test_v2" if number % 3 == 0 else "dev_v2")
                row = parse_candidate(raw, chunk, documents[chunk.doc_id], f"syn-{args.purpose}-{number:04}", split, paragraph)
                if normalize(row["question"]) in seen:
                    raise ValueError("повтор вопроса")
                seen.add(normalize(row["question"]))
                accepted.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                status = "candidate"
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                status = str(exc)
            log.write(json.dumps({"chunk_id": chunk.chunk_id, "raw": raw, "status": status}, ensure_ascii=False) + "\n")
            log.flush()
            print(len(accepted), args.count, chunk.title, status, flush=True)
            if len(accepted) >= args.count:
                break
    review_path = output / "review.csv"
    existing = list(csv.DictReader(review_path.open(encoding="utf-8-sig"))) if review_path.exists() else []
    reviewed_ids = {r["id"] for r in existing}
    with review_path.open("a", encoding="utf-8-sig" if not existing else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "reference_answer", "answer_terms", "quote", "accept", "reviewer", "reviewer_type", "comment"])
        if not existing:
            writer.writeheader()
        for row in accepted:
            if row["id"] in reviewed_ids:
                continue
            writer.writerow({"id": row["id"], "question": row["question"], "reference_answer": row["reference_answer"],
                             "answer_terms": json.dumps(row["answer_terms"], ensure_ascii=False), "quote": row["evidence"][0]["quote"]})
    print(f"Кандидатов: {len(accepted)}; требуется проверка review.csv", flush=True)


if __name__ == "__main__":
    main()
