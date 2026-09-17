"""Автоматические метрики ответа по оценочному набору data/eval/qa.jsonl.

Все проверки механические и воспроизводимые, без LLM-судьи:

* `evidence_in_context` — среди фрагментов промпта был фрагмент с
  цитатой-доказательством (поиск принёс ответ модели);
* `cited_evidence` — ответ сослался на такой фрагмент (ссылка действительно
  ведёт туда, где написан ответ);
* `term_hit` — ответ упоминает ключевой термин эталона (`answer_terms`): грубая
  проверка правильности, она не ловит неверный ответ с правильным словом и
  верный ответ другими словами;
* для вопросов без ответа — отказ и ответ со ссылками (выдумка с опорой на
  нерелевантные фрагменты).

Покрытие цитаты фрагментом считается так же, как в оценке поиска
(`evaluation.covers`): тот же документ и не меньше половины символов цитаты.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from .evaluation import covers
from .qa import QAItem, mentions_answer_term


@dataclass
class ChunkSpan:
    doc_id: str
    start: int
    end: int


def span_from_chunk_id(chunk_id: str) -> ChunkSpan:
    """`<doc_id>:<start>-<end>` → границы фрагмента в документе."""
    doc_id, _, span = chunk_id.rpartition(":")
    start, end = span.split("-")
    return ChunkSpan(doc_id, int(start), int(end))


def _has_evidence(chunk_ids: Sequence[str], item: QAItem) -> bool:
    if any(ev.start < 0 or ev.end <= ev.start for ev in item.evidence):
        raise ValueError('для метрик evidence загрузите QA с documents: нужны разрешённые смещения цитат')
    return any(covers(span_from_chunk_id(c), ev) for c in chunk_ids for ev in item.evidence)


def score_answer(answer: dict, item: QAItem) -> dict:
    """Метрики одного ответа (`Answer.to_dict()`) на вопрос из набора."""
    refused = answer["refused"]
    cited = [s["chunk_id"] for s in answer["sources"]]
    row = {
        "id": item.id,
        "answerable": item.answerable,
        "refused": refused,
        "refusal_reason": answer["refusal_reason"],
        "answered_with_sources": not refused and bool(cited),
        "unsupported": not refused and not cited,
    }
    if item.answerable:
        row["evidence_in_context"] = _has_evidence(answer["context"], item)
        row["cited_evidence"] = not refused and _has_evidence(cited, item)
        row["term_hit"] = not refused and mentions_answer_term(answer["answer"], item.answer_terms)
    return row


def summarize_scores(rows: Sequence[dict], answers: Sequence[dict] = ()) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    n = max(len(answerable), 1)
    summary = {
        "answerable": {
            "questions": len(answerable),
            "answered": sum(not r["refused"] for r in answerable),
            "answered_with_sources": sum(r["answered_with_sources"] for r in answerable),
            "term_hit": sum(r["term_hit"] for r in answerable),
            "evidence_in_context": sum(r["evidence_in_context"] for r in answerable),
            "cited_evidence": sum(r["cited_evidence"] for r in answerable),
            # поиск принёс ответ, а модель отказалась — ошибка генерации
            "refused_with_evidence_in_context": sum(r["refused"] and r["evidence_in_context"] for r in answerable),
            # ответа в контексте не было, а модель ответила — риск выдумки
            "answered_without_evidence_in_context": sum(not r["refused"] and not r["evidence_in_context"] for r in answerable),
        },
        "unanswerable": {
            "questions": len(unanswerable),
            "refused": sum(r["refused"] for r in unanswerable),
            "answered_with_sources": sum(r["answered_with_sources"] for r in unanswerable),
            "answered_without_sources": sum(r["unsupported"] for r in unanswerable),
        },
    }
    summary["answerable"]["term_hit_rate"] = round(summary["answerable"]["term_hit"] / n, 4)
    summary["answerable"]["cited_evidence_rate"] = round(summary["answerable"]["cited_evidence"] / n, 4)
    if answers:
        generation = [a["timings"].get("generation_s", 0) + a["timings"].get("retry_s", 0)
                      for a in answers if "generation_s" in a["timings"]]
        retrieval = [a["timings"].get("retrieval_s", 0) for a in answers]
        summary["cost"] = {
            "generation_s_median": round(statistics.median(generation), 2) if generation else None,
            "retrieval_s_median": round(statistics.median(retrieval), 3) if retrieval else None,
            "context_tokens_mean": round(statistics.mean(a["context_tokens"] for a in answers), 1),
            "chunks_dropped_for_limit": sum(len(a.get("dropped_for_limit", [])) for a in answers),
            "llm_calls": sum(("generation_s" in a["timings"]) + ("retry_s" in a["timings"]) for a in answers),
        }
    return summary
