"""Оценочный набор вопросов: ответ-эталон и доказательства (evidence spans).

Файл data/eval/qa.jsonl:

* вопросы с ответом — запросы semantic-search-engine (тот же id и разметка
  страниц) плюс эталонный ответ и цитаты-доказательства: точные подстроки текста
  документа, в которых содержится ответ;
* вопросы без ответа — из data/eval/unanswerable.jsonl (слова, которых нет в
  корпусе, перечислены в `absent_terms`).

Цитата хранится текстом, а не смещениями: смещения пересчитываются по текущим
документам, и если парсер изменит текст так, что цитаты не станет, загрузка
упадёт, а не начнёт молча мерить мимо.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .ingest import SourceDocument


@dataclass
class Evidence:
    slug: str
    doc_id: str
    quote: str
    start: int = -1      # границы цитаты в SourceDocument.text
    end: int = -1


@dataclass
class QAItem:
    id: str
    split: str
    question: str
    answerable: bool
    reference_answer: str = ""
    # ключевые термины ответа (любой из): API, свойство, понятие; в вопросе их нет
    answer_terms: list[str] = field(default_factory=list)
    relevant: list[dict] = field(default_factory=list)     # [{"slug", "grade"}] из semantic-search-engine
    evidence: list[Evidence] = field(default_factory=list)
    kind: str = ""                                         # для вопросов без ответа
    absent_terms: list[str] = field(default_factory=list)


def mentions_answer_term(text: str, terms: list[str]) -> bool:
    """Упоминает ли ответ хотя бы один ключевой термин (без учёта регистра).

    Грубая автоматическая проверка правильности: «length» в ответе на вопрос о
    числе элементов массива. Совпадение ищется с начала слова, чтобы основа
    «лексическ» находила «лексического», а «sort» не находился внутри «resort».
    """
    return any(re.search(r"(?<!\w)" + re.escape(t), text, re.IGNORECASE) for t in terms)


def load_qa(
    path: str | Path,
    documents: Iterable[SourceDocument] | None = None,
    split: str | None = None,
) -> list[QAItem]:
    """Прочитать набор; если переданы документы, найти в них каждую цитату."""
    texts = {d.doc_id: d.text for d in documents} if documents is not None else None
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split and row["split"] != split:
            continue
        evidence = [Evidence(**e) for e in row.pop("evidence", [])]
        item = QAItem(**row, evidence=evidence)
        if texts is not None:
            for ev in item.evidence:
                if ev.doc_id not in texts:
                    raise ValueError(f"{item.id}: документа {ev.slug} нет в корпусе")
                ev.start = texts[ev.doc_id].find(ev.quote)
                if ev.start < 0:
                    raise ValueError(f"{item.id}: цитаты нет в тексте {ev.slug}: {ev.quote[:60]!r}")
                ev.end = ev.start + len(ev.quote)
        items.append(item)
    return items
