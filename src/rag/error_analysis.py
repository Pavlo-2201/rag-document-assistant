"""Разбор ошибок по этапам конвейера: парсер, поиск, переранжирование, генерация.

Этап ошибки определяется механически, по цитатам-доказательствам
(`data/eval/qa.jsonl`) и сохранённым выдачам:

* `retrieval` — абзаца с ответом нет среди кандидатов первого этапа (top-N);
* `ranking` — абзац среди кандидатов был, но не попал в top-k, переданные
  модели (с reranker'ом — это его ошибка, без него — ошибка порядка первого
  этапа);
* `generation` — абзац с ответом был в промпте, а модель отказалась или
  ответила без ключевого термина;
* для вопроса без ответа ошибка всегда `generation`: модель ответила, хотя
  должна была отказаться (порог поиска на dev не помог, см. «Нет ответа»).

Ошибки парсера ищутся отдельно, по следам в тексте фрагментов, которые
видела модель (`parser_artifacts`):

* `html_entity` — нераскодированная HTML-сущность (`&lt;div&gt;`, `&#x20;`);
* `tex` — исходник формулы TeX из MathML-разметки вместо самой формулы;
* `backslash` — одиночный `\\` вне формулы: остаток удалённого тега,
  перенос строки Markdown или экранированная кавычка в аргументе макроса;
* `empty_code` — пустой инлайн-код (тег внутри него удалён).

Такой след не обязательно стал причиной ошибки, поэтому он записывается
флагом рядом с этапом, а не заменяет его.

Метка «верно» — ключевой термин в ответе (`term_hit`); её неточность
наследуется и сюда, поэтому разбор показывает, где искать, а не выносит
приговор каждому ответу.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

ARTIFACT_PATTERNS = {
    "html_entity": re.compile(r"&(?:lt|gt|amp|quot|apos|nbsp|#x?[0-9a-fA-F]+);"),
    "tex": re.compile(r"\\(?:begin|end|frac|mathrm|mathtt|text|left|right|operatorname)\{?"),
    # «\» перед пробелом, знаком препинания или концом строки, но не «\\»
    "backslash": re.compile(r"(?<![\\\w])\\(?=[\s,.)]|$)", re.MULTILINE),
    "empty_code": re.compile(r"(?<!`)``(?!`)"),
}
CODE_BLOCK_RE = re.compile(r"^\s*```.*?^\s*```", re.MULTILINE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def parser_artifacts(texts: Sequence[str]) -> list[dict]:
    """Следы ошибок разбора Markdown в текстах фрагментов (вне блоков кода).

    Одиночный «\\» в строке с формулой TeX — разделитель строк матрицы, а не
    отдельный след: формула уже отмечена как `tex`.
    """
    found = []
    for text in texts:
        text = INLINE_CODE_RE.sub("·", CODE_BLOCK_RE.sub(" ", text))
        for line in text.split("\n"):
            for kind, pattern in ARTIFACT_PATTERNS.items():
                if kind == "backslash" and ARTIFACT_PATTERNS["tex"].search(line):
                    continue
                for match in pattern.finditer(line):
                    found.append({"kind": kind, "context": line[max(0, match.start() - 30):match.end() + 30]})
                    if kind == "tex":
                        break       # одна формула — один след
    return found


def attribute(score: dict, first_stage_rank: int | None, candidates: int, top_k: int) -> str:
    """Этап, на котором потерян ответ; `ok` — ошибки нет.

    `first_stage_rank` — позиция первого фрагмента с цитатой в выдаче первого
    этапа (None — нет среди `candidates`).
    """
    if not score["answerable"]:
        return "ok" if score["refused"] else "generation"
    if score["term_hit"]:
        return "ok"
    if score["evidence_in_context"]:
        return "generation"
    if first_stage_rank is None or first_stage_rank > candidates:
        return "retrieval"
    return "ranking"


def summarize_errors(rows: Sequence[dict]) -> dict:
    stages = Counter(r["stage"] for r in rows)
    return {
        "by_stage": dict(stages),
        "answerable_by_stage": dict(Counter(r["stage"] for r in rows if r["answerable"])),
        "unanswerable_by_stage": dict(Counter(r["stage"] for r in rows if not r["answerable"])),
        "with_parser_artifacts_in_context": sum(bool(r["parser_artifacts"]) for r in rows),
        "errors_with_parser_artifacts_in_context": sum(bool(r["parser_artifacts"]) and r["stage"] != "ok" for r in rows),
        "parser_artifacts_by_kind": dict(Counter(a["kind"] for r in rows for a in r["parser_artifacts"])),
    }
