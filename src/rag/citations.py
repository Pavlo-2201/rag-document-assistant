"""Ссылки в ответе: разбор [n] и проверка по списку переданных фрагментов."""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class CitationCheck:
    cited: list[int]       # номера в порядке первого упоминания, без повторов
    valid: list[int]       # те из них, что есть среди переданных фрагментов
    invalid: list[int]     # ссылки на несуществующие номера


def extract_citations(answer: str) -> list[int]:
    numbers: list[int] = []
    for match in CITATION_RE.finditer(answer):
        for part in re.split(r"[,;]", match.group(1)):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    cited = extract_citations(answer)
    valid = [n for n in cited if 1 <= n <= n_sources]
    return CitationCheck(cited=cited, valid=valid, invalid=[n for n in cited if n not in valid])


def strip_invalid_citations(answer: str, invalid: list[int]) -> str:
    """Убрать из текста ссылки на несуществующие номера: показывать их нельзя."""
    if not invalid:
        return answer

    def fix(match: re.Match) -> str:
        kept = [p.strip() for p in re.split(r"[,;]", match.group(2)) if int(p) not in invalid]
        return f"{match.group(1)}[{', '.join(kept)}]" if kept else ""

    # пробел перед ссылкой удаляется вместе с ней, остальной текст не трогается
    return re.sub(r"(\s*)\[(\d+(?:\s*[,;]\s*\d+)*)\]", fix, answer).strip()
