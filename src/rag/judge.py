"""LLM-судья: обоснованность ответа фрагментами и совпадение с эталоном.

Судья — та же локальная Qwen2.5-1.5B-Instruct, что генерирует ответы (другой
модели в окружении нет, платные API не используются). Поэтому его оценки — не
истина, а ещё один шумный сигнал: модель оценивает в том числе свои же ответы,
а маленькая модель плохо различает «частично» и «да». В отчёте вердикты судьи
сравниваются с механическими метриками (`answer_eval`), а модель, промпты и
критерии сохраняются рядом с результатами (`judge_config`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERDICTS = ("ДА", "ЧАСТИЧНО", "НЕТ")

GROUNDEDNESS_CRITERIA = (
    "ДА — каждое утверждение ответа прямо следует из фрагментов; "
    "ЧАСТИЧНО — часть утверждений есть во фрагментах, часть нет; "
    "НЕТ — ответ не опирается на фрагменты или противоречит им."
)
CORRECTNESS_CRITERIA = (
    "ДА — ответ по смыслу совпадает с эталоном и отвечает на вопрос; "
    "ЧАСТИЧНО — ответ верен лишь отчасти или неполон; "
    "НЕТ — ответ неверен, не отвечает на вопрос или противоречит эталону."
)

GROUNDEDNESS_SYSTEM = (
    "Ты проверяешь ответ помощника по документации. Даны фрагменты документации и ответ. "
    "Оцени только одно: подтверждается ли ответ этими фрагментами. Собственные знания не используй. "
    f"Критерии: {GROUNDEDNESS_CRITERIA} Ответь одним словом: ДА, ЧАСТИЧНО или НЕТ."
)
CORRECTNESS_SYSTEM = (
    "Ты сравниваешь ответ помощника с эталонным ответом на тот же вопрос. "
    f"Критерии: {CORRECTNESS_CRITERIA} Ответь одним словом: ДА, ЧАСТИЧНО или НЕТ."
)

_VERDICT_RE = re.compile(r"\b(ДА|ЧАСТИЧНО|НЕТ)\b", re.IGNORECASE)


@dataclass
class Verdict:
    verdict: str          # ДА | ЧАСТИЧНО | НЕТ | ? (не удалось разобрать)
    raw: str


def parse_verdict(raw: str) -> Verdict:
    match = _VERDICT_RE.search(raw.replace("ё", "е"))
    return Verdict(match.group(1).upper() if match else "?", raw)


def groundedness_messages(question: str, answer: str, fragments: list[str]) -> list[dict]:
    body = "\n\n".join(f"Фрагмент {i}:\n{text}" for i, text in enumerate(fragments, start=1))
    user = f"{body}\n\nВопрос: {question}\n\nОтвет помощника: {answer}\n\nПодтверждается ли ответ фрагментами? ДА, ЧАСТИЧНО или НЕТ?"
    return [{"role": "system", "content": GROUNDEDNESS_SYSTEM}, {"role": "user", "content": user}]


def correctness_messages(question: str, answer: str, reference: str) -> list[dict]:
    user = (f"Вопрос: {question}\n\nЭталонный ответ: {reference}\n\nОтвет помощника: {answer}\n\n"
            "Совпадает ли ответ помощника с эталоном по смыслу? ДА, ЧАСТИЧНО или НЕТ?")
    return [{"role": "system", "content": CORRECTNESS_SYSTEM}, {"role": "user", "content": user}]


class Judge:
    def __init__(self, llm, max_tokens: int = 8):
        self.llm = llm
        self.max_tokens = max_tokens

    def groundedness(self, question: str, answer: str, fragments: list[str]) -> Verdict:
        if not fragments:
            # ответ без ссылок не на что проверять: он не подтверждён по определению
            return Verdict("НЕТ", "")
        return parse_verdict(self.llm.chat(groundedness_messages(question, answer, fragments), self.max_tokens))

    def correctness(self, question: str, answer: str, reference: str) -> Verdict:
        return parse_verdict(self.llm.chat(correctness_messages(question, answer, reference), self.max_tokens))

    def config(self) -> dict:
        return {
            "model": getattr(self.llm, "name", ""),
            # файл весов, версия llama-cpp-python и параметры жадного декодирования
            "backend": getattr(self.llm, "fingerprint", ""),
            "max_tokens": self.max_tokens,
            "groundedness": {"system": GROUNDEDNESS_SYSTEM, "criteria": GROUNDEDNESS_CRITERIA,
                             "user_template": groundedness_messages("{question}", "{answer}", ["{fragment}"])[1]["content"]},
            "correctness": {"system": CORRECTNESS_SYSTEM, "criteria": CORRECTNESS_CRITERIA,
                            "user_template": correctness_messages("{question}", "{answer}", "{reference}")[1]["content"]},
        }
