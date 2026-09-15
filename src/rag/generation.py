"""Ответ по найденному контексту: поиск → промпт с лимитом → LLM → ответ и источники."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .citations import CITATION_RE, check_citations, strip_invalid_citations
from .llm import LLMBackend
from .prompt import NO_ANSWER_TEXT, RETRY_EMPTY, build_prompt

_CITE = r"\[\d+(?:\s*[,;]\s*\d+)*\]"
# предложение с фразой отказа (с ссылками до и после него); оно убирается
# целиком, например «К сожалению, в документах нет ответа на этот вопрос [2].»
_REFUSAL_RE = re.compile(
    rf"(?:^|(?<=[.!?\n]))[^.!?\n]*в документах нет ответа[^.!?\n]*[.!?]?\s*»?(?:\s*{_CITE})*",
    re.IGNORECASE | re.MULTILINE,
)


def has_content(text: str) -> bool:
    """В ответе есть что-то кроме ссылок и знаков препинания."""
    return bool(re.search(r"\w", CITATION_RE.sub(" ", text)))


@dataclass
class ModelReply:
    """Как понят ответ модели: отказ, ответ или ответ с лишней фразой отказа."""

    text: str
    refusal: str | None        # model — только фраза отказа; empty — только ссылки
    partial_refusal: bool      # был текст по делу и фраза отказа; фраза убрана


def read_reply(raw: str) -> ModelReply:
    """Отделить фразу отказа от содержательной части ответа.

    Маленькая модель иногда отвечает по одному фрагменту и добавляет отказ
    про другой («[2] В документах нет ответа…»). Отказом считается ответ, в
    котором кроме фразы отказа и ссылок ничего нет; иначе фраза отказа
    убирается вместе со своими ссылками, а остальное остаётся ответом.
    """
    found = bool(_REFUSAL_RE.search(raw))
    rest = _REFUSAL_RE.sub("", raw) if found else raw
    if not has_content(rest):
        return ModelReply(NO_ANSWER_TEXT, "model" if found else "empty", False)
    if found:
        rest = re.sub(r"[ \t]+\n", "\n", rest)
        rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
    return ModelReply(rest, None, found)


def is_refusal(text: str) -> bool:
    """Ответ модели целиком — отказ (фраза отказа и, возможно, ссылки)."""
    return read_reply(text).refusal == "model"


@dataclass
class Source:
    number: int          # номер [n] в ответе
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str             # ссылка на раздел документа
    retrieval_rank: int
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]                   # только реально процитированные фрагменты
    context: list[str]                      # chunk_id всех фрагментов, переданных модели
    invalid_citations: list[int] = field(default_factory=list)
    dropped_for_limit: list[str] = field(default_factory=list)
    refused: bool = False
    # retrieval — поиск не нашёл ничего достаточно близкого, модель не вызывалась;
    # model — модель ответила, что в документах ответа нет; empty — в ответе
    # модели нет текста, только ссылки
    refusal_reason: str | None = None
    # модель дала ответ, но не сослалась ни на один фрагмент: подтверждения нет
    unsupported: bool = False
    # в ответе по делу была и фраза отказа; она убрана из показанного ответа
    partial_refusal: bool = False
    # первый ответ был из одних ссылок, и модель спросили ещё раз; raw_answer — второй ответ
    retried: bool = False
    first_raw_answer: str = ""
    raw_answer: str = ""
    top_score: float | None = None
    backend: str = ""
    is_mock: bool = False
    timings: dict = field(default_factory=dict)
    context_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationConfig:
    top_k: int = 5                    # сколько фрагментов брать из поиска
    max_context_tokens: int = 2000    # лимит токенов на фрагменты в промпте
    max_answer_tokens: int = 300
    # порог близости лучшего фрагмента: ниже него ответ не генерируется.
    # None — порог не применяется. Шкала зависит от поиска (косинус у e5,
    # не ограниченная сверху оценка у BM25), поэтому подбирается под поиск на dev
    min_retrieval_score: float | None = None
    refusal: str = "strict"           # правило отказа в промпте: strict или lenient
    # один повторный запрос, если модель ответила одними номерами фрагментов
    # (на dev: 8 из 36 вопросов с ответом без повтора получали отказ)
    retry_empty: bool = True


class RAGPipeline:
    def __init__(self, retriever, llm: LLMBackend, config: GenerationConfig | None = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or GenerationConfig()

    def retrieve(self, question: str):
        return self.retriever.search(question, top_k=self.config.top_k)

    def answer(self, question: str, on_token=None) -> Answer:
        timings = {}
        started = time.perf_counter()
        hits = self.retrieve(question)
        timings["retrieval_s"] = round(time.perf_counter() - started, 3)
        return self.answer_from_hits(question, hits, timings, on_token=on_token)

    def refuse(self, question: str, reason: str, hits, timings: dict, raw: str = "", **extra) -> Answer:
        return Answer(
            question=question, answer=NO_ANSWER_TEXT, sources=[],
            context=extra.pop("context", []), refused=True, refusal_reason=reason,
            raw_answer=raw, backend=self.llm.name, is_mock=self.llm.is_mock, timings=timings,
            top_score=round(hits[0].score, 4) if hits else None, **extra,
        )

    def answer_from_hits(self, question: str, hits, timings: dict | None = None, on_token=None) -> Answer:
        timings = dict(timings or {})
        threshold = self.config.min_retrieval_score
        if not hits or (threshold is not None and hits[0].score < threshold):
            return self.refuse(question, "retrieval", hits, timings)

        prompt = build_prompt(question, hits, self.llm.count_tokens, self.config.max_context_tokens,
                              refusal=self.config.refusal)
        started = time.perf_counter()
        if hasattr(self.llm, "stream_chat"):
            parts = []
            for part in self.llm.stream_chat(prompt.messages, max_tokens=self.config.max_answer_tokens):
                if not parts:
                    timings["first_token_s"] = round(time.perf_counter() - started, 3)
                    timings["first_token_total_s"] = round(timings.get("retrieval_s", 0) + timings["first_token_s"], 3)
                parts.append(part)
                if on_token:
                    on_token(part)
            raw = "".join(parts).strip()
        else:
            raw = self.llm.chat(prompt.messages, max_tokens=self.config.max_answer_tokens)
            if on_token:
                on_token(raw)
        timings["generation_s"] = round(time.perf_counter() - started, 3)
        reply = read_reply(raw)
        retry = {}
        if reply.refusal == "empty" and self.config.retry_empty:
            started = time.perf_counter()
            messages = prompt.messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": RETRY_EMPTY}]
            retry = {"retried": True, "first_raw_answer": raw}
            raw = self.llm.chat(messages, max_tokens=self.config.max_answer_tokens)
            timings["retry_s"] = round(time.perf_counter() - started, 3)
            reply = read_reply(raw)

        context = [h.chunk.chunk_id for h in prompt.sources]
        # отказ не сопровождается источниками, даже если модель поставила ссылку
        if reply.refusal:
            return self.refuse(
                question, reply.refusal, hits, timings, raw=raw, context=context,
                invalid_citations=check_citations(raw, len(prompt.sources)).invalid,
                context_tokens=prompt.context_tokens, **retry,
            )
        check = check_citations(reply.text, len(prompt.sources))
        text = strip_invalid_citations(reply.text, check.invalid)
        sources = []
        for number in check.valid:
            hit = prompt.sources[number - 1]
            chunk = hit.chunk
            sources.append(Source(number, chunk.chunk_id, chunk.doc_id, chunk.title,
                                  " > ".join(chunk.section_path), chunk.url, hit.rank, round(hit.score, 4)))
        return Answer(
            question=question,
            answer=text,
            sources=sources,
            context=context,
            invalid_citations=check.invalid,
            unsupported=not sources,
            partial_refusal=reply.partial_refusal,
            raw_answer=raw,
            **retry,
            top_score=round(hits[0].score, 4),
            dropped_for_limit=[h.chunk.chunk_id for h in prompt.dropped],
            backend=self.llm.name,
            is_mock=self.llm.is_mock,
            timings=timings,
            context_tokens=prompt.context_tokens,
        )


def format_answer(answer: Answer) -> str:
    lines = [answer.answer, ""]
    if answer.sources:
        lines.append("Источники:")
        for source in answer.sources:
            section = f" — {source.section}" if source.section else ""
            lines.append(f"[{source.number}] {source.title}{section}\n    {source.url}")
    elif answer.unsupported:
        lines.append("Модель не сослалась ни на один фрагмент: ответ не подтверждён документами.")
    else:
        lines.append("Источники: нет")
    if answer.is_mock:
        lines.append("\n(ответ mock-бэкенда: проверка связности, не модель)")
    return "\n".join(lines)
