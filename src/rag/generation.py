"""Ответ по найденному контексту: поиск → промпт с лимитом → LLM → ответ и источники."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .citations import check_citations, strip_invalid_citations
from .llm import LLMBackend
from .prompt import build_prompt


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


class RAGPipeline:
    def __init__(self, retriever, llm: LLMBackend, config: GenerationConfig | None = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or GenerationConfig()

    def retrieve(self, question: str):
        return self.retriever.search(question, top_k=self.config.top_k)

    def answer(self, question: str) -> Answer:
        timings = {}
        started = time.perf_counter()
        hits = self.retrieve(question)
        timings["retrieval_s"] = round(time.perf_counter() - started, 3)
        return self.answer_from_hits(question, hits, timings)

    def answer_from_hits(self, question: str, hits, timings: dict | None = None) -> Answer:
        timings = dict(timings or {})
        prompt = build_prompt(question, hits, self.llm.count_tokens, self.config.max_context_tokens)
        started = time.perf_counter()
        raw = self.llm.chat(prompt.messages, max_tokens=self.config.max_answer_tokens)
        timings["generation_s"] = round(time.perf_counter() - started, 3)

        check = check_citations(raw, len(prompt.sources))
        text = strip_invalid_citations(raw, check.invalid)
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
            context=[h.chunk.chunk_id for h in prompt.sources],
            invalid_citations=check.invalid,
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
    else:
        lines.append("Источники: нет")
    if answer.is_mock:
        lines.append("\n(ответ mock-бэкенда: проверка связности, не модель)")
    return "\n".join(lines)
